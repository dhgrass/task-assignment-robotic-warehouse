# ============================================================
# FILE: tarware_ext/sb3/graph_assignment_env.py
# ============================================================
"""SB3 env for explicit AGV assignment with heuristic mission parity.

Design:
- SB3 controls only new AGV->request assignments (explicit indices).
- A stateful heuristic controller executes full mission lifecycle for AGVs and
  pickers: PICKING -> DELIVERING -> RETURNING.
- Exposes a standard single-agent Gymnasium API for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings

import gymnasium as gym
import numpy as np

import tarware  # noqa: F401

from tarware_ext.controllers import HeuristicController
from tarware_ext.envs.tarware_adapter import TarwareAdapter, Transition
from tarware_ext.graphs import AssignmentGraphBuilder
from tarware_ext.sb3.assignment_obs_encoder import encode_assignment_obs
from tarware_ext.sb3.graph_assignment_obs_encoder import encode_graph_assignment_obs
from tarware.definitions import AgentType


@dataclass
class GraphAssignmentConfig:
    env_id: str
    max_request_slots: Optional[int] = None
    top_k: Optional[int] = None  # Deprecated alias for max_request_slots.
    max_steps: int = 200
    seed: Optional[int] = None
    distance_mode: str = "manhattan"
    obs_backend: str = "assignment"  # "assignment" (A) or "graph" (B)
    graph_encoder_mode: str = "manual"  # "manual" or "gnn" when obs_backend="graph"
    verbose: bool = False
    debug_first_n_steps: int = 3


class GraphAssignmentEnv(gym.Env):
    """Single-agent wrapper where the action is explicit AGV->item assignment."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: GraphAssignmentConfig) -> None:
        super().__init__()
        self.cfg = config
        self._t = 0

        self._obs_backend = str(self.cfg.obs_backend).strip().lower()
        if self._obs_backend not in ("assignment", "graph"):
            raise ValueError(
                "GraphAssignmentConfig.obs_backend must be 'assignment' or 'graph'."
            )

        self._graph_encoder_mode = str(self.cfg.graph_encoder_mode).strip().lower()
        if self._graph_encoder_mode not in ("manual", "gnn"):
            raise ValueError("GraphAssignmentConfig.graph_encoder_mode must be 'manual' or 'gnn'.")

        self._agv_feat_dim = 6
        self._slot_feat_dim = 7
        self._global_feat_dim = 4

        base = gym.make(self.cfg.env_id, disable_env_checker=True)
        # Keep per-agent done semantics so we can build proper terminated/truncated.
        self.env = TarwareAdapter(base, done_all=False)
        self.controller = HeuristicController()
        self.graph_builder = AssignmentGraphBuilder()

        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_deliveries = 0
        self._ep_clashes = 0
        self._ep_stucks = 0

        _obs, _info = self.env.reset(seed=self.cfg.seed)
        unwrapped = self._unwrap_env()
        self.controller.reset(unwrapped, seed=self.cfg.seed)

        agents = list(getattr(unwrapped, "agents", []))
        self.num_agents = len(agents) or 1
        self.num_agvs = sum(1 for a in agents if getattr(a, "type", None) == AgentType.AGV)
        self.num_agvs = max(1, int(self.num_agvs))

        configured_slots = self.cfg.max_request_slots
        if configured_slots is None and self.cfg.top_k is not None:
            warnings.warn(
                "GraphAssignmentConfig.top_k is deprecated; use max_request_slots instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            configured_slots = int(self.cfg.top_k)

        if configured_slots is None:
            inferred_tasks = int(getattr(unwrapped, "request_queue_size", 0) or 0)
            if inferred_tasks <= 0:
                inferred_tasks = len(getattr(unwrapped, "request_queue", []))
            configured_slots = max(1, int(inferred_tasks))

        self.max_request_slots = max(1, int(configured_slots))

        obs_dim = (
            self.num_agvs * (self._agv_feat_dim + self.max_request_slots * self._slot_feat_dim)
            + self._global_feat_dim
        )
        self.observation_space = gym.spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Explicit AGV assignment: 0=no assignment, 1..R=request_queue slot.
        self.action_space = gym.spaces.MultiDiscrete([self.max_request_slots + 1] * self.num_agvs)

        self._last_obs: np.ndarray = self._encode_obs(unwrapped)

    def _unwrap_env(self) -> Any:
        cand: Any = self.env
        for _ in range(6):
            if hasattr(cand, "unwrapped") and getattr(cand, "unwrapped") is not cand:
                cand = getattr(cand, "unwrapped")
                continue
            if hasattr(cand, "env") and getattr(cand, "env") is not cand:
                cand = getattr(cand, "env")
                continue
            break
        return cand

    def _update_episode_counters(self, reward: float, info: Dict[str, Any]) -> None:
        self._ep_len += 1
        self._ep_return += float(reward)
        if "shelf_deliveries" in info:
            self._ep_deliveries += int(info.get("shelf_deliveries", 0))
        if "clashes" in info:
            self._ep_clashes += int(info.get("clashes", 0))
        if "stucks" in info:
            self._ep_stucks += int(info.get("stucks", 0))

    def _encode_obs(self, env: Any) -> np.ndarray:
        if self._obs_backend == "graph":
            graph = self.graph_builder.build(env, controller=self.controller)
            return encode_graph_assignment_obs(
                graph,
                max_request_slots=self.max_request_slots,
                num_agvs=self.num_agvs,
                obs_shape=self.observation_space.shape,
                agv_feat_dim=self._agv_feat_dim,
                slot_feat_dim=self._slot_feat_dim,
                global_feat_dim=self._global_feat_dim,
                encoder_mode=self._graph_encoder_mode,
            )

        return encode_assignment_obs(
            env,
            controller=self.controller,
            max_request_slots=self.max_request_slots,
            num_agvs=self.num_agvs,
            obs_shape=self.observation_space.shape,
            agv_feat_dim=self._agv_feat_dim,
            slot_feat_dim=self._slot_feat_dim,
            global_feat_dim=self._global_feat_dim,
        )

    def _attach_episode_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        episode_length = max(self._ep_len, 1)
        pick_rate = float(self._ep_deliveries) * 3600.0 / (5.0 * float(episode_length))

        out = dict(info) if isinstance(info, dict) else {}
        out["episode"] = {
            "r": float(self._ep_return),
            "l": int(self._ep_len),
            "deliveries": int(self._ep_deliveries),
            "clashes": int(self._ep_clashes),
            "stucks": int(self._ep_stucks),
            "pick_rate": float(pick_rate),
        }
        return out

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_deliveries = 0
        self._ep_clashes = 0
        self._ep_stucks = 0

        _obs, info = self.env.reset(seed=seed if seed is not None else self.cfg.seed, options=options)
        unwrapped = self._unwrap_env()
        self.controller.reset(unwrapped, seed=seed if seed is not None else self.cfg.seed)

        self._last_obs = self._encode_obs(unwrapped)

        if self.cfg.verbose:
            print(
                f"[reset] num_tasks={len(getattr(unwrapped, 'request_queue', []))} "
                f"num_agvs={self.num_agvs} request_slots={self.max_request_slots}"
            )

        return self._last_obs.copy(), info if isinstance(info, dict) else {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._t += 1

        assignments = [int(x) for x in np.asarray(action).reshape(-1).tolist()]
        unwrapped = self._unwrap_env()
        loc_actions = self.controller.step(unwrapped, rl_agv_assignments=assignments)

        if self.cfg.verbose and self._t <= self.cfg.debug_first_n_steps:
            print(f"[assignment] t={self._t} rl={assignments} env_actions={loc_actions}")

        step_out = self.env.step(loc_actions)

        if isinstance(step_out, Transition):
            reward = float(step_out.reward_team)
            terminated = bool(all(step_out.terminated_by_agent))
            truncated = bool(step_out.done_all and not terminated)
            info = step_out.info if isinstance(step_out.info, dict) else {}
        else:
            _obs, reward_raw, terminated_raw, truncated_raw, info_raw = step_out
            reward = float(np.sum(reward_raw)) if isinstance(reward_raw, (list, tuple, np.ndarray)) else float(reward_raw)
            if isinstance(terminated_raw, (list, tuple, np.ndarray)):
                terminated = bool(all(bool(x) for x in terminated_raw))
            else:
                terminated = bool(terminated_raw)
            if isinstance(truncated_raw, (list, tuple, np.ndarray)):
                done_all = bool(all(bool(t) or bool(tr) for t, tr in zip(terminated_raw, truncated_raw)))
                truncated = bool(done_all and not terminated)
            else:
                truncated = bool(truncated_raw)
            info = info_raw if isinstance(info_raw, dict) else {}

        self._update_episode_counters(reward=reward, info=info)

        obs_vec = self._encode_obs(self._unwrap_env())
        self._last_obs = obs_vec

        if self._t >= self.cfg.max_steps:
            truncated = True

        if terminated or truncated:
            info = self._attach_episode_info(info)

        if self.cfg.verbose and (self._t <= 3 or terminated or truncated):
            print(
                f"[step] t={self._t} reward={reward:.3f} term={terminated} trunc={truncated} "
                f"tasks={len(getattr(self._unwrap_env(), 'request_queue', []))}"
            )

        return obs_vec.copy(), reward, terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
