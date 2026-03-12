# ============================================================
# FILE: tarware_ext/sb3/graph_sb3_env.py
# ============================================================
"""
Graph-wrapped Gymnasium env for SB3 PPO (MVP).

- Under the hood: TA-RWARE (gym.make) wrapped by TarwareAdapter.
- Each reset/step: build GraphState using GraphBuilderV0(top_k=K).
- Observation: fixed vector using encode_graph_obs (feature engineering).
- Action: MultiDiscrete([K+1]*num_agvs), each AGV selects a candidate task:
    0 = idle
    1..K = pick candidate index
  Internally this is translated to TA-RWARE macro actions (loc_id per agent).

We start controlling only AGVs. Pickers stay idle (0) for MVP simplicity.

This version adds:
- Better debug logs (action -> task_idx -> loc_id) for first steps.
- Episode MRTA metrics in info["episode"] so SB3 logs are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import tarware  # noqa: F401  # registers envs

from tarware_ext.envs.tarware_adapter import TarwareAdapter, Transition
from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.graphs.schema import GraphState, NodeType
from tarware_ext.sb3.graph_obs_encoder import GraphObsSpec, encode_graph_obs


@dataclass
class GraphSB3Config:
    env_id: str
    top_k: int = 2
    max_steps: int = 200
    seed: Optional[int] = None
    distance_mode: str = "manhattan"
    verbose: bool = False
    debug_first_n_steps: int = 3  # why: avoid flooding console


class GraphSB3Env(gym.Env):
    """Single-agent interface controlling multiple agents internally (centralized control)."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: GraphSB3Config) -> None:
        super().__init__()
        self.cfg = config
        self._t = 0

        base = gym.make(self.cfg.env_id)
        self.env = TarwareAdapter(base)
        self.builder = GraphBuilderV0(distance_mode=self.cfg.distance_mode, top_k=self.cfg.top_k)

        # One reset to infer counts and build initial graph snapshot
        _obs, _info = self.env.reset(seed=self.cfg.seed)
        target_env = self._unwrap_env()
        g = self.builder.build(target_env)

        self.num_agents = int(g.metadata.get("num_agents", len(getattr(target_env, "agents", [])) or 1))
        self.num_agvs = int(g.metadata.get("num_agvs", 0))
        if self.num_agvs <= 0:
            self.num_agvs = sum(1 for nid in g.agent_node_ids if g.node_types[int(nid)] == NodeType.AGV)
        self.num_agvs = max(1, self.num_agvs)

        self.obs_spec = GraphObsSpec(num_agvs=self.num_agvs, top_k=self.cfg.top_k)

        self.observation_space = gym.spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(self.obs_spec.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([self.cfg.top_k + 1] * self.num_agvs)

        self._last_graph: Optional[GraphState] = g
        self._last_obs: np.ndarray = encode_graph_obs(g, self.obs_spec)

        # Episode accumulators (for SB3 "episode" info + MRTA metrics)
        self._ep_return: float = 0.0
        self._ep_len: int = 0
        self._ep_deliveries: int = 0
        self._ep_clashes: int = 0
        self._ep_stucks: int = 0

        # Debug: keep last mapping action -> (task_idx, loc_id)
        self._last_debug_mapping: Dict[str, Any] = {}

    def _unwrap_env(self) -> Any:
        """Extract the underlying Warehouse-like env."""
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

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_deliveries = 0
        self._ep_clashes = 0
        self._ep_stucks = 0
        self._last_debug_mapping = {}

        _obs, info = self.env.reset(seed=seed if seed is not None else self.cfg.seed, options=options)

        g = self.builder.build(self._unwrap_env())
        self._last_graph = g
        self._last_obs = encode_graph_obs(g, self.obs_spec)

        if self.cfg.verbose:
            print(f"[reset] num_tasks={len(g.task_node_ids)} top_k={self.cfg.top_k}")

        return self._last_obs.copy(), info if isinstance(info, dict) else {}

    def _translate_action_to_loc_ids(self, action: np.ndarray) -> List[int]:
        """
        Convert SB3 action (per-AGV candidate choice) to TA-RWARE actions (loc_id per agent).

        Also stores debug mapping in self._last_debug_mapping:
          sb3_action -> chosen task_idx -> chosen loc_id (per AGV)
        """
        self._last_debug_mapping = {
            "sb3_action": [int(x) for x in np.asarray(action).tolist()],
            "chosen_tasks": [],
            "chosen_loc_ids": [],
        }

        if self._last_graph is None:
            return [0 for _ in range(self.num_agents)]

        g = self._last_graph
        top_k_candidates = g.metadata.get("top_k_candidates") if g.metadata else None
        if top_k_candidates is None:
            return [0 for _ in range(self.num_agents)]

        actions_all = [0 for _ in range(self.num_agents)]

        # Assumption (MVP): g.agent_node_ids order matches env.agents order
        agv_counter = 0
        for agent_idx, nid in enumerate(g.agent_node_ids):
            if g.node_types[int(nid)] != NodeType.AGV:
                continue
            if agv_counter >= self.num_agvs:
                break

            choice = int(action[agv_counter])
            chosen_task = None
            chosen_loc = 0

            if choice > 0:
                cand_list = top_k_candidates[agv_counter] if agv_counter < len(top_k_candidates) else []
                cand_pos = choice - 1
                if 0 <= cand_pos < len(cand_list):
                    task_idx = int(cand_list[cand_pos])
                    if 0 <= task_idx < len(g.task_loc_ids):
                        chosen_task = task_idx
                        chosen_loc = int(g.task_loc_ids[task_idx])
                        actions_all[agent_idx] = chosen_loc

            self._last_debug_mapping["chosen_tasks"].append(chosen_task)
            self._last_debug_mapping["chosen_loc_ids"].append(int(chosen_loc))
            agv_counter += 1

        # Pickers: idle in MVP (0). Later you can add a heuristic here.
        return actions_all

    def _update_episode_counters(self, reward: float, info: Dict[str, Any]) -> None:
        self._ep_len += 1
        self._ep_return += float(reward)

        # Align with run_heuristic.py semantics if keys exist
        if "shelf_deliveries" in info:
            self._ep_deliveries += int(info.get("shelf_deliveries", 0))
        if "clashes" in info:
            self._ep_clashes += int(info.get("clashes", 0))
        if "stucks" in info:
            self._ep_stucks += int(info.get("stucks", 0))

    def _attach_episode_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        # SB3 logs episode info if present under key "episode"
        # We include MRTA metrics for interpretability.
        episode_length = max(self._ep_len, 1)
        pick_rate = float(self._ep_deliveries) * 3600.0 / (5.0 * float(episode_length))

        info = dict(info) if isinstance(info, dict) else {}
        info["episode"] = {
            # SB3 standard keys
            "r": float(self._ep_return),
            "l": int(self._ep_len),
            # Extra MRTA metrics
            "deliveries": int(self._ep_deliveries),
            "clashes": int(self._ep_clashes),
            "stucks": int(self._ep_stucks),
            "pick_rate": float(pick_rate),
        }
        return info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        loc_actions = self._translate_action_to_loc_ids(action)

        step_out = self.env.step(loc_actions)

        if isinstance(step_out, Transition):
            reward = float(step_out.reward_team)
            terminated = bool(step_out.done_all)
            truncated = False
            info = step_out.info if isinstance(step_out.info, dict) else {}
        else:
            _obs, reward_raw, terminated_raw, truncated_raw, info = step_out
            reward = float(np.sum(reward_raw)) if isinstance(reward_raw, (list, tuple, np.ndarray)) else float(reward_raw)
            terminated = bool(terminated_raw)
            truncated = bool(truncated_raw)
            info = info if isinstance(info, dict) else {}

        # Episode accumulators (so we can log MRTA metrics per episode)
        self._update_episode_counters(reward=reward, info=info)

        # Update graph snapshot and obs
        g = self.builder.build(self._unwrap_env())
        self._last_graph = g
        obs_vec = encode_graph_obs(g, self.obs_spec)
        self._last_obs = obs_vec

        # Time truncation
        if self._t >= self.cfg.max_steps:
            truncated = True

        # Debug: mapping action->task->loc_id on first steps
        if self.cfg.verbose and self._t <= self.cfg.debug_first_n_steps:
            print(
                f"[map] t={self._t} sb3_action={self._last_debug_mapping.get('sb3_action')} "
                f"chosen_tasks={self._last_debug_mapping.get('chosen_tasks')} "
                f"chosen_loc_ids={self._last_debug_mapping.get('chosen_loc_ids')}"
            )

        # End-of-episode: attach info["episode"] for SB3 logs
        if terminated or truncated:
            info = self._attach_episode_info(info)

        if self.cfg.verbose and (self._t <= 3 or terminated or truncated):
            print(
                f"[step] t={self._t} reward={reward:.3f} term={terminated} trunc={truncated} "
                f"tasks={len(g.task_node_ids)}"
            )
            if (terminated or truncated) and isinstance(info, dict) and "episode" in info:
                ep = info["episode"]
                print(
                    f"[episode] len={ep.get('l')} return={ep.get('r'):.3f} "
                    f"deliveries={ep.get('deliveries')} clashes={ep.get('clashes')} "
                    f"stucks={ep.get('stucks')} pick_rate={ep.get('pick_rate'):.2f}"
                )

        return obs_vec.copy(), reward, terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()