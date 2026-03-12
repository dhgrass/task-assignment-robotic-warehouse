# ============================================================
# FILE: tarware_ext/sb3/graph_sb3_env.py
# ============================================================
"""
Graph-wrapped Gymnasium env for SB3 PPO (MVP).

SB3 controls only AGVs (RL). Pickers follow a simple step-wise heuristic:
they move to the same loc_id targets selected for AGVs, using rack sections.

Why: If pickers stay idle in tarware-small-* envs, deliveries usually remain 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import tarware  # noqa: F401

from tarware_ext.envs.tarware_adapter import TarwareAdapter, Transition
from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.graphs.schema import GraphState, NodeType

# Reuse the same split helpers used by GraphGreedyPolicy (step-wise heuristic)
from tarware.utils.utils import flatten_list, split_list  # type: ignore

from tarware_ext.sb3.graph_obs_encoder import GraphObsSpec, encode_graph_obs


@dataclass
class GraphSB3Config:
    env_id: str
    top_k: int = 2
    max_steps: int = 200
    seed: Optional[int] = None
    distance_mode: str = "manhattan"
    verbose: bool = False
    debug_first_n_steps: int = 3  # evita spamear consola


class GraphSB3Env(gym.Env):
    """
    Single-agent interface controlling multiple agents internally (centralized control).

    Action space (SB3): MultiDiscrete([K+1]*num_agvs)
      - 0 = idle
      - 1..K = choose candidate task
    Internally we translate to TA-RWARE actions: loc_id per agent (AGVs + pickers).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: GraphSB3Config) -> None:
        super().__init__()
        self.cfg = config
        self._t = 0

        base = gym.make(self.cfg.env_id, disable_env_checker=True)  # reduce warnings
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

        # For picker heuristic (zones/sections)
        self._agents: List[Any] = []
        self._pickers: List[Any] = []
        self._picker_sections: List[List[Tuple[int, int]]] = []
        self._loc_to_yx: Dict[int, Tuple[int, int]] = {}

        # Episode counters (optional; you already added in your previous patch)
        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_deliveries = 0
        self._ep_clashes = 0
        self._ep_stucks = 0

        self._last_debug_mapping: Dict[str, Any] = {}

        # Build initial picker sections
        self._refresh_picker_layout(target_env)

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

    def _refresh_picker_layout(self, unwrapped_env: Any) -> None:
        """
        Precompute:
          - list of pickers
          - rack sections per picker (like GraphGreedyPolicy)
          - loc_id -> (y,x) map
        """
        self._agents = list(getattr(unwrapped_env, "agents", []))
        self._pickers = [a for a in self._agents if getattr(a, "type", None).name == "PICKER"]  # AgentType.PICKER

        self._loc_to_yx = dict(getattr(unwrapped_env, "action_id_to_coords_map", {}))

        sections = list(getattr(unwrapped_env, "rack_groups", []))
        if not self._pickers:
            self._picker_sections = []
            return

        picker_sections = split_list(sections, max(1, len(self._pickers)))
        picker_sections = [flatten_list(l) for l in picker_sections]
        # Each section is a list of coords (y,x)
        self._picker_sections = picker_sections

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        self._ep_return = 0.0
        self._ep_len = 0
        self._ep_deliveries = 0
        self._ep_clashes = 0
        self._ep_stucks = 0
        self._last_debug_mapping = {}

        _obs, info = self.env.reset(seed=seed if seed is not None else self.cfg.seed, options=options)

        unwrapped = self._unwrap_env()
        self._refresh_picker_layout(unwrapped)

        g = self.builder.build(unwrapped)
        self._last_graph = g
        self._last_obs = encode_graph_obs(g, self.obs_spec)

        if self.cfg.verbose:
            print(f"[reset] num_tasks={len(g.task_node_ids)} top_k={self.cfg.top_k}")

        return self._last_obs.copy(), info if isinstance(info, dict) else {}

    def _assign_pickers_to_agv_targets(self, actions_all: List[int]) -> None:
        """
        Simple step-wise picker heuristic:
        - For each AGV target loc_id != 0, send the corresponding zone picker to same loc_id.
        - If multiple AGVs target same zone, first one wins.
        """
        if not self._pickers or not self._picker_sections:
            return

        # Track which pickers already assigned this step
        used_picker_idxs: set[int] = set()

        # Determine AGV targets (loc_id) from actions_all
        for agent_idx, loc_id in enumerate(actions_all):
            if loc_id <= 0:
                continue

            yx = self._loc_to_yx.get(int(loc_id))
            if yx is None:
                continue

            # Find picker zone for this (y,x)
            picker_idx = None
            for i, section in enumerate(self._picker_sections):
                if (int(yx[0]), int(yx[1])) in section:
                    picker_idx = i
                    break

            if picker_idx is None or picker_idx in used_picker_idxs:
                continue

            # Map picker index to env agent index: assume pickers appear in env.agents order
            picker_agent = self._pickers[picker_idx]
            try:
                picker_agent_idx = self._agents.index(picker_agent)
            except ValueError:
                continue

            actions_all[picker_agent_idx] = int(loc_id)
            used_picker_idxs.add(picker_idx)

    def _translate_action_to_loc_ids(self, action: np.ndarray) -> List[int]:
        """
        SB3 action -> loc_id per env agent.
        - AGVs from SB3 (top-k mapping)
        - Pickers from heuristic that follows AGV targets (zone-based)
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

        # AGVs from SB3
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

        # Pickers follow AGV targets (heuristic)
        self._assign_pickers_to_agv_targets(actions_all)
        return actions_all

    def _update_episode_counters(self, reward: float, info: Dict[str, Any]) -> None:
        self._ep_len += 1
        self._ep_return += float(reward)
        if "shelf_deliveries" in info:
            self._ep_deliveries += int(info.get("shelf_deliveries", 0))
        if "clashes" in info:
            self._ep_clashes += int(info.get("clashes", 0))
        if "stucks" in info:
            self._ep_stucks += int(info.get("stucks", 0))

    def _attach_episode_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        episode_length = max(self._ep_len, 1)
        pick_rate = float(self._ep_deliveries) * 3600.0 / (5.0 * float(episode_length))

        info = dict(info) if isinstance(info, dict) else {}
        info["episode"] = {
            "r": float(self._ep_return),
            "l": int(self._ep_len),
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

        self._update_episode_counters(reward=reward, info=info)

        unwrapped = self._unwrap_env()
        g = self.builder.build(unwrapped)
        self._last_graph = g
        obs_vec = encode_graph_obs(g, self.obs_spec)
        self._last_obs = obs_vec

        if self._t >= self.cfg.max_steps:
            truncated = True

        if self.cfg.verbose and self._t <= self.cfg.debug_first_n_steps:
            print(
                f"[map] t={self._t} sb3_action={self._last_debug_mapping.get('sb3_action')} "
                f"chosen_tasks={self._last_debug_mapping.get('chosen_tasks')} "
                f"chosen_loc_ids={self._last_debug_mapping.get('chosen_loc_ids')}"
            )

        if terminated or truncated:
            info = self._attach_episode_info(info)

        if self.cfg.verbose and (self._t <= 3 or terminated or truncated):
            print(
                f"[step] t={self._t} reward={reward:.3f} term={terminated} trunc={truncated} "
                f"tasks={len(g.task_node_ids)}"
            )
            if (terminated or truncated) and "episode" in info:
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