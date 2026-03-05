"""Minimal training data collection & imitation scaffold for GraphRL.

This script collects (GraphState, chosen_task_index) pairs using the
`GraphScorePolicy` as teacher and saves them to a pickle file. The purpose is
to provide a minimal dataset and a place to plug-in a training loop later.

Usage (example):
    .venv/bin/python scripts/train_graph_rl.py --env-id tarware-small-2agvs-1pickers-globalobs-v1 --episodes 5 --steps 200 --out data/imit_data.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym

from tarware_ext.graphs.builder_v0 import GraphBuilderV0
from tarware_ext.policies import GraphScorePolicy
from tarware_ext.envs import TarwareAdapter


def sample_dataset(env_id: str, episodes: int, steps: int, out_path: Path) -> None:
    samples: List[Dict[str, Any]] = []
    for ep in range(episodes):
        env = TarwareAdapter(gym.make(env_id))
        builder = GraphBuilderV0()
        policy = GraphScorePolicy(distance_mode="manhattan", assigner="greedy")
        obs, _ = env.reset()
        policy.reset(env.unwrapped if hasattr(env, "unwrapped") else env)
        for _ in range(steps):
            # Build GraphState from the unwrapped env to access internals
            target_env = env.unwrapped if hasattr(env, "unwrapped") else env
            g = builder.build(target_env)
            # teacher actions (loc_id per agent)
            actions = policy.act(env)
            # convert actions loc_id -> task_idx (or -1)
            task_indices = []
            for a in actions:
                if int(a) == 0:
                    task_indices.append(-1)
                    continue
                try:
                    task_idx = g.task_loc_ids.index(int(a))
                except ValueError:
                    task_idx = -1
                task_indices.append(int(task_idx))

            sample = {
                "node_features": g.node_features.copy(),
                "edge_index": g.edge_index.copy(),
                "agent_node_ids": list(g.agent_node_ids),
                "task_node_ids": list(g.task_node_ids),
                "task_loc_ids": list(g.task_loc_ids),
                "action_mask": None if g.action_mask is None else g.action_mask.copy(),
                "metadata": dict(g.metadata) if g.metadata is not None else {},
                "teacher_task_indices": list(task_indices),
            }
            samples.append(sample)
            # step with teacher actions
            env.step(actions)
        env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", default="data/imit_data.pkl")
    args = parser.parse_args()

    sample_dataset(args.env_id, args.episodes, args.steps, Path(args.out))


if __name__ == "__main__":
    main()
"""Placeholder for future graph-based RL training."""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Training entrypoint will be added once graph builder/policy is ready.")


if __name__ == "__main__":
    main()
