"""Evaluate policies on TA-RWARE envs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tarware  # noqa: F401
from tarware_ext.envs import TarwareAdapter
from tarware_ext.logs import CSVLogger
from tarware_ext.policies import DistanceMode, GraphGreedyPolicy, HeuristicPolicy, RandomPolicy
from tarware_ext.runners import evaluate


def _make_env(env_id: str) -> Callable[[], TarwareAdapter]:
    def _factory() -> TarwareAdapter:
        env = gym.make(env_id)
        return TarwareAdapter(env)

    return _factory


def _build_policy(name: str, env: TarwareAdapter, distance: str | None = None):
    if name == "random":
        return RandomPolicy(env)
    if name == "heuristic":
        return HeuristicPolicy(env)
    if name == "graph_greedy":
        mode = DistanceMode(distance or DistanceMode.MANHATTAN.value)
        return GraphGreedyPolicy(distance_mode=mode)
    raise ValueError(f"Unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--policy", choices=["random", "heuristic", "graph_greedy"], default="random")
    parser.add_argument("--distance", choices=["manhattan", "find_path"], default="manhattan")
    parser.add_argument("--active-alpha", type=int, default=3)
    parser.add_argument("--max-active-agvs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv", default="eval.csv")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    env = TarwareAdapter(gym.make(args.env_id))
    if args.policy == "graph_greedy":
        policy = _build_policy(
            args.policy,
            env,
            distance=args.distance,
        )
        policy.active_alpha = args.active_alpha
        if args.max_active_agvs is not None:
            policy.max_active_agvs = args.max_active_agvs
    else:
        policy = _build_policy(args.policy, env, distance=args.distance)
    env.close()

    eval_fn = _make_env(args.env_id)
    results = evaluate(
        eval_fn,
        policy,
        episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
    )
    if not results:
        return

    summary = results["summary"]
    episodes = results["episodes"]

    print(
        " | ".join(
            [
                f"episodes={int(summary['episodes'])}",
                f"mean_return={summary['mean_return']:.2f}",
                f"mean_deliveries={summary['mean_deliveries']:.2f}",
                f"mean_clashes={summary['mean_clashes']:.2f}",
                f"mean_stuck={summary['mean_stuck']:.2f}",
                f"mean_pick_rate={summary['mean_pick_rate']:.2f}",
                f"overall_pick_rate={summary['overall_pick_rate']:.2f}",
                f"mean_episode_length={summary['mean_episode_length']:.2f}",
                f"mean_fps={summary['mean_fps']:.2f}",
            ]
        )
    )

    if not args.no_csv:
        fieldnames = [
            "episode",
            "seed",
            "env_id",
            "distance_mode",
            "active_alpha",
            "max_active_agvs",
            "episode_length",
            "shelf_deliveries",
            "clashes",
            "stucks",
            "global_episode_return",
            "pick_rate",
            "fps",
        ]
        logger = CSVLogger(args.csv, fieldnames=fieldnames)
        for row in episodes:
            enriched = dict(row)
            enriched["env_id"] = args.env_id
            enriched["distance_mode"] = args.distance
            enriched["active_alpha"] = args.active_alpha
            enriched["max_active_agvs"] = args.max_active_agvs
            logger.log({key: enriched.get(key) for key in fieldnames})
        logger.close()


if __name__ == "__main__":
    main()
