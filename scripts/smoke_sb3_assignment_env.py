# ============================================================
# FILE: scripts/smoke_sb3_assignment_env.py
# ============================================================
"""Smoke test for GraphAssignmentEnv without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/...` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tarware_ext.sb3 import GraphAssignmentConfig, GraphAssignmentEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--obs-backend", choices=["assignment", "graph"], default="assignment")
    p.add_argument("--max-request-slots", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None, help="Deprecated alias of --max-request-slots")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    max_request_slots = args.max_request_slots if args.max_request_slots is not None else args.top_k

    env = GraphAssignmentEnv(
        GraphAssignmentConfig(
            env_id=args.env_id,
            obs_backend=args.obs_backend,
            max_request_slots=max_request_slots,
            max_steps=args.steps,
            seed=args.seed,
            verbose=args.verbose,
        )
    )

    obs, info = env.reset(seed=args.seed)
    print("obs shape:", obs.shape, "| info keys:", list(info.keys()) if isinstance(info, dict) else type(info))

    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if t < 3 or t == args.steps - 1:
            print(f"t={t:03d} action={action.tolist()} reward={reward:.3f} term={terminated} trunc={truncated}")
        if terminated or truncated:
            print("Episode ended early.")
            break

    env.close()


if __name__ == "__main__":
    main()
