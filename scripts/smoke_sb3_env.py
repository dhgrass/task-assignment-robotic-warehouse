# ============================================================
# FILE: scripts/smoke_sb3_env.py
# ============================================================
"""
Smoke test: GraphSB3Env reset/step without training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/...` without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tarware_ext.sb3.graph_sb3_env import GraphSB3Env, GraphSB3Config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", required=True)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    env = GraphSB3Env(
        GraphSB3Config(
            env_id=args.env_id,
            top_k=args.top_k,
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
            print(f"t={t:03d} action={action} reward={reward:.3f} term={terminated} trunc={truncated}")
        if terminated or truncated:
            print("Episode ended early.")
            break

    env.close()


if __name__ == "__main__":
    main()