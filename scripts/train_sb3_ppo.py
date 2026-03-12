# ============================================================
# FILE: scripts/train_sb3_ppo.py
# ============================================================
"""
Train SB3 PPO on GraphSB3Env (MVP).

SB3 learns an MLP policy over the fixed observation vector.
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
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--out", type=str, default="data/sb3_ppo_model")
    args = p.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError("Instala SB3 con: pip install stable-baselines3") from exc

    env = GraphSB3Env(
        GraphSB3Config(
            env_id=args.env_id,
            top_k=args.top_k,
            max_steps=args.steps,
            seed=args.seed,
            verbose=True,
        )
    )

    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    env.close()
    print("Modelo guardado en:", out_path)


if __name__ == "__main__":
    main()