# ============================================================
# FILE: scripts/train_sb3_assignment_ppo.py
# ============================================================
"""Train SB3 PPO on GraphAssignmentEnv (explicit AGV assignment)."""

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
    p.add_argument("--obs-backend", choices=["assignment", "graph", "graph_dict"], default="assignment")
    p.add_argument("--graph-encoder-mode", choices=["manual", "gnn"], default="manual")
    p.add_argument("--graph-gnn-arch", choices=["sage", "gcn", "gat"], default="sage")
    p.add_argument("--gnn-emb-dim", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--gnn-dropout", type=float, default=0.0)
    p.add_argument("--max-request-slots", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None, help="Deprecated alias of --max-request-slots")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--out", type=str, default="data/sb3_assignment_ppo_model")
    args = p.parse_args()

    max_request_slots = args.max_request_slots if args.max_request_slots is not None else args.top_k

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise RuntimeError("Instala SB3 con: pip install '.[sb3]' o pip install stable-baselines3") from exc

    env = GraphAssignmentEnv(
        GraphAssignmentConfig(
            env_id=args.env_id,
            obs_backend=args.obs_backend,
            graph_encoder_mode=args.graph_encoder_mode,
            graph_gnn_arch=args.graph_gnn_arch,
            max_request_slots=max_request_slots,
            max_steps=args.steps,
            seed=args.seed,
            verbose=True,
        )
    )

    if args.obs_backend == "graph_dict":
        from tarware_ext.sb3.gnn_feature_extractor import GnnFeatureExtractor

        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": GnnFeatureExtractor,
                "features_extractor_kwargs": {
                    "emb_dim": args.gnn_emb_dim,
                    "gnn_layers": args.gnn_layers,
                    "dropout": args.gnn_dropout,
                    "architecture": args.graph_gnn_arch,
                },
            },
            verbose=1,
            seed=args.seed,
        )
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)

    model.learn(total_timesteps=args.timesteps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    env.close()
    print("Modelo guardado en:", out_path)


if __name__ == "__main__":
    main()
