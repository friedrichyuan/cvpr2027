"""Small MjLab PPO entry point for ARX reference-repair smoke training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.config import RslRlModelCfg, RslRlPpoAlgorithmCfg

from .env_cfg import make_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard/checkpoint directory (default: timestamped logs/arx_reference)",
    )
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--trajectory",
        default=None,
        help="Optional single reference stem, e.g. 51, for feasibility validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or (
        Path("logs/arx_reference") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = make_env_cfg(args.reference_dir, args.num_envs)
    if args.trajectory is not None:
        files = sorted(Path(args.reference_dir).expanduser().glob("*.npz"))
        matches = [i for i, path in enumerate(files) if path.stem == Path(args.trajectory).stem]
        if len(matches) != 1:
            raise FileNotFoundError(f"Reference '{args.trajectory}' was not found")
        env_cfg.commands["reference"].fixed_trajectory_id = matches[0]
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    cfg = RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128),
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(num_learning_epochs=2, num_mini_batches=2),
        num_steps_per_env=24,
        max_iterations=args.iterations,
        save_interval=args.save_interval,
        logger="tensorboard",
        experiment_name="arx_reference_repair",
    )
    print(f"TensorBoard log directory: {log_dir}")
    runner = MjlabOnPolicyRunner(wrapped, asdict(cfg), log_dir=str(log_dir), device=args.device)
    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
