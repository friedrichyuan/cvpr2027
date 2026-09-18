"""Load an ARX PPO checkpoint and replay one exported trajectory in MuJoCo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.config import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.viewer.native import NativeMujocoViewer

from .env_cfg import make_env_cfg


def runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return the architecture used by :mod:`arx_mjlab.train`."""
    return RslRlOnPolicyRunnerCfg(
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
        logger="tensorboard",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        required=True,
        help="Reference filename or stem under --reference-dir, e.g. 17 or 17.npz",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    reference_dir = args.reference_dir.expanduser().resolve()
    reference_files = sorted(reference_dir.glob("*.npz"))
    if not reference_files:
        raise FileNotFoundError(f"No references under {reference_dir}")
    requested_name = Path(args.trajectory).name
    requested_stem = Path(requested_name).stem
    matches = [index for index, path in enumerate(reference_files) if path.stem == requested_stem]
    if len(matches) != 1:
        raise FileNotFoundError(f"Reference '{args.trajectory}' was not found under {reference_dir}")
    trajectory_index = matches[0]
    print(f"Replaying reference [{trajectory_index}]: {reference_files[trajectory_index].name}")

    cfg = make_env_cfg(str(reference_dir), num_envs=1)
    cfg.commands["reference"].fixed_trajectory_id = trajectory_index
    cfg.episode_length_s = 1.0e6
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg()), device=args.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    try:
        NativeMujocoViewer(env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
