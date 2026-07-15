#!/usr/bin/env python3
"""Continue Do-as-I-Do processed AoE data into a Sharpa robot render.

This runner starts from the stage-1 processed layout:

  <output-root>/mano/<embodiment>/<task>/<data-id>/trajectory_keypoints.npz
  <output-root>/mano/<embodiment>/<task>/task_info.json
  <output-root>/assets/objects/<task>/visual.obj

It then runs convex decomposition, scene generation, IK, pedestal resolution,
and a short/full MuJoCo-Warp optimization. It must be executed on the remote
machine with the Do-as-I-Do retargeting environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "experiments" / "foundation_jar_visual_demo_v1"
DEFAULT_RETARGET_ROOT = DEFAULT_EXPERIMENT / "intermediates" / "do_as_i_do" / "retargeting_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--do-as-i-do-root",
        type=Path,
        default=REPO_ROOT / "third_party" / "do-as-i-do" / "retargeting",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RETARGET_ROOT,
    )
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--embodiment", default="bimanual", choices=["right", "left", "bimanual", "auto"])
    parser.add_argument("--dataset", default="do_as_i_do")
    parser.add_argument("--robot", default="sharpa")
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--max-sim-steps", type=int, default=20)
    parser.add_argument("--ik-end-idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.do_as_i_do_root = args.do_as_i_do_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    sys.path.insert(0, str(args.do_as_i_do_root))

    from launch import HAND_FLOOR_COLLISION, OBJECT_FLOOR_COLLISION, load_mjwp_config
    from retargeting.pipeline.decompose_mesh import main as decompose_mesh
    from retargeting.pipeline.generate_scene import main as generate_scene
    from retargeting.pipeline.optimize_physics import main as optimize_physics
    from retargeting.pipeline.resolve_pedestal import resolve_scene_pedestal
    from retargeting.pipeline.solve_ik import main as solve_ik

    output_root = str(args.output_root)
    print(f"[run_do_as_i_do_sharpa] do_as_i_do_root={args.do_as_i_do_root}", flush=True)
    print(f"[run_do_as_i_do_sharpa] output_root={args.output_root}", flush=True)

    decompose_mesh(
        output_root_dir=output_root,
        task=args.task,
        dataset_name=args.dataset,
        data_id=args.data_id,
        embodiment_type=args.embodiment,
        thicken=0.002,
        dilate=0.002,
        force=args.force,
    )
    generate_scene(
        output_root_dir=output_root,
        task=args.task,
        dataset_name=args.dataset,
        data_id=args.data_id,
        embodiment_type=args.embodiment,
        robot_type=args.robot,
        show_viewer=False,
        friction_scale=1.5,
        object_floor_collision=OBJECT_FLOOR_COLLISION,
        hand_floor_collision=HAND_FLOOR_COLLISION,
        use_pedestal=True,
        use_support=True,
        force=args.force,
        add_ur3_arm=True,
    )
    solve_ik(
        output_root_dir=output_root,
        task=args.task,
        dataset_name=args.dataset,
        data_id=args.data_id,
        embodiment_type=args.embodiment,
        robot_type=args.robot,
        show_viewer=False,
        save_video=False,
        force=args.force,
        smoothing=True,
        end_idx=args.ik_end_idx,
    )

    cfg = load_mjwp_config(
        output_root_dir=output_root,
        dataset_name=args.dataset,
        task=args.task,
        data_id=args.data_id,
        robot_type=args.robot,
        embodiment_type=args.embodiment,
        seed=args.seed,
        wait_on_finish=False,
        max_sim_steps=args.max_sim_steps,
        force=args.force,
        show_viewer=False,
        save_video=not args.no_video,
    )
    resolve_scene_pedestal(
        output_root_dir=output_root,
        dataset_name=args.dataset,
        robot_type=args.robot,
        embodiment_type=args.embodiment,
        task=args.task,
        data_id=args.data_id,
        use_pedestal=True,
        use_support=True,
        hand_object_distance_thresh=cfg.hand_object_distance_thresh,
        force=args.force,
    )
    optimize_physics(cfg)
    print(cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
