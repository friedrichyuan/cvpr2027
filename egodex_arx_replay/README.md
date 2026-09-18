# EgoDex → ARX5 MuJoCo replay

This is the first, deliberately non-retargeted stage of the project.  It loads
one calibrated EgoDex episode and replays its tracked body, hands, and camera
as a debug overlay in the ARX5 table scene.  It can also convert each hand to
an Ego2Robot-style parallel-gripper reference and draw its TCP frame and jaws.
The ARX5 remains in its open keyframe; no IK or control policy is involved yet.

The replay uses one fixed coordinate transform for the whole episode:

- EgoDex axes: right / up / backwards;
- ARX scene axes: forward / left / up;
- the first EgoDex `hip` is anchored at `[0, 0, -0.30]` m in the ARX scene.

The lowered default accounts for the height difference between the human body
and ARX arm roots.  In `stack/0`, it places the converted gripper centres in
the ARX's central vertical workspace.  It is a fixed episode transform, not a
moving-base optimisation.

This makes the data's 3D geometry visible in the robot workspace without
introducing an invalid, frame-varying robot base.  Later IK and RL stages will
consume the same fixed-frame reference trajectory.

## Run

Install the three Python dependencies in the environment used for MuJoCo, then
run this command from the repository root:

```bash
MUJOCO_GL=glfw python -m egodex_arx_replay.replay --mode both
```

The default episode is `/home/ymq/code/EGODEX_DATASET/test/stack/0.hdf5` and
the default scene is `assets/mujoco_arx_scene/scene.xml`.  Both are overridable:

```bash
MUJOCO_GL=glfw python -m egodex_arx_replay.replay \
  --episode /home/ymq/code/EGODEX_DATASET/test/stack/21.hdf5 \
  --scene assets/mujoco_arx_scene/scene.xml \
  --mode gripper
```

Controls: `Space` play/pause, `J`/`L` or arrow keys step, `R` restart, and
`-`/`+` change replay speed.  Standard MuJoCo mouse controls orbit the scene.

`--mode skeleton` is the original raw-motion replay.  `--mode gripper` shows
only converted left/right parallel grippers; `--mode both` overlays them on the
human skeleton.  The gripper construction follows Ego2Robot exactly: virtual
fingertip `0.7 * index_tip + 0.3 * middle_tip`, TCP midpoint with thumb,
opening width equal to their separation, and the wrist-to-tip vector resolves
the grasp-frame roll.  The two rendered plates are separated by that width.

All overlays are viewer-only debug geometry, so they cannot collide with the
robot.  The ARX5, table, physical grippers, actuators, calibrated robot-head
camera and workspace are all loaded from the supplied MJCF scene.
