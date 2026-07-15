# AoE-Reconstruct-Retarget

An interaction-reconstruction and human-to-robot motion-retargeting toolkit. It transforms Open-AoE egocentric data—smartphone video with reconstructed MANO hands—into reusable reconstruction assets and robot-executable joint trajectories for simulation validation and real-robot replay.

## Integrated Reconstruction and Retargeting Methods

| Method | Path | Supported robots | Capabilities |
|---|---|---|---|
| **Phantom** | [`phantom/`](phantom/) | Unitree G1 + Dex3 / Inspire | Kinematic retargeting with arm IK and dex-retargeting, plus MuJoCo visualization |
| **Retarget Galbot** | [`retarget_galbot/`](retarget_galbot/) | Galbot real hardware / Galaxea bimanual platform | Integrated Galbot real-robot support, palm-to-TCP IK, gripper mapping, egoview synthesis, and LeRobot/Rerun export |
| **Retarget Lab** | [`retarget-lab/`](retarget-lab/) | EgoInfinity/G1, Do-as-I-Do/Sharpa, SPIDER/XHand | External-method integration, 6-DoF reconstruction adapters, and a 12-cell comparison matrix |

## Retarget Lab Coverage

`retarget-lab/` organizes the comparison space as follows:

```text
trajectory_6dof = egoinfinity | do_as_i_do
hand_source     = aoe | estimated
retargeting     = egoinfinity | do_as_i_do | spider
```

The full matrix contains `2 trajectory_6dof × 2 hand_source × 3 retargeting = 12 cells`. This directory does not vendor EgoInfinity, Do-as-I-Do, SPIDER, robot assets, model weights, or generated videos. Follow its README to obtain the required third-party dependencies and run them locally.

## Contributing a Method

Read [CONTRIBUTING.md](../CONTRIBUTING.md), then add a self-contained subdirectory under `aoe-reconstruct-retarget/` with its own README, dependency declaration, and CLI entry point.
