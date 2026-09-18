# ARX AC one MuJoCo Scene

Generated from `/home/ymq/code/LifEgo/thirdparty/ARX_Model/AC one/URDF/ACone`.

Files:
- `scene.xml`: MuJoCo MJCF scene.
- `meshes/*.STL`: local ARX mesh assets copied from the URDF package.

Scene convention:
- Robot base frame remains at `z=0`. The rectangular base bottom sits on the tabletop at `z=-0.085`.
- The full AC one URDF is loaded: left arm, right arm, and both two-finger prismatic grippers.
- ARX working direction is base `+X`; `+Z` is up.
- The normal workspace marker is centered in front of the robot at `[0.4, 0.0, -0.085]` m.
- The tabletop is a box centered at `[0.3, 0.0, -0.11000000000000001]` m with half-size `[0.9, 0.55, 0.025]` m.
- The scene uses a MuJoCo-style blue checker floor plane at `z=-0.7849999999999999`, 0.70 m below the tabletop.
- The HEAD camera pose is estimated from the middle head/camera component on `base_link.STL`.
- HEAD camera source mesh bounds are min `[0.011579534038901329, -0.02500000037252903, 0.23600000143051147]` m, max `[0.09618636965751648, 0.02500000037252903, 0.2946948707103729]` m.

HEAD camera:
- Device: `RealSense D405`.
- Resolution: `640x480`.
- Distortion model: `inverse_brown_conrady`.
- K:
  - `[393.030548, 0, 312.918335]`
  - `[0, 392.679291, 240.937149]`
  - `[0, 0, 1]`
- Distortion coeffs `[k1, k2, p1, p2, k3]`: `[-0.050604139, 0.056275558, 0.000794928, 0.000940709, -0.018167889]`.
- `T_cam_in_base` (OpenCV camera axes as columns): `[[0.0, 0.7071067811865475, 0.7071067811865476, 0.09618636965751648], [1.0, -0.0, 0.0, 0.0], [0.0, 0.7071067811865476, -0.7071067811865475, 0.2653474360704422], [0.0, 0.0, 0.0, 1.0]]`.
- Optical axis points toward base `+X` and downward by `45.0` deg.

Coordinate note:
- MuJoCo is right-handed. With `+X` forward and `+Z` up, the URDF's `left_*` branch is located at
  positive `Y` and the `right_*` branch is located at negative `Y`. The names come from the vendor
  URDF; downstream code should rely on frame names and measured positions rather than assuming
  the semantic side from the sign of `Y`.

Important frames:
- `left_flange`: left wrist body `left_link6` origin.
- `left_tcp`: approximate midpoint between the left gripper fingers; zero pose points along base `+X`.
- `tcp`: alias of `left_tcp` for single-arm pipeline compatibility.
- `right_flange`: right wrist body `right_link16` origin.
- `right_tcp`: approximate midpoint between the right gripper fingers; zero pose points along base `+X`.
- `humanego_eef_marker`: mocap target marker used by replay/retargeting scripts.

Open gripper keyframe:
- `open`: all arm revolute joints are clamped around zero, and all four prismatic finger joints are at their upper limits.

Load with MuJoCo:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("/home/ymq/code/LifEgo/ego2exe/assets/mujoco_arx_scene/scene.xml")
data = mujoco.MjData(model)
```
