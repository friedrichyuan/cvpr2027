# 蚂蚁 AoE 第一人称手部操作数据集

## 📋 概述

本数据集包含第一人称（Egocentric）视角下的**手部三维重建数据**、**相机轨迹**及**细粒度原子动作标注**。数据来源于真实场景的手机采集视频，经自研处理流水线（视觉 SLAM 相机轨迹估计 + 参数化手部重建）处理后，提供了完整的 MANO 手部 3D 姿态参数、metric-scale 相机轨迹，以及细粒度原子动作标注。

### 数据规格


| 项目        | 数值                                        |
| --------- | ----------------------------------------- |
| **帧率**    | 30 FPS                                    |
| **视频分辨率** | 1920×1080 / 1080×720                      |
| **手部模型**  | MANO（15 个手指关节姿态 + 10 维形状参数）               |
| **相机轨迹**  | 视觉 SLAM 相机轨迹估计（metric-scale，相机→世界齐次变换输出） |
| **手部重建**  | MANO 参数化手部重建（世界坐标系 + 相机坐标系双输出）      |
| **标注方式**  | 自动标注（时间段级原子动作，含场景与左右手归属）                  |


---

## 📁 目录结构

每条数据（一个视频片段）的标准输出目录结构如下：

```
<样本目录>/                                          # 视频片段根目录
├── raw_video.mp4                                  # 原始采集视频（1920×1080, 30 FPS）
├── video_info.json                                # 原始相机参数与设备信息
├── ego_annotation/                                # 标注产物
│   └── ego_action_annotation.json                 # 原子动作标注（时间段级）
└── ego_process/                                    # 处理产物
    ├── ego_hands_reconstruction/                   # 手部三维重建结果
    │   ├── hands.npz                               # 手部 MANO 参数 + 相机位姿（核心数据）
    │   ├── camera_traj.npz                         # 相机轨迹（视觉 SLAM）
    │   └── visualization/                          # 可视化结果
    │       ├── hands_combined.mp4                  # 手部 mesh 叠加视频
    │       └── overview.png                        # 轨迹总览图
    └── ego_undistorted_video/                      # 去畸变视频
        ├── raw_video_undistorted.mp4               # 去畸变视频文件
        └── undistorted_video_info.json            # 去畸变后相机参数
```

---

## 📄 核心数据文件详解

### 1. `video_info.json` — 原始视频相机参数

位于每个样本的根目录下，记录原始采集设备信息和相机内参（标定值，含畸变）。

**示例**:

```json
{
  "deviceInfo": {
    "brand": "samsung",
    "model": "SM-S9080",
    "androidVersion": "13"
  },
  "cameraParams": {
    "usedCameraId": "2",
    "resolution": "1920x1080",
    "fx_pixels": 787.912353515625,
    "fy_pixels": 591.310546875,
    "cx_pixels": 961.2589721679688,
    "cy_pixels": 539.4728393554688,
    "max_fx_pixels": 787.912353515625,
    "max_fy_pixels": 787.912353515625,
    "minFocusDistance_meters": 10,
    "lensDistortion": "[-0.0019264653, 0.014610575, -0.009157502, 0.0, 0.0]"
  }
}
```


| 字段                                     | 类型     | 说明                                             |
| -------------------------------------- | ------ | ---------------------------------------------- |
| `deviceInfo.brand`                     | string | 手机品牌                                           |
| `deviceInfo.model`                     | string | 手机型号                                           |
| `deviceInfo.androidVersion`            | string | Android 版本                                     |
| `cameraParams.usedCameraId`            | string | 采集所用摄像头 ID                                     |
| `cameraParams.resolution`              | string | 视频分辨率 `"宽x高"`                                  |
| `cameraParams.fx_pixels`               | float  | 焦距 fx（像素单位，标定值）                                |
| `cameraParams.fy_pixels`               | float  | 焦距 fy（像素单位，标定值）                                |
| `cameraParams.cx_pixels`               | float  | 主点 cx（像素单位）                                    |
| `cameraParams.cy_pixels`               | float  | 主点 cy（像素单位）                                    |
| `cameraParams.max_fx_pixels`           | float  | 最大焦距 fx（像素单位）                                  |
| `cameraParams.max_fy_pixels`           | float  | 最大焦距 fy（像素单位）                                  |
| `cameraParams.minFocusDistance_meters` | float  | 最小对焦距离（米）                                      |
| `cameraParams.lensDistortion`          | string | 镜头畸变系数（JSON 格式数组，5 个系数 `[k1, k2, p1, p2, k3]`） |


> **注意**：原始内参中 `fx_pixels` 与 `fy_pixels` 不相等（横纵采样比例不同）。使用手部数据和相机轨迹时，应使用去畸变后的内参（见下文 `undistorted_video_info.json`）。

---

### 2. `undistorted_video_info.json` — 去畸变后相机参数

位于 `ego_process/ego_undistorted_video/` 目录下。**使用手部数据和相机轨迹时应使用此文件中的参数**。

在 `video_info.json` 的基础上，畸变系数被清零，内参重新计算，并新增了视场角（FOV）、视频文件名和帧率等字段。

**示例**（取自一个示例样本）:

```json
{
  "deviceInfo": {
    "brand": "samsung",
    "model": "SM-S9080",
    "androidVersion": "13"
  },
  "cameraParams": {
    "usedCameraId": "2",
    "resolution": "1920x1080",
    "fx_pixels": 787.8279111480762,
    "fy_pixels": 791.2080732719322,
    "cx_pixels": 961.8658327877191,
    "cy_pixels": 539.9728967260908,
    "max_fx_pixels": 787.912353515625,
    "max_fy_pixels": 787.912353515625,
    "minFocusDistance_meters": 10,
    "lensDistortion": "[0.0, 0.0, 0.0, 0.0, 0.0]",
    "fov_x_degrees": 101.25169246382505,
    "fov_y_degrees": 68.62712143077444,
    "fov_x_radians": 1.7671754067104768,
    "fov_y_radians": 1.1977692251329757
  },
  "video_filename": "raw_video_undistorted.mp4",
  "fps": 30
}
```

与 `video_info.json` 的关键区别：


| 区别项                                            | 原始内参文件        | 去畸变内参文件                       | 说明                                    |
| ---------------------------------------------- | ------------- | ----------------------------- | ------------------------------------- |
| `cameraParams.fx_pixels` / `fy_pixels`         | 原始标定值（fx≠fy）  | 去畸变后重新计算值                     | 用于手部 / 轨迹数据的投影                        |
| `cameraParams.cx_pixels` / `cy_pixels`         | 原始值           | 去畸变后重新计算值                     | —                                     |
| `cameraParams.lensDistortion`                  | 原始畸变系数        | `"[0.0, 0.0, 0.0, 0.0, 0.0]"` | 去畸变后为零                                |
| `cameraParams.fov_x_degrees` / `fov_y_degrees` | ❌ 无           | ✅ 有                           | 水平 / 垂直视场角（度）                         |
| `cameraParams.fov_x_radians` / `fov_y_radians` | ❌ 无           | ✅ 有                           | 水平 / 垂直视场角（弧度）                        |
| `video_filename`                               | ❌ 无           | ✅ 有                           | 去畸变视频文件名（`raw_video_undistorted.mp4`） |
| `fps`                                          | ❌ 无（在原始文件中亦无） | ✅ 有                           | 视频帧率（30）                              |


---

### 3. `hands.npz` — 手部三维重建数据（核心数据）

位于 `ego_process/ego_hands_reconstruction/` 目录下，由参数化手部重建模块生成。使用 [MANO](https://mano.is.tue.mpg.de/) 手部参数化模型表示。

> **记号约定**：下文形状中，`T` = 视频帧数，`2` = 左右手维度（索引 `0` = 左手，`1` = 右手）。

**加载方式**:

```python
import numpy as np
data = np.load("hands.npz")
print(data.files)
# ['R_w2c', 't_w2c', 'R_c2w', 't_c2w', 'pred_trans', 'pred_rot',
#  'pred_trans_cam', 'pred_rot_cam', 'pred_hand_pose', 'pred_betas',
#  'pred_valid', 'focal']
```

#### 3.1 相机位姿参数


| 字段      | 形状          | 数据类型    | 单位  | 说明                                                                               |
| ------- | ----------- | ------- | --- | -------------------------------------------------------------------------------- |
| `R_w2c` | `(T, 3, 3)` | float32 | —   | 世界→相机旋转矩阵                                                                        |
| `t_w2c` | `(T, 3)`    | float32 | 米   | 世界→相机平移向量                                                                        |
| `R_c2w` | `(T, 3, 3)` | float32 | —   | 相机→世界旋转矩阵                                                                        |
| `t_c2w` | `(T, 3)`    | float32 | 米   | 相机→世界平移向量                                                                        |
| `focal` | `()`        | float64 | 像素  | 焦距标量（与 `undistorted_video_info.json` 中 `fx_pixels`、`intrinsic[0,0]` 一致） |


**坐标系转换关系**:

```python
# 世界坐标系 → 相机坐标系
p_cam = R_w2c[t] @ p_world + t_w2c[t]

# 相机坐标系 → 世界坐标系
p_world = R_c2w[t] @ p_cam + t_c2w[t]
```

#### 3.2 手部 MANO 参数（世界坐标系）

由相机坐标系结果经相机位姿转换得到：`p_world = R_c2w @ p_cam + t_c2w`。


| 字段           | 形状          | 数据类型    | 单位  | 说明                 |
| ------------ | ----------- | ------- | --- | ------------------ |
| `pred_trans` | `(2, T, 3)` | float32 | 米   | 手腕平移（世界坐标系）        |
| `pred_rot`   | `(2, T, 3)` | float32 | 弧度  | 手腕全局旋转（轴角表示，世界坐标系） |


#### 3.3 手部 MANO 参数（相机坐标系）

手部重建在相机坐标系下的原生输出（每帧相对当前帧相机）。与世界坐标系互转：`p_cam = R_w2c @ p_world + t_w2c`。


| 字段               | 形状          | 数据类型    | 单位  | 说明                 |
| ---------------- | ----------- | ------- | --- | ------------------ |
| `pred_trans_cam` | `(2, T, 3)` | float32 | 米   | 手腕平移（相机坐标系）        |
| `pred_rot_cam`   | `(2, T, 3)` | float32 | 弧度  | 手腕全局旋转（轴角表示，相机坐标系） |


#### 3.4 MANO 形状与姿态参数


| 字段               | 形状           | 数据类型    | 单位  | 说明                            |
| ---------------- | ------------ | ------- | --- | ----------------------------- |
| `pred_hand_pose` | `(2, T, 45)` | float32 | 弧度  | 15 个手指关节姿态（每关节 3 维轴角），与坐标系无关  |
| `pred_betas`     | `(2, T, 10)` | float32 | —   | MANO 形状参数（10 维 PCA 系数），与坐标系无关 |


#### 3.5 有效性标记


| 字段           | 形状       | 数据类型    | 单位  | 说明                                              |
| ------------ | -------- | ------- | --- | ----------------------------------------------- |
| `pred_valid` | `(2, T)` | float32 | —   | 逐帧检测有效性，取值为 `1.0`（有效）/ `0.0`（无效）。需自行转换为 bool 使用 |


> **⚠️ 重要**: 所有形状为 `(2, ...)` 的数组，第一维索引 `[0]` = **左手**，`[1]` = **右手**。`pred_valid` 为浮点掩码（`0.0`/`1.0`），使用前需转为布尔；无效帧（`pred_valid == 0`）的其他字段值不可信，必须先过滤。

#### 使用示例

```python
import numpy as np
from scipy.spatial.transform import Rotation

data = np.load("hands.npz")

# 获取右手（索引 1）有效帧的世界坐标系位置
valid_right = data['pred_valid'][1] > 0       # (T,) 有效性标记转 bool，[1]=右手
right_trans = data['pred_trans'][1]           # (T, 3) 手腕平移，世界坐标系，米
right_trans_valid = right_trans[valid_right]  # (N_valid, 3) 过滤后有效帧

# 将世界坐标系手部位置投影到图像
focal = data['focal']                         # () 焦距标量，像素（= fx）
R_w2c = data['R_w2c']                         # (T, 3, 3) 世界→相机旋转矩阵
t_w2c = data['t_w2c']                         # (T, 3) 世界→相机平移向量，米

frame_idx = 100
p_cam = R_w2c[frame_idx] @ right_trans[frame_idx] + t_w2c[frame_idx]
# fx/fy/cx/cy 从 undistorted_video_info.json 或 camera_traj.npz 的 intrinsic 获取
u = fx * p_cam[0] / p_cam[2] + cx
v = fy * p_cam[1] / p_cam[2] + cy

# 使用 MANO 模型重建手部 mesh（右手 = 索引 1）
rot_aa = data['pred_rot'][1, frame_idx]           # (3,) 手腕旋转，轴角，弧度
hand_pose = data['pred_hand_pose'][1, frame_idx]  # (45,) 关节姿态，弧度
betas = data['pred_betas'][1, frame_idx]          # (10,) 形状参数，无量纲
```

---

### 4. `camera_traj.npz` — 相机轨迹

位于 `ego_process/ego_hands_reconstruction/` 目录下。由视觉 SLAM 相机轨迹估计模块生成。

**加载方式**:

```python
import numpy as np
data = np.load("camera_traj.npz")
print(data.files)   # ['cam_c2w', 'intrinsic']
```


| 字段          | 形状          | 数据类型    | 单位      | 说明                                              |
| ----------- | ----------- | ------- | ------- | ----------------------------------------------- |
| `cam_c2w`   | `(T, 4, 4)` | float64 | 米（平移部分） | 逐帧 camera-to-world 4×4 齐次变换矩阵                   |
| `intrinsic` | `(3, 3)`    | float64 | 像素      | 相机内参矩阵（去畸变后，与 `undistorted_video_info.json` 一致） |


> **注意**：本次 POC 交付的 `camera_traj.npz` **仅包含** `cam_c2w` 与 `intrinsic` 两个字段，**不包含**逐帧深度图（`depths`）。

#### 变换矩阵说明

`cam_c2w[t]` 为第 t 帧的 4×4 齐次变换矩阵，将相机坐标系中的点变换到世界坐标系：

```
cam_c2w = [[R(3×3), t(3×1)],
           [0 0 0,    1   ]]

p_world = cam_c2w @ [p_cam; 1]
```

#### 内参矩阵

```
intrinsic = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
```

**实际示例**（取自一个示例样本，1920×1080 分辨率）:

```
[[787.83   0.00  961.87]
 [  0.00 791.21  539.97]
 [  0.00   0.00    1.00]]
```

#### 投影 3D 世界点到图像

```python
import numpy as np

data = np.load("camera_traj.npz")
cam_c2w = data['cam_c2w']    # (T, 4, 4) camera-to-world 齐次变换矩阵
K = data['intrinsic']        # (3, 3) 内参矩阵，像素

# 世界点 → 相机系 → 图像
T_w2c = np.linalg.inv(cam_c2w[frame_idx])   # world-to-camera
p_cam = T_w2c @ np.array([x, y, z, 1.0])    # 世界坐标 → 相机坐标
p_img = K @ p_cam[:3]                        # 相机坐标 → 图像坐标
u, v = p_img[0] / p_img[2], p_img[1] / p_img[2]  # 归一化得到像素坐标
```

---

### 5. `ego_action_annotation.json` — 原子动作标注

位于 `ego_annotation/` 目录下。包含**时间段级别**的细粒度原子操作标注，是一个 JSON 数组，每个元素为一个时间段。各样本的段数不同（如 7 ~ 39 段）。

**标注流程**: 视频 → 原子动作识别 → 时间段分割 → JSON 输出

#### 5.1 标注格式

```json
[
  {
    "id": 1,
    "start_ts": "0.0",
    "end_ts": "3.0",
    "start_frame": "0",
    "end_frame": "90",
    "scene": "bedroom",
    "atomic_action": [
      {
        "verb": "grasp",
        "object": "plastic lid",
        "hand": "right",
        "description": "The person grasps a small clear plastic lid from the table surface.",
        "bbox": [630, 320, 720, 420],
        "confidence": 0.95
      }
    ]
  }
]
```

#### 5.2 字段说明


| 字段                            | 类型     | 说明                                                           |
| ----------------------------- | ------ | ------------------------------------------------------------ |
| `id`                          | int    | 标注段序号（从 1 开始递增）                                              |
| `start_ts`                    | string | 起始时间戳（秒）                                                     |
| `end_ts`                      | string | 结束时间戳（秒）                                                     |
| `start_frame`                 | string | 起始帧编号（≈ round(start_ts × fps)，fps=30）                        |
| `end_frame`                   | string | 结束帧编号（≈ round(end_ts × fps)，fps=30）                          |
| `scene`                       | string | 当前场景描述（英文，如 `"bedroom"`、`"laundry room"`）                    |
| `atomic_action`               | array  | 原子动作列表（每段通常 1 个动作）                                           |
| `atomic_action[].verb`        | string | 动作动词（英文，细粒度操作原语）                                             |
| `atomic_action[].object`      | string | 交互物体（英文）                                                     |
| `atomic_action[].hand`        | string | 执行该动作的手：`"left"` / `"right"` / `"both"` / `"none"`           |
| `atomic_action[].description` | string | 动作详细描述（英文，1 句话）                                              |
| `atomic_action[].bbox`        | array  | 动作区域边界框 `[x1, y1, x2, y2]`，为 `raw_video.mp4` 像素坐标（1920×1080） |
| `atomic_action[].confidence`  | float  | 置信度分数（0.0 ~ 1.0）                                             |


#### 5.3 标注规范

- **完整覆盖**: 标注从 0.0s 到视频最后一帧，无间隙无重叠
- **时间连续**: `clip[i].end_ts == clip[i+1].start_ts`
- **一段一动作**: 每个时间段通常包含一个原子操作原语
- **动词选择**: 使用细粒度操作动词（如 `grasp`、`twist`、`slide`），避免模糊动词（如 `use`、`do`）
- **视觉可见**: 仅描述视觉可观察的动作，不推断隐藏物体或动作

> **注意**:
>
> - `start_ts`、`end_ts`、`start_frame`、`end_frame` 均为**字符串类型**，使用时需转换为数值。
> - 帧号 / bbox 均位于 `raw_video.mp4` 的原始像素 / 帧空间；若需对应去畸变视频或手部 / 轨迹数据，请按帧号索引（帧序一致）。

---

### 6. 视频文件


| 文件                          | 位置                                                    | 说明                                  |
| --------------------------- | ----------------------------------------------------- | ----------------------------------- |
| `raw_video.mp4`             | 样本根目录                                                 | 原始采集视频（1920×1080, 30 FPS）           |
| `raw_video_undistorted.mp4` | `ego_process/ego_undistorted_video/`                  | 去畸变后视频                              |
| `hands_combined.mp4`        | `ego_process/ego_hands_reconstruction/visualization/` | 手部 3D mesh 叠加在视频帧上的可视化视频（左手紫色，右手蓝色） |
| `overview.png`              | `ego_process/ego_hands_reconstruction/visualization/` | 相机轨迹与手部运动轨迹总览图                      |


---

## 🔧 坐标系定义

### 相机坐标系（OpenCV 约定）

```
    Z（前/光轴方向）
    ↑
    |
    |
    ●──────→ X（右）
   /
  /
Y（下）
```

- **X 轴**: 朝右（图像 u 方向）
- **Y 轴**: 朝下（图像 v 方向）
- **Z 轴**: 朝前（光轴方向，深度为正）

### 世界坐标系

由视觉 SLAM 定义，以第一帧相机位姿为参考。所有 3D 坐标以**米**为单位。

### 坐标系转换

```python
# 相机坐标系 → 世界坐标系
p_world = R_c2w @ p_cam + t_c2w          # hands.npz 中的 3×3 旋转 + 3×1 平移
p_world = cam_c2w @ [p_cam; 1]           # camera_traj.npz 中的 4×4 齐次变换

# 世界坐标系 → 相机坐标系
p_cam = R_w2c @ p_world + t_w2c          # hands.npz 中的 3×3 旋转 + 3×1 平移

# 相机坐标系 → 图像坐标系（fx/fy/cx/cy 来自 undistorted_video_info.json 或 intrinsic 矩阵）
# hands.npz 中的 focal 标量 = fx（去畸变后 fx 与 fy 仍略有差异，投影时建议用 intrinsic 的 fx/fy）
u = fx * p_cam[0] / p_cam[2] + cx
v = fy * p_cam[1] / p_cam[2] + cy
```

---

## 📊 数据指标验证

### 相机内参标定精度

内参差异全部 < 1%，平均仅 0.64%，畸变系数数量级在 10⁻³ 以内。

### 手部姿态估计性能

经 Procrustes 对齐后手部姿态所有节点约 1cm 精度。


| 指标           | 数值     | 单位  | 含义                                               |
| ------------ | ------ | --- | ------------------------------------------------ |
| **PA-MPJPE** | 10.5   | mm  | 手部关节经 Procrustes 刚性对齐（旋转+平移+尺度）的平均关节点位置误差        |
| **WA-MPJPE** | 11.9   | mm  | 手部关节在世界坐标系下经 Procrustes 对齐（旋转+平移，不含尺度）的平均关节点位置误差 |
| **AUC**      | 0.9900 | —   | 手部关节 PCK 曲线下面积，越接近 1 越好                          |


### 相机轨迹估计性能


| 指标            | 数值   | 单位  | 含义                               |
| ------------- | ---- | --- | -------------------------------- |
| **ATE**       | 4.4  | mm  | 相机轨迹经 7-DoF 对齐（旋转+平移+尺度）后的平均位置误差 |
| **ATE-S**     | 14.1 | mm  | 相机轨迹经 6-DoF 对齐（不含尺度）后的平均位置误差     |
| **RPE-Trans** | 1.00 | mm  | 相邻帧间位移预测的相对位姿误差（平移）              |
| **RPE-Rot**   | 5.40 | °   | 相邻帧间旋转预测的相对位姿误差（旋转）              |


---

## 关键注意事项

### 有效性检查

- **单位统一**: 所有 3D 坐标以**米**为单位
- **旋转表示**: 手部旋转使用**轴角**（axis-angle）表示，需转换为旋转矩阵使用
- **手部索引**: `hands.npz` 中形状为 `(2, ...)` 的数组，`[0]` = **左手**，`[1]` = **右手**
- **有效性掩码类型**: `pred_valid` 为 **float32**（`0.0`/`1.0`），使用前需转为布尔
- **内参固定**: 相机内参在所有帧中保持不变（去畸变后）
- **有效性过滤**: 必须使用 `pred_valid` 过滤无效帧，无效帧数据不可信
- **时间戳类型**: 标注文件中的时间戳和帧号均为**字符串**，使用时需转换

---

## 🔍 可视化

### 使用脚本重新生成手部可视化

```bash
python visualization/visualize_hands.py /path/to/<样本目录>
```

其中 `<样本目录>` 为视频片段的根目录路径。

### 可视化输出

- `hands_combined.mp4`: 手部 3D mesh 叠加在视频帧上（左手紫色，右手蓝色）
- `overview.png`: 相机轨迹和手部运动轨迹总览图

---

## ✅ 自检清单

使用数据前建议确认：

- [ ] `hands.npz` 和 `camera_traj.npz` 中的帧数 T 一致，且与 `raw_video.mp4` 帧数一致
- [ ] `pred_valid`（float32）已转为布尔并用于过滤无效帧，无效帧数据未被使用
- [ ] 手部索引取用正确：`[0]` = 左手，`[1]` = 右手
- [ ] 3D 坐标单位为米，值在合理范围内（手部位置通常在相机前方 0.2m ~ 1.0m）
- [ ] 旋转矩阵满足正交性（R^T R ≈ I，det(R) ≈ 1）
- [ ] `camera_traj.npz` 中的 `intrinsic` 与 `undistorted_video_info.json` 中的内参一致
- [ ] 使用去畸变后的内参（`undistorted_video_info.json`），而非原始内参
- [ ] 标注文件中的时间戳和帧号已从字符串转换为数值
- [ ] 标注时间段完整覆盖视频全长，无间隙无重叠

---

## 📐 数据一致性说明

### 帧数对齐

所有时序数据的帧数 T 保持严格一致：


| 数据                                            | 帧数维度 | 说明                                                   |
| --------------------------------------------- | ---- | ---------------------------------------------------- |
| `hands.npz` 各数组                               | T    | 第二维（手部参数，形状 `(2, T, ...)` ）或第一维（相机参数，形状 `(T, ...)` ） |
| `camera_traj.npz` 中 `cam_c2w`                 | T    | 第一维                                                  |
| `raw_video.mp4` / `raw_video_undistorted.mp4` | T 帧  | 与上述时序数组逐帧对齐                                          |


### 内参一致性

以下三处的内参值应保持一致（均为去畸变后的值）：

1. `undistorted_video_info.json` 中的 `fx_pixels`, `fy_pixels`, `cx_pixels`, `cy_pixels`
2. `camera_traj.npz` 中的 `intrinsic` 矩阵（`[[fx,0,cx],[0,fy,cy],[0,0,1]]`）
3. `hands.npz` 中的 `focal` 标量（= `fx`）

### 坐标系一致性


| 数据                                | 坐标系   | 说明                     |
| --------------------------------- | ----- | ---------------------- |
| `pred_trans` / `pred_rot`         | 世界坐标系 | 手部在全局空间中的位置和朝向         |
| `pred_trans_cam` / `pred_rot_cam` | 相机坐标系 | 手部相对于当前帧相机的位置和朝向       |
| `cam_c2w`                         | 世界坐标系 | 相机在全局空间中的位姿            |
| `R_c2w` / `t_c2w`                 | 世界坐标系 | 与 `cam_c2w` 的旋转和平移部分一致 |


