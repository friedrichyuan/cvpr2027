# Open-AoE-2000H

2000 小时 ego-centric 操作数据集，基于消费级智能手机采集。

## 数据格式

每个 segment 目录结构：

```
raw_{collector_id}_seg_{segment_id}/
├── raw_video.mp4                   # 去畸变后的 ego-centric 视频
├── video_info.json                 # 设备信息、相机内参、分辨率、帧率
├── ego_annotation/
│   └── ego_action_annotation.json  # 原子动作切分 + 中英双语 caption
└── ego_process/
    ├── hands.npz                   # MANO 手部重建 (HaWoR)
    └── camera_traj.npz             # 相机轨迹 (MegaSAM)
```

## 数据筛选门限

- **pred_valid 覆盖率** > 80%（MANO 重建有效帧占比）
- **IK failure rate** < 5%（retarget 到 28D joint space 的失败率）
- **相机轨迹连续性**：相邻帧位移 < 10cm

## 当前状态

数据筛选中，详见 [PROJECT.md](../../PROJECT.md) WP1。

## 数据格式详细说明

参见 `../../inner/sample_data/AoE_dataset_README.md`
