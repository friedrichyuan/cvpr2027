# AoE-Retarget-Replay

Human-to-Robot 运动重映射工具集，将 AoE ego-centric 数据（手机录制 + MANO 双手重建）转换为机器人可执行的关节轨迹，支持仿真验证与真机回放。

## 已集成的重映射方法

| 方法 | 路径 | 支持的机器人 | 说明 |
|------|------|-------------|------|
| **Phantom** | [`phantom/`](phantom/) | Unitree G1 + Dex3 / Inspire | 运动学重映射 (IK + dex-retargeting) + MuJoCo 可视化 |
| **Retarget Lab** | [`retarget-lab/`](retarget-lab/) | EgoInfinity/G1、Do-as-I-Do/Sharpa、SPIDER/XHand | 面向 EgoInfinity、Do-as-I-Do、SPIDER 的实验集成与 12-cell 对比矩阵 |

## Retarget Lab 覆盖范围

`retarget-lab/` 当前组织为：

```text
trajectory_6dof = egoinfinity | do_as_i_do
hand_source     = aoe | estimated
retargeting     = egoinfinity | do_as_i_do | spider
```

完整矩阵为 `2 trajectory_6dof x 2 hand_source x 3 retargeting = 12 cells`。该目录不 vendor EgoInfinity、Do-as-I-Do、SPIDER、机器人资产、模型权重或生成视频；需要按其 README 拉取第三方依赖并在本地运行。

## 规划中 / 待完善

| 方法 | 说明 |
|------|------|
| **AGILE** | 物理可行性验证 |
| **银河通用真机** | Retarget 到银河通用真机 |



## 贡献新方法

参见 [CONTRIBUTING.md](../../CONTRIBUTING.md)，在 `aoe-retarget-replay/` 下新建子目录，包含独立的 README、依赖声明和 CLI 入口。
