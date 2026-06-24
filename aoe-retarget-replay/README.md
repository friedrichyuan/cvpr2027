# AoE-Retarget-Replay

Human-to-Robot 运动重映射工具集，将 AoE ego-centric 数据（手机录制 + MANO 双手重建）转换为机器人可执行的关节轨迹，支持仿真验证与真机回放。

## 已集成的重映射方法

| 方法 | 路径 | 支持的机器人 | 说明 |
|------|------|-------------|------|
| **Phantom** | [`phantom/`](phantom/) | Unitree G1 + Dex3 / Inspire | 运动学重映射 (IK + dex-retargeting) + MuJoCo 可视化 |

## 规划中

| 方法 | 说明 |
|------|------|
| **AGILE** | 物理可行性验证 |
| **SPIDER** | 物理可行性验证 |



## 贡献新方法

参见 [CONTRIBUTING.md](../../CONTRIBUTING.md)，在 `aoe-retarget-replay/` 下新建子目录，包含独立的 README、依赖声明和 CLI 入口。