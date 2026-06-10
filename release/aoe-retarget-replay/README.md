# AoE-Retarget-Replay

Human-to-Robot 运动重映射工具，将 AoE ego-centric 数据转换为机器人可执行的关节轨迹。

## 架构

```
Layer 1: 数据预处理     — AoE NPZ 解码 → 统一表示
Layer 2: 运动学重映射   — 臂部(现有IK) + 手指(dex-retargeting)
Layer 3: 仿真验证       — MuJoCo 2×2 可视化 (triple_play.py)
Layer 4: 真机 Replay    — Unitree G1 + Inspire Hand SDK
```

## 目标 Joint Space

28D: `[L_ARM(7), R_ARM(7), L_HAND(6), R_HAND(6), PAD(2)]`

## 已验证平台

- **机器人**: Unitree G1 + Inspire 5-fingered hands
- **仿真**: MuJoCo

## 核心依赖

- `dex-retargeting` (手指映射，需为 Inspire Hand 编写配置)
- `mujoco` (仿真验证)
- `numpy`, `scipy` (运动学计算)

## 推荐升级路径

- Phase 1 (当前): 现有 IK + MuJoCo 可视化
- Phase 2: 集成 GMR (2,306⭐) 做全身 retarget
- Phase 3: 集成 SPIDER (464⭐) 做物理可行性验证

## 当前状态

核心脚本完成 80%，详见 [PROJECT.md](../../PROJECT.md) WP3。
