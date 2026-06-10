# AoE-Training-Ready

Open-AoE 数据到主流模型训练的格式转换工具和 Training Recipe。

## 架构

```
AoE NPZ → retarget_npz_to_lerobot.py → LeRobot v2.1 格式
                                            │
                        ┌───────────────────┼───────────────────┐
                        ▼                   ▼                   ▼
                   fastWAM Recipe     GR00T Recipe        π₀.₅ Recipe
                   (已验证 ✅)       (已验证基础 🟡)     (待打通 🔴)
```

## 已支持模型

| 模型 | Stars | 状态 | 验证结果 |
|------|-------|------|---------|
| fastWAM | — | ✅ 已验证 | multihead, Step 100, Loss 0.523 |
| GR00T N1.5 | 7,280⭐ | 🟡 有验证基础 | Close Laptop SR 45%→95% (via FLARE) |
| π₀.₅ (OpenPI) | 12,220⭐ | 🔴 待打通 | — |
| RDT-1B | 1,717⭐ | 🔴 待打通 | — |

## 数据格式中枢

以 **LeRobot v2.1** (24,781⭐) 作为数据格式标准层，所有模型适配从 LeRobot 格式出发。

## 核心文件

- `retarget_npz_to_lerobot.py` — AoE NPZ → LeRobot v2.1 (完成度 60%)
- `recipes/fastwam/` — fastWAM 训练配置
- `recipes/groot/` — GR00T N1.7 训练配置 (待完善)
- `recipes/openpi/` — π₀.₅ 训练配置 (待开发)

## 当前状态

fastWAM 已验证，格式转换器 60% 完成，详见 [PROJECT.md](../../PROJECT.md) WP4。
