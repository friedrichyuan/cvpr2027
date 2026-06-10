# Open-AoE 内部工具集

本目录包含 Open-AoE 项目内部使用的数据分析与可视化脚本。

---

## visualize_data_distribution.py

### 功能

对 AoE 采集数据进行全方位分布可视化，生成技术报告所需的 publication-ready 图表。

### 生成图表列表

| 序号 | 文件名 | 内容 |
|------|--------|------|
| 1 | `openaoe_verb_wordcloud.png` | 原子动作动词词云 |
| 2 | `openaoe_object_wordcloud.png` | 被操作物体名词词云 |
| 3 | `openaoe_device_wordcloud.png` | 手机品牌型号词云 |
| 4 | `openaoe_scene_pie.png` | 采集场景饼图 |
| 5 | `openaoe_fov_horizontal.png` | 水平 FOV 分布直方图 |
| 6 | `openaoe_fov_2d.png` | FOV 二维分布热力图 (H vs V) |
| 7 | `openaoe_duration_dist.png` | 片段时长分布 |
| 8 | `openaoe_fps_dist.png` | 帧率分布 |
| 9 | `openaoe_resolution_dist.png` | 视频分辨率分布 |
| 10 | `openaoe_hand_usage.png` | 手部使用分布（左/右/双手）|
| 11 | `openaoe_verb_top20.png` | Top 20 动词柱状图 |
| 12 | `openaoe_object_top20.png` | Top 20 物体柱状图 |
| 13 | `openaoe_action_density.png` | 动作密度分布（actions/sec）|
| 14 | `openaoe_collector_dist.png` | 采集者贡献分布 |
| 15 | `openaoe_actions_per_segment.png` | 每段动作数量分布 |
| 16 | `openaoe_summary_dashboard.png` | 单页总览仪表盘 |
| - | `openaoe_statistics.json` | 原始统计数据（可供 LaTeX 表格引用）|

### 环境依赖

```bash
pip install matplotlib numpy wordcloud
```

如需支持中文词云，确保系统安装了 Noto Sans CJK 或文泉驿字体：
```bash
# Ubuntu/Debian
sudo apt install fonts-noto-cjk
```

### 使用方法

```bash
# 基本用法
python visualize_data_distribution.py \
    --data_dir /path/to/delivery/zhiyuan/20260525 \
    --output_dir /path/to/Open-AoE/illustrations

# 自定义输出前缀（适用于多版本数据对比）
python visualize_data_distribution.py \
    --data_dir /path/to/20260610 \
    --output_dir /path/to/illustrations \
    --prefix openaoe_v2
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--data_dir` | 是 | 数据段目录路径（包含 `raw_*_seg_*` 子目录）|
| `--output_dir` | 是 | 图表输出目录 |
| `--prefix` | 否 | 输出文件名前缀，默认 `openaoe`。不同数据版本可用不同前缀 |

### 数据目录结构要求

脚本期望的数据目录结构：

```
data_dir/
├── raw_{collector_id}_seg_{segment_id}/
│   ├── video_info.json              # 必需：设备信息、相机参数
│   └── ego_annotation/
│       └── ego_action_annotation.json  # 必需：动作标注
├── raw_{collector_id}_seg_{segment_id}/
│   └── ...
└── ...
```

### 复用指南

当新版本数据交付时：

1. 直接修改 `--data_dir` 指向新数据路径
2. 使用 `--prefix` 区分版本（如 `openaoe_20260610`）
3. 输出的 `*_statistics.json` 可用于自动填充 LaTeX 表格中的数据统计

### 输出样式说明

- 所有图表 300 DPI，适合直接嵌入技术报告
- 词云图尺寸 1600x800，白底
- 统计图含均值/中位数标注线及基本统计量文本框
- 饼图自动将小类合并为 "Other"
