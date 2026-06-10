#!/usr/bin/env python3
"""
Open-AoE Data Distribution Visualization Toolkit

Generates publication-ready figures for the Open-AoE technical report:
- Atomic action verb word cloud
- Manipulated object noun word cloud
- Phone brand/model word cloud
- Collection scene pie chart
- Camera FOV distribution histogram
- Video duration distribution
- Video resolution & FPS distribution
- Hand usage distribution (left/right/both)
- Segment count per collector heatmap
- Action density (actions per second) distribution

Usage:
    python visualize_data_distribution.py --data_dir <path_to_segments> --output_dir <path_to_output>

Example:
    python visualize_data_distribution.py \
        --data_dir /path/to/20260525 \
        --output_dir /path/to/Open-AoE/illustrations
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from wordcloud import WordCloud

# ---------- Style Configuration ----------

FONT_PATH = None
for candidate in [
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]:
    if os.path.exists(candidate):
        FONT_PATH = candidate
        break

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

if FONT_PATH and 'CJK' in FONT_PATH:
    prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False

COLORMAP = 'viridis'
PIE_COLORS = plt.cm.Set3.colors


# ---------- Data Loading ----------

def load_all_segments(data_dir: str):
    """Load video_info and annotations from all segments."""
    data_dir = Path(data_dir)
    segments = []

    for seg_dir in sorted(data_dir.iterdir()):
        if not seg_dir.is_dir() or not seg_dir.name.startswith('raw_'):
            continue

        entry = {'dir_name': seg_dir.name}

        # Parse collector_id and seg_id from directory name
        parts = seg_dir.name.split('_')
        if len(parts) >= 4:
            entry['collector_id'] = parts[1]
            entry['seg_id'] = parts[3]

        # Load video_info
        video_info_path = seg_dir / 'video_info.json'
        if video_info_path.exists():
            try:
                with open(video_info_path, 'r') as f:
                    entry['video_info'] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Load annotation
        annotation_path = seg_dir / 'ego_annotation' / 'ego_action_annotation.json'
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    entry['annotations'] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        segments.append(entry)

    return segments


def extract_statistics(segments: list) -> dict:
    """Extract all statistical data from segments."""
    stats = {
        'verbs': Counter(),
        'objects': Counter(),
        'scenes': Counter(),
        'brands': Counter(),
        'models': Counter(),
        'brand_models': Counter(),
        'fov_h': [],
        'fov_v': [],
        'fps_values': [],
        'resolutions': Counter(),
        'hands': Counter(),
        'durations': [],
        'collector_seg_counts': Counter(),
        'action_densities': [],
        'actions_per_segment': [],
    }

    for seg in segments:
        # Device info
        vi = seg.get('video_info', {})
        device = vi.get('deviceInfo', {})
        camera = vi.get('cameraParams', {})

        brand = device.get('brand', 'Unknown')
        model = device.get('model', 'Unknown')
        stats['brands'][brand] += 1
        stats['models'][model] += 1
        stats['brand_models'][f"{brand} {model}"] += 1

        # Camera params
        if 'fovHorizontal_degrees' in camera:
            stats['fov_h'].append(camera['fovHorizontal_degrees'])
        if 'fovVertical_degrees' in camera:
            stats['fov_v'].append(camera['fovVertical_degrees'])
        if 'fps' in camera:
            stats['fps_values'].append(camera['fps'])
        if 'resolution' in camera:
            stats['resolutions'][camera['resolution']] += 1

        # Collector stats
        collector_id = seg.get('collector_id', 'unknown')
        stats['collector_seg_counts'][collector_id] += 1

        # Annotations
        annotations = seg.get('annotations', [])
        stats['actions_per_segment'].append(len(annotations))

        seg_duration = 0
        for item in annotations:
            scene = item.get('scene', 'unknown')
            stats['scenes'][scene] += 1

            end_ts = float(item.get('end_ts', 0))
            if end_ts > seg_duration:
                seg_duration = end_ts

            for action in item.get('atomic_action', []):
                verb = action.get('verb', '').strip().lower()
                obj = action.get('object', '').strip().lower()
                hand = action.get('hand', '').strip().lower()

                if verb:
                    stats['verbs'][verb] += 1
                if obj:
                    stats['objects'][obj] += 1
                if hand:
                    stats['hands'][hand] += 1

        if seg_duration > 0:
            stats['durations'].append(seg_duration)
            n_actions = sum(len(item.get('atomic_action', [])) for item in annotations)
            if n_actions > 0:
                stats['action_densities'].append(n_actions / seg_duration)

    return stats


# ---------- Visualization Functions ----------

def plot_wordcloud(counter: Counter, title: str, output_path: str, colormap: str = 'viridis'):
    """Generate and save a word cloud from a Counter."""
    if not counter:
        print(f"  [SKIP] No data for: {title}")
        return

    wc = WordCloud(
        width=1600, height=800,
        background_color='white',
        colormap=colormap,
        font_path=FONT_PATH,
        max_words=150,
        min_font_size=8,
        relative_scaling=0.5,
        prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(counter)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


def plot_pie_chart(counter: Counter, title: str, output_path: str, top_n: int = 12):
    """Generate a pie chart, grouping small slices into 'Other'."""
    if not counter:
        print(f"  [SKIP] No data for: {title}")
        return

    total = sum(counter.values())
    sorted_items = counter.most_common(top_n)
    top_sum = sum(v for _, v in sorted_items)
    other = total - top_sum

    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    if other > 0:
        labels.append('Other')
        values.append(other)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=PIE_COLORS[:len(values)],
        pctdistance=0.85, startangle=90,
        textprops={'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


def plot_histogram(values: list, title: str, xlabel: str, output_path: str,
                   bins: int = 30, color: str = '#2196F3', add_stats: bool = True):
    """Generate a histogram with optional statistics annotation."""
    if not values:
        print(f"  [SKIP] No data for: {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    arr = np.array(values)

    n, bin_edges, patches = ax.hist(arr, bins=bins, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(arr.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {arr.mean():.1f}')
    ax.axvline(np.median(arr), color='orange', linestyle='-.', linewidth=1.5, label=f'Median: {np.median(arr):.1f}')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=11)

    if add_stats:
        stats_text = (f'N = {len(arr)}\n'
                      f'Mean = {arr.mean():.2f}\n'
                      f'Std = {arr.std():.2f}\n'
                      f'Min = {arr.min():.2f}\n'
                      f'Max = {arr.max():.2f}')
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.grid(axis='y', alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


def plot_bar_chart(counter: Counter, title: str, xlabel: str, output_path: str,
                   top_n: int = 20, horizontal: bool = True, color: str = '#4CAF50'):
    """Generate a bar chart from a Counter."""
    if not counter:
        print(f"  [SKIP] No data for: {title}")
        return

    items = counter.most_common(top_n)
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))

    if horizontal:
        labels = labels[::-1]
        values = values[::-1]
        ax.barh(range(len(labels)), values, color=color, alpha=0.8, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Count', fontsize=12)
    else:
        ax.bar(range(len(labels)), values, color=color, alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
        ax.set_ylabel('Count', fontsize=12)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


def plot_fov_2d(fov_h: list, fov_v: list, title: str, output_path: str):
    """Generate a 2D scatter/hexbin of horizontal vs vertical FOV."""
    if not fov_h or not fov_v:
        print(f"  [SKIP] No data for: {title}")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(fov_h, fov_v, gridsize=20, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('Horizontal FOV (degrees)', fontsize=12)
    ax.set_ylabel('Vertical FOV (degrees)', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count')
    ax.grid(alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


def plot_hand_usage(hand_counter: Counter, output_path: str):
    """Generate a donut chart for hand usage distribution."""
    if not hand_counter:
        print(f"  [SKIP] No data for hand usage")
        return

    canonical = Counter()
    for hand, count in hand_counter.items():
        if 'both' in hand or 'two' in hand:
            canonical['both'] += count
        elif 'left' in hand:
            canonical['left'] += count
        elif 'right' in hand:
            canonical['right'] += count
        else:
            canonical[hand] += count

    labels = list(canonical.keys())
    values = list(canonical.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=colors[:len(values)],
        pctdistance=0.8, startangle=90,
        wedgeprops=dict(width=0.5),
        textprops={'fontsize': 12}
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax.set_title('Hand Usage Distribution', fontsize=16, fontweight='bold', pad=20)
    total = sum(values)
    ax.text(0, 0, f'N={total}', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


def plot_collector_distribution(collector_counts: Counter, output_path: str, top_n: int = 30):
    """Generate a bar chart showing segment counts per collector."""
    if not collector_counts:
        print(f"  [SKIP] No data for collector distribution")
        return

    items = collector_counts.most_common(top_n)
    labels = [f"C{item[0]}" for item in items]
    values = [item[1] for item in items]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), values, color=plt.cm.tab20(np.linspace(0, 1, len(labels))),
                  alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Number of Segments', fontsize=12)
    ax.set_title(f'Segment Count per Collector (Top {top_n})', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)

    stats_text = (f'Total Collectors: {len(collector_counts)}\n'
                  f'Total Segments: {sum(collector_counts.values())}\n'
                  f'Avg Segments/Collector: {sum(collector_counts.values()) / len(collector_counts):.1f}')
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


def plot_summary_dashboard(stats: dict, output_path: str):
    """Generate a single-page summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Open-AoE Dataset Distribution Summary', fontsize=18, fontweight='bold', y=0.98)

    # 1. Top verbs bar
    ax1 = fig.add_subplot(2, 3, 1)
    top_verbs = stats['verbs'].most_common(10)
    if top_verbs:
        ax1.barh([v[0] for v in top_verbs][::-1], [v[1] for v in top_verbs][::-1],
                 color='#2196F3', alpha=0.8)
        ax1.set_title('Top 10 Action Verbs', fontweight='bold')
        ax1.set_xlabel('Count')

    # 2. Top objects bar
    ax2 = fig.add_subplot(2, 3, 2)
    top_objs = stats['objects'].most_common(10)
    if top_objs:
        ax2.barh([v[0] for v in top_objs][::-1], [v[1] for v in top_objs][::-1],
                 color='#4CAF50', alpha=0.8)
        ax2.set_title('Top 10 Objects', fontweight='bold')
        ax2.set_xlabel('Count')

    # 3. Scene pie
    ax3 = fig.add_subplot(2, 3, 3)
    if stats['scenes']:
        top_scenes = stats['scenes'].most_common(6)
        total = sum(stats['scenes'].values())
        scene_labels = [s[0] for s in top_scenes]
        scene_values = [s[1] for s in top_scenes]
        other = total - sum(scene_values)
        if other > 0:
            scene_labels.append('other')
            scene_values.append(other)
        ax3.pie(scene_values, labels=scene_labels, autopct='%1.0f%%',
                colors=PIE_COLORS[:len(scene_values)], textprops={'fontsize': 8})
        ax3.set_title('Scene Distribution', fontweight='bold')

    # 4. FOV histogram
    ax4 = fig.add_subplot(2, 3, 4)
    if stats['fov_h']:
        ax4.hist(stats['fov_h'], bins=20, color='#FF9800', alpha=0.75, edgecolor='white')
        ax4.set_title('Horizontal FOV', fontweight='bold')
        ax4.set_xlabel('Degrees')
        ax4.set_ylabel('Count')

    # 5. Duration histogram
    ax5 = fig.add_subplot(2, 3, 5)
    if stats['durations']:
        ax5.hist(stats['durations'], bins=20, color='#9C27B0', alpha=0.75, edgecolor='white')
        ax5.set_title('Segment Duration', fontweight='bold')
        ax5.set_xlabel('Seconds')
        ax5.set_ylabel('Count')

    # 6. Brand pie
    ax6 = fig.add_subplot(2, 3, 6)
    if stats['brands']:
        top_brands = stats['brands'].most_common(5)
        total = sum(stats['brands'].values())
        brand_labels = [b[0] for b in top_brands]
        brand_values = [b[1] for b in top_brands]
        other = total - sum(brand_values)
        if other > 0:
            brand_labels.append('other')
            brand_values.append(other)
        ax6.pie(brand_values, labels=brand_labels, autopct='%1.0f%%',
                colors=PIE_COLORS[:len(brand_values)], textprops={'fontsize': 9})
        ax6.set_title('Device Brand', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] {output_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description='Open-AoE Data Distribution Visualization Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the segment data directory (e.g., .../20260525)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save generated figures')
    parser.add_argument('--prefix', type=str, default='openaoe',
                        help='Filename prefix for output figures (default: openaoe)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from: {args.data_dir}")
    segments = load_all_segments(args.data_dir)
    print(f"Loaded {len(segments)} segments")

    print("Extracting statistics...")
    stats = extract_statistics(segments)

    print(f"\nDataset Summary:")
    print(f"  Segments: {len(segments)}")
    print(f"  Unique verbs: {len(stats['verbs'])}")
    print(f"  Unique objects: {len(stats['objects'])}")
    print(f"  Unique scenes: {len(stats['scenes'])}")
    print(f"  Unique device models: {len(stats['models'])}")
    print(f"  Unique collectors: {len(stats['collector_seg_counts'])}")
    print(f"  Total actions: {sum(stats['verbs'].values())}")

    prefix = args.prefix
    out = args.output_dir

    print("\nGenerating visualizations...")

    # 1. Verb word cloud
    plot_wordcloud(stats['verbs'],
                   'Atomic Action Verb Distribution',
                   os.path.join(out, f'{prefix}_verb_wordcloud.png'),
                   colormap='plasma')

    # 2. Object noun word cloud
    plot_wordcloud(stats['objects'],
                   'Manipulated Object Distribution',
                   os.path.join(out, f'{prefix}_object_wordcloud.png'),
                   colormap='viridis')

    # 3. Phone brand+model word cloud
    plot_wordcloud(stats['brand_models'],
                   'Device Brand & Model Distribution',
                   os.path.join(out, f'{prefix}_device_wordcloud.png'),
                   colormap='Set2')

    # 4. Scene pie chart
    plot_pie_chart(stats['scenes'],
                   'Collection Scene Distribution',
                   os.path.join(out, f'{prefix}_scene_pie.png'))

    # 5. Camera FOV distribution (horizontal)
    plot_histogram(stats['fov_h'],
                   'Camera Horizontal FOV Distribution',
                   'Horizontal FOV (degrees)',
                   os.path.join(out, f'{prefix}_fov_horizontal.png'),
                   color='#FF9800')

    # 6. FOV 2D (horizontal vs vertical)
    plot_fov_2d(stats['fov_h'], stats['fov_v'],
                'Camera FOV Distribution (H vs V)',
                os.path.join(out, f'{prefix}_fov_2d.png'))

    # 7. Video duration distribution
    plot_histogram(stats['durations'],
                   'Segment Duration Distribution',
                   'Duration (seconds)',
                   os.path.join(out, f'{prefix}_duration_dist.png'),
                   color='#9C27B0')

    # 8. FPS distribution
    plot_bar_chart(Counter(stats['fps_values']),
                   'Frame Rate Distribution',
                   'FPS',
                   os.path.join(out, f'{prefix}_fps_dist.png'),
                   horizontal=False, color='#009688')

    # 9. Resolution distribution
    plot_bar_chart(stats['resolutions'],
                   'Video Resolution Distribution',
                   'Resolution',
                   os.path.join(out, f'{prefix}_resolution_dist.png'),
                   horizontal=False, color='#3F51B5')

    # 10. Hand usage
    plot_hand_usage(stats['hands'],
                    os.path.join(out, f'{prefix}_hand_usage.png'))

    # 11. Top verbs bar chart
    plot_bar_chart(stats['verbs'],
                   'Top 20 Atomic Action Verbs',
                   'Verb',
                   os.path.join(out, f'{prefix}_verb_top20.png'),
                   top_n=20, color='#2196F3')

    # 12. Top objects bar chart
    plot_bar_chart(stats['objects'],
                   'Top 20 Manipulated Objects',
                   'Object',
                   os.path.join(out, f'{prefix}_object_top20.png'),
                   top_n=20, color='#4CAF50')

    # 13. Action density distribution
    plot_histogram(stats['action_densities'],
                   'Action Density Distribution (Actions per Second)',
                   'Actions / Second',
                   os.path.join(out, f'{prefix}_action_density.png'),
                   color='#E91E63')

    # 14. Collector segment count distribution
    plot_collector_distribution(stats['collector_seg_counts'],
                               os.path.join(out, f'{prefix}_collector_dist.png'))

    # 15. Actions per segment
    plot_histogram(stats['actions_per_segment'],
                   'Number of Actions per Segment',
                   'Action Count',
                   os.path.join(out, f'{prefix}_actions_per_segment.png'),
                   color='#795548')

    # 16. Summary dashboard
    plot_summary_dashboard(stats,
                           os.path.join(out, f'{prefix}_summary_dashboard.png'))

    # Save raw statistics as JSON for downstream use
    stats_export = {
        'n_segments': len(segments),
        'n_unique_verbs': len(stats['verbs']),
        'n_unique_objects': len(stats['objects']),
        'n_unique_scenes': len(stats['scenes']),
        'n_unique_devices': len(stats['models']),
        'n_unique_collectors': len(stats['collector_seg_counts']),
        'n_total_actions': sum(stats['verbs'].values()),
        'top_verbs': stats['verbs'].most_common(50),
        'top_objects': stats['objects'].most_common(50),
        'scenes': dict(stats['scenes']),
        'brands': dict(stats['brands']),
        'brand_models': dict(stats['brand_models'].most_common(30)),
        'resolutions': dict(stats['resolutions']),
        'fps_values': dict(Counter(stats['fps_values'])),
        'hands': dict(stats['hands']),
        'fov_h_stats': {
            'mean': float(np.mean(stats['fov_h'])) if stats['fov_h'] else None,
            'std': float(np.std(stats['fov_h'])) if stats['fov_h'] else None,
            'min': float(np.min(stats['fov_h'])) if stats['fov_h'] else None,
            'max': float(np.max(stats['fov_h'])) if stats['fov_h'] else None,
        },
        'duration_stats': {
            'mean': float(np.mean(stats['durations'])) if stats['durations'] else None,
            'std': float(np.std(stats['durations'])) if stats['durations'] else None,
            'min': float(np.min(stats['durations'])) if stats['durations'] else None,
            'max': float(np.max(stats['durations'])) if stats['durations'] else None,
            'total_hours': float(np.sum(stats['durations']) / 3600) if stats['durations'] else None,
        },
    }

    stats_path = os.path.join(out, f'{prefix}_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_export, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {stats_path}")

    print(f"\nDone! Generated {16} figures + 1 statistics JSON in: {out}")


if __name__ == '__main__':
    main()
