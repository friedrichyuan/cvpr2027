"""Annotation overlays: info panel and a bottom timeline bar.

The panels follow a light "white / blue" theme: near-white card backgrounds, a
solid blue header band, dark slate body text and blue accents, designed to match
the Open-AoE brand look while keeping strong contrast/legibility.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# White / blue theme palette (all colours are BGR for OpenCV).
# ---------------------------------------------------------------------------
C_PANEL_BG = (252, 247, 240)     # near-white, faint cool tint
C_CARD_BG = (245, 235, 222)      # slightly deeper card / row background
C_HEADER = (245, 133, 38)        # brand blue header band (#2685F5)
C_BLUE = (224, 120, 26)          # accent blue (text / bars)
C_BLUE_DARK = (176, 86, 12)      # deeper blue for emphasis
C_TEXT = (74, 54, 36)            # dark navy-slate body text
C_TEXT_MUTED = (140, 116, 92)    # muted secondary text
C_TRACK = (236, 224, 206)        # timeline empty-track fill
C_WHITE = (255, 255, 255)
C_OUTLINE = (210, 196, 178)      # subtle separators / borders

# Height of the unified top title bar shared by all three sections.
HEADER_H = 52

# Curated blue -> teal -> indigo -> periwinkle timeline palette (BGR). Adjacent
# entries are kept visually distinct while staying within the brand's cool range.
SEGMENT_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (245, 133, 38),    # brand blue
    (196, 158, 41),    # teal
    (210, 110, 28),    # deep azure
    (170, 196, 92),    # aqua-teal
    (224, 120, 26),    # azure
    (150, 96, 36),     # indigo-blue
    (236, 178, 96),    # sky
    (208, 132, 168),   # periwinkle
    (176, 86, 12),     # deep blue
    (190, 200, 120),   # pale teal
    (138, 150, 70),    # slate-teal
    (224, 150, 200),   # lavender-periwinkle
)

# Per-hand accent colours used for the info-panel action markers (BGR).
HAND_COLORS = {
    "right": (245, 133, 38),     # blue
    "left": (0, 170, 255),       # amber
    "both": (180, 130, 40),      # teal-blue
    None: (245, 133, 38),
}


def load_annotations(json_path: str) -> List[dict]:
    """Load the action-annotation list from ``ego_action_annotation.json``."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def annotation_for_frame(annotations: List[dict], frame_idx: int) -> Optional[dict]:
    """Return the annotation segment containing ``frame_idx`` (or None)."""
    for ann in annotations:
        if int(ann["start_frame"]) <= frame_idx <= int(ann["end_frame"]):
            return ann
    return None


def _hand_color(hand: Optional[str]) -> Tuple[int, int, int]:
    if hand is None:
        return HAND_COLORS[None]
    return HAND_COLORS.get(str(hand).lower(), HAND_COLORS[None])


def _action_label(action: dict) -> str:
    """Compact ``"verb object"`` label for one atomic action."""
    verb = str(action.get("verb", "") or "").replace("_", " ").strip()
    obj = str(action.get("object", "") or "").strip()
    return (verb + " " + obj).strip() or "action"


def _wrap_text(text: str, max_w: int, fs: float, th: int) -> List[str]:
    """Greedy word-wrap ``text`` to lines that fit ``max_w`` px at scale ``fs``."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = str(text).split()
    if not words:
        return []
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = cur + " " + w
        (tw, _), _ = cv2.getTextSize(cand, font, fs, th)
        if tw <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def draw_logo(frame: np.ndarray, margin: int = 12) -> np.ndarray:
    """Overlay a tidy blue ``AoE`` wordmark in the top-right of ``frame``.

    Drawn on a subtle translucent white rounded pill so it stays legible over any
    footage. Modifies ``frame`` in place.
    """
    fh, fw = frame.shape[:2]
    fs = 0.85
    th = 2
    (tw, tht), _ = cv2.getTextSize("AoE", cv2.FONT_HERSHEY_DUPLEX, fs, th)
    pad_x, pad_y = 12, 9
    box_w, box_h = tw + 2 * pad_x, tht + 2 * pad_y
    x1 = fw - margin - box_w
    y1 = margin
    x2, y2 = x1 + box_w, y1 + box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_HEADER, 2, cv2.LINE_AA)
    baseline = y1 + pad_y + tht
    cv2.putText(frame, "AoE", (x1 + pad_x, baseline),
                cv2.FONT_HERSHEY_DUPLEX, fs, C_HEADER, th, cv2.LINE_AA)
    return frame


def draw_top_header(width: int, sections: List[Tuple[int, str]]) -> np.ndarray:
    """Render the unified top title bar shared by all three columns.

    Args:
        width: full composite width.
        sections: ``[(x_start, title), ...]`` one per column; titles are drawn at
            a common baseline / font / padding so the section tops line up.

    Returns:
        a ``(HEADER_H, width, 3)`` BGR strip (white with blue titles).
    """
    bar = np.full((HEADER_H, width, 3), C_PANEL_BG, dtype=np.uint8)
    fs, th = 0.82, 2
    baseline = int(HEADER_H * 0.66)
    for x_start, title in sections:
        cv2.putText(bar, title, (x_start + 16, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, C_HEADER, th, cv2.LINE_AA)
    cv2.line(bar, (0, HEADER_H - 2), (width, HEADER_H - 2), C_HEADER, 2, cv2.LINE_AA)
    return bar


def draw_info_panel(width: int, height: int, annotation: Optional[dict],
                    frame_idx: int, total_frames: int, fps: float,
                    sample_name: str) -> np.ndarray:
    """Render the annotation info panel (white / blue theme).

    The section title ("Action Annotation") lives in the shared top header bar,
    so this panel starts with the sample name. All body text is wrapped to the
    panel width (in pixels) so nothing clips at the right edge.
    """
    panel = np.full((height, width, 3), C_PANEL_BG, dtype=np.uint8)
    left = 20
    text_w = width - 2 * left
    name = sample_name
    while name and cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] > text_w:
        name = name[:-1]
    if name != sample_name:
        name = name[:-3] + "..."
    cv2.putText(panel, name, (left, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT_MUTED, 1, cv2.LINE_AA)

    if annotation is None:
        cv2.putText(panel, "No annotation", (left, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEXT_MUTED, 2, cv2.LINE_AA)
        return panel

    y = 70
    start_f, end_f = int(annotation["start_frame"]), int(annotation["end_frame"])
    progress = (frame_idx - start_f) / max(end_f - start_f, 1)
    cv2.putText(panel, f"Segment #{annotation.get('id', '?')}", (left, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_BLUE_DARK, 2, cv2.LINE_AA)
    y += 32
    cv2.putText(panel, f"t {annotation.get('start_ts','?')}-{annotation.get('end_ts','?')}s"
                       f"  f {start_f}-{end_f}", (left, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT_MUTED, 1, cv2.LINE_AA)
    y += 24
    bx, bw, bh = left, width - 2 * left, 14
    cv2.rectangle(panel, (bx, y), (bx + bw, y + bh), C_TRACK, -1)
    cv2.rectangle(panel, (bx, y), (bx + int(bw * min(max(progress, 0), 1)), y + bh),
                  C_BLUE, -1)
    cv2.rectangle(panel, (bx, y), (bx + bw, y + bh), C_OUTLINE, 1)
    y += bh + 28

    cv2.rectangle(panel, (14, y - 24), (width - 14, y + 8), C_CARD_BG, -1)
    cv2.rectangle(panel, (14, y - 24), (width - 14, y + 8), C_OUTLINE, 1)
    cv2.putText(panel, f"Scene: {annotation.get('scene', 'n/a')}", (left, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_BLUE_DARK, 2, cv2.LINE_AA)
    y += 36
    cv2.line(panel, (left, y), (width - left, y), C_OUTLINE, 1)
    y += 26
    cv2.putText(panel, "Atomic Actions:", (left, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_BLUE_DARK, 2, cv2.LINE_AA)
    y += 32

    body_x = 44
    body_w = width - body_x - left
    for action in annotation.get("atomic_action", []):
        if y > height - 60:
            break
        color = _hand_color(action.get("hand"))
        cv2.circle(panel, (28, y - 6), 6, color, -1, cv2.LINE_AA)
        label = f"{action.get('verb','?')} -> {action.get('object','?')}"
        for i, line in enumerate(_wrap_text(label, body_w, 0.62, 2)):
            cv2.putText(panel, line, (body_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_TEXT, 2, cv2.LINE_AA)
            y += 26
        conf = float(action.get("confidence", 0) or 0)
        cc = C_BLUE_DARK if conf >= 0.8 else C_TEXT_MUTED if conf >= 0.5 else (40, 40, 210)
        cv2.putText(panel, f"hand: {action.get('hand','n/a')}   conf: {conf:.2f}",
                    (body_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, cc, 1, cv2.LINE_AA)
        y += 24
        for line in _wrap_text(action.get("description", ""), body_w, 0.5, 1):
            if y > height - 50:
                break
            cv2.putText(panel, line, (body_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT_MUTED, 1, cv2.LINE_AA)
            y += 22
        y += 12
    return panel


def _segment_color(seg_id: int) -> Tuple[int, int, int]:
    """Stable colour per segment id from the curated blue/teal/indigo palette."""
    return SEGMENT_PALETTE[int(seg_id) % len(SEGMENT_PALETTE)]


def _text_on(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Pick black or white text for legibility over ``color`` (BGR)."""
    b, g, r = color
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return (30, 25, 20) if lum > 150 else C_WHITE


def _fit_label(text: str, max_w: int, fs: float, th: int) -> Optional[str]:
    """Return ``text`` (or an ellipsised form) if it fits ``max_w`` px, else None.

    Prefers no text when the block is clearly too small: an ellipsised label is
    only returned when at least four leading characters remain visible.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, fs, th)
    if tw <= max_w:
        return text
    for n in range(len(text) - 1, 3, -1):
        cand = text[:n].rstrip() + "..."
        (tw, _), _ = cv2.getTextSize(cand, font, fs, th)
        if tw <= max_w:
            return cand
    return None


def draw_timeline_bar(width: int, height: int, annotations: List[dict],
                      frame_idx: int, total_frames: int) -> np.ndarray:
    """Render a horizontal video-scrubber timeline of annotation segments.

    Each segment is a coloured block spanning its frame range. Blocks wide enough
    to fit are labelled with the (first) atomic action's ``"verb object"`` text;
    blocks too narrow for a legible label are left unlabelled. A blue playhead
    marks ``frame_idx``.
    """
    bar = np.full((height, width, 3), C_PANEL_BG, dtype=np.uint8)
    pad = 12
    track_x0, track_x1 = pad, width - pad
    track_w = track_x1 - track_x0
    track_y0 = 18
    track_y1 = height - 20
    total = max(total_frames - 1, 1)

    cv2.rectangle(bar, (track_x0, track_y0), (track_x1, track_y1), C_TRACK, -1)
    cv2.rectangle(bar, (track_x0, track_y0), (track_x1, track_y1), C_OUTLINE, 1)

    fs, th = 0.42, 1
    block_h = track_y1 - track_y0
    for ann in annotations:
        s = int(ann["start_frame"])
        e = int(ann["end_frame"])
        x_a = track_x0 + int(track_w * s / total)
        x_b = track_x0 + int(track_w * e / total)
        x_b = max(x_b, x_a + 1)
        color = _segment_color(int(ann.get("id", 0)))
        cv2.rectangle(bar, (x_a, track_y0), (x_b, track_y1), color, -1)
        cv2.rectangle(bar, (x_a, track_y0), (x_b, track_y1), C_PANEL_BG, 1)

        actions = ann.get("atomic_action", [])
        if not actions:
            continue
        label = _action_label(actions[0])
        fitted = _fit_label(label, (x_b - x_a) - 8, fs, th)
        if fitted is None:
            continue
        (tw, lh), _ = cv2.getTextSize(fitted, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = x_a + max(3, ((x_b - x_a) - tw) // 2)
        ty = track_y0 + (block_h + lh) // 2
        cv2.putText(bar, fitted, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    _text_on(color), th, cv2.LINE_AA)

    # playhead
    px = track_x0 + int(track_w * min(frame_idx, total) / total)
    cv2.line(bar, (px, track_y0 - 6), (px, track_y1 + 6), C_BLUE_DARK, 2, cv2.LINE_AA)
    cv2.circle(bar, (px, track_y0 - 6), 4, C_BLUE_DARK, -1, cv2.LINE_AA)

    cv2.putText(bar, f"{frame_idx}/{total_frames}", (track_x0, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT_MUTED, 1, cv2.LINE_AA)
    return bar
