#!/usr/bin/env python3
"""
Figure 2 (qualitative examples) renderer
=======================================================================================

The submitted Figure 2 composites were assembled by hand and had no generating script.
The first version's red prediction / green ground-truth outlines were hard to see on
parking-lot backgrounds; this version uses thicker, higher-contrast lines and adds
coordinate deviation values. This script regenerates all three panels reproducibly:

  * every outline is drawn over a black halo, so it reads on light and dark ground;
  * colors: ground truth neon green (solid), OBB-PBD magenta (dashed), zero-shot
    LocateAnything baseline cyan (dotted), crop marker yellow;
  * every panel uses the same crop aspect ratio, so the three composites have identical
    height without post-hoc cropping;
  * per-panel deviations (prediction minus ground truth, in original 4000x2250 pixels)
    are written to JSON for the table under the figure.

Inputs are the same artifacts the submitted figure used: the selected OBB-PBD checkpoint's (seed 1)
full-test predictions (targets and OBB-PBD boxes in the 1024x576 model frame) and
FIG2_BASELINE_3PANEL.json (the zero-shot baseline's corner bins for the three records).

    uv run --no-project --with pillow --with numpy --with shapely \\
        python scripts/render_fig2_qualitative.py --images DIR --baseline FIG2_BASELINE_3PANEL.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from obb_pbd.geometry import OrientedBox  # noqa: E402
from obb_pbd.metrics import aabb_iou, angle_error_deg, obb_to_polygon, rotated_iou  # noqa: E402

PANELS = (("a", 8, "fig2a_success_composite.jpg"),
          ("b", 1322, "fig2b_success_composite.jpg"),
          ("c", 1811, "fig2c_failure_composite.jpg"))
MODEL_W, MODEL_H = 1024.0, 576.0
ORIG_W, ORIG_H = 4000, 2250
OUT_W = 1400                      # composite width, as in the submitted figure
CROP_ASPECT = 2.0                 # width / height of every enlarged crop
SEPARATOR = 24

GT = (57, 255, 20)                # neon green
OURS = (255, 0, 200)              # magenta
BASE = (0, 213, 255)              # cyan
CROP = (255, 212, 0)              # yellow
HALO = (0, 0, 0)


def scaled(box: OrientedBox) -> OrientedBox:
    sx, sy = ORIG_W / MODEL_W, ORIG_H / MODEL_H
    return OrientedBox(box.cx * sx, box.cy * sy, box.w * sx, box.h * sy, box.theta_deg)


def baseline_box(bins) -> OrientedBox:
    x1, y1, x2, y2 = (b / 1000.0 for b in bins)
    x1, x2 = sorted((x1 * ORIG_W, x2 * ORIG_W))
    y1, y2 = sorted((y1 * ORIG_H, y2 * ORIG_H))
    w, h = x2 - x1, y2 - y1
    if w >= h:
        return OrientedBox((x1 + x2) / 2, (y1 + y2) / 2, w, h, 0.0)
    return OrientedBox((x1 + x2) / 2, (y1 + y2) / 2, h, w, 90.0)


def segments(points, pattern):
    """Yield (start, end) sub-segments of a closed polygon for a dash pattern (on, off)."""
    on, off = pattern
    pts = list(points) + [points[0]]
    remaining, drawing = on, True     # length left in the current dash or gap
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while length - pos > 1e-9:
            step = min(remaining, length - pos)
            if drawing:
                t0, t1 = pos / length, (pos + step) / length
                yield ((x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                       (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1))
            pos += step
            remaining -= step
            if remaining <= 1e-9:
                drawing = not drawing
                remaining = on if drawing else off


def outline(draw, points, color, width, style):
    points = [tuple(map(float, p)) for p in points]
    if style == "solid":
        parts = list(zip(points, points[1:] + points[:1]))
    else:
        pattern = (3.2 * width, 2.0 * width) if style == "dashed" else (0.9 * width, 1.4 * width)
        parts = list(segments(points, pattern))
    for layer_color, layer_width in ((HALO, width + max(4, width // 2) * 2), (color, width)):
        for a, b in parts:
            draw.line([a, b], fill=layer_color, width=int(layer_width))
        if style == "solid":
            for p in points:
                r = layer_width / 2
                draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=layer_color)


def crop_window(boxes, margin=2.2):
    pts = np.vstack([obb_to_polygon(b) for b in boxes])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * margin, (y1 - y0) * margin * CROP_ASPECT, 360.0)
    h = w / CROP_ASPECT
    left = min(max(cx - w / 2, 0), ORIG_W - w)
    top = min(max(cy - h / 2, 0), ORIG_H - h)
    return left, top, w, h


def render(image_path, gt, ours, base, out_path):
    frame = Image.open(image_path).convert("RGB")
    assert frame.size == (ORIG_W, ORIG_H), frame.size
    left, top, cw, ch = crop_window([gt, ours, base])

    # top: full scene with crop marker and small outlines
    scene_scale = OUT_W / ORIG_W
    scene = frame.resize((OUT_W, round(ORIG_H * scene_scale)), Image.LANCZOS)
    d = ImageDraw.Draw(scene)
    for box, color, style in ((base, BASE, "dotted"), (gt, GT, "solid"), (ours, OURS, "dashed")):
        outline(d, obb_to_polygon(box) * scene_scale, color, 4, style)
    rect = np.array([[left, top], [left + cw, top], [left + cw, top + ch], [left, top + ch]]) * scene_scale
    outline(d, rect, CROP, 5, "solid")

    # bottom: enlarged crop
    crop_scale = OUT_W / cw
    crop = frame.crop((round(left), round(top), round(left + cw), round(top + ch)))
    crop = crop.resize((OUT_W, round(OUT_W / CROP_ASPECT)), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    offset = np.array([left, top])
    for box, color, style in ((base, BASE, "dotted"), (gt, GT, "solid"), (ours, OURS, "dashed")):
        outline(d, (obb_to_polygon(box) - offset) * crop_scale, color, 9, style)

    canvas = Image.new("RGB", (OUT_W, scene.height + SEPARATOR + crop.height), (255, 255, 255))
    canvas.paste(scene, (0, 0))
    canvas.paste(crop, (0, scene.height + SEPARATOR))
    canvas.save(out_path, quality=92)
    return canvas.size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="directory with the original AVVG JPGs")
    parser.add_argument("--baseline", required=True, help="FIG2_BASELINE_3PANEL.json")
    parser.add_argument("--predictions",
                        default="experiments/obb_pbd/OBB_FULL_TEST.predictions.jsonl")
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument("--deviations-out", default="experiments/analysis/FIG2_DEVIATIONS.json")
    args = parser.parse_args()

    rows = {json.loads(l)["sample_index"]: json.loads(l) for l in open(args.predictions, encoding="utf-8")}
    base_rows = {r["index"]: r for r in json.load(open(args.baseline, encoding="utf-8"))}
    report = {}
    for tag, index, name in PANELS:
        row, brow = rows[index], base_rows[index]
        assert row["image_id"] == brow["image_id"]
        t, p = row["target"], row["prediction"]
        gt = scaled(OrientedBox(t["cx"], t["cy"], t["w"], t["h"], t["theta_deg"]))
        ours = scaled(OrientedBox(p["cx"], p["cy"], p["w"], p["h"], p["theta_deg"]))
        base = baseline_box(brow["coords_raw_bins"])
        # the rendered geometry must reproduce the numbers quoted in the caption
        assert abs(rotated_iou(ours, gt) - row["rotated_iou"]) < 1e-6
        assert abs(rotated_iou(base, gt) - brow["iou_rotated"]) < 1e-3
        assert abs(aabb_iou(base, gt) - brow["iou_aabb"]) < 1e-3
        size = render(Path(args.images) / row["image_id"], gt, ours, base, Path(args.out_dir) / name)
        report[tag] = {
            "record": index, "image_id": row["image_id"], "composite_size": size,
            "ours": {"d_cx": ours.cx - gt.cx, "d_cy": ours.cy - gt.cy, "d_w": ours.w - gt.w,
                     "d_h": ours.h - gt.h, "angle_error_deg": angle_error_deg(ours.theta_deg, gt.theta_deg),
                     "rotIoU": rotated_iou(ours, gt)},
            "baseline": {"d_cx": base.cx - gt.cx, "d_cy": base.cy - gt.cy,
                         "rotIoU": rotated_iou(base, gt), "aabbIoU": aabb_iou(base, gt)},
            "gt": {"cx": gt.cx, "cy": gt.cy, "w": gt.w, "h": gt.h, "theta_deg": gt.theta_deg},
        }
        print(tag, index, size, {k: round(v, 2) for k, v in report[tag]["ours"].items()},
              {k: round(v, 2) for k, v in report[tag]["baseline"].items()})
    Path(args.deviations_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.deviations_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
