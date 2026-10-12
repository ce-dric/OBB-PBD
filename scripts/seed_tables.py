#!/usr/bin/env python3
"""
every number behind the three-seed Tables 2 and 3
========================================================================

Reads the per-record full-test predictions of the three trained arms for seeds 1-3 and the
zero-shot baseline's per-record IoUs, and writes one JSON with, per seed:
  * per-arm metrics (rotIoU and aabbIoU with image-cluster 95% CIs, Acc@0.50, Acc@0.75,
    decoded-only angle MAE, decoded %), and
  * the paired decomposition steps of Table 3 (baseline -> HBB-PBD -> HBB-envelope -> OBB-PBD,
    and the total baseline -> OBB-PBD) with image-cluster CIs,
plus the across-seed mean and sample SD of every quantity. Seed 1 is checked against the
published Table 2/3 values. Bootstrap: 120 test images resampled with replacement, all records
of a drawn image pooled, 10,000 draws, seed 0 (as in obb_pbd.reporting._cluster_ci).

    uv run --no-project --with numpy python scripts/seed_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_seed_deltas import cluster_ci

ROOT = Path(__file__).resolve().parents[1]
OBB_DIR = ROOT / "experiments" / "obb_pbd"
CTRL = ROOT / "experiments" / "hbb_controls"
SEEDS = ROOT / "experiments" / "analysis" / "seeds"
ARMS = {
    "s1": {"OBB": OBB_DIR / "OBB_FULL_TEST.predictions.jsonl",
           "ENVELOPE": CTRL / "HBB_ENVELOPE" / "HBB_ENVELOPE_FULL_TEST.predictions.jsonl",
           "HBBPBD": CTRL / "HBB_PBD" / "HBB_PBD_FULL_TEST.predictions.jsonl"},
    **{tag: {"OBB": SEEDS / f"{tag}_OBB_FULL_TEST.predictions.jsonl",
             "ENVELOPE": SEEDS / f"{tag}_ENVELOPE_FULL_TEST.predictions.jsonl",
             "HBBPBD": SEEDS / f"{tag}_HBBPBD_FULL_TEST.predictions.jsonl"} for tag in ("s2", "s3")},
}
BASELINE = ROOT / "experiments" / "analysis" / "BASELINE_ZEROSHOT_PER_RECORD.jsonl"
OUT = ROOT / "experiments" / "analysis" / "SEED_TABLES.json"


def load(path):
    rows = sorted((json.loads(line) for line in open(path, encoding="utf-8")),
                  key=lambda r: int(r["sample_index"]))
    assert [int(r["sample_index"]) for r in rows] == list(range(6139)), path
    return rows


def arm_metrics(rows, images):
    rot = np.array([float(r["rotated_iou"]) for r in rows])
    aab = np.array([float(r["aabb_iou"]) for r in rows])
    angles = [float(r["angle_error_deg"]) for r in rows if r.get("angle_error_deg") is not None]
    return {"rotIoU": cluster_ci(rot, images), "aabbIoU": cluster_ci(aab, images),
            "acc50": float((rot >= 0.5).mean()), "acc75": float((rot >= 0.75).mean()),
            "angle_mae": float(np.mean(angles)), "decoded_pct": float(100 * np.mean([r["decoded"] for r in rows]))}, rot, aab


def main() -> int:
    base = load(BASELINE)
    images = np.array([r["image_id"] for r in base])
    base_rot = np.array([r["rotated_iou"] for r in base])
    base_aab = np.array([r["aabb_iou"] for r in base])
    report = {"baseline": {"rotIoU": cluster_ci(base_rot, images), "aabbIoU": cluster_ci(base_aab, images),
                           "acc50": float((base_rot >= 0.5).mean()), "acc75": float((base_rot >= 0.75).mean())},
              "seeds": {}}
    for tag, paths in ARMS.items():
        seed = {"arms": {}, "steps": {}}
        vals = {}
        for arm, path in paths.items():
            rows = load(path)
            assert all(r["image_id"] == i for r, i in zip(rows, images)), (tag, arm)
            seed["arms"][arm], vals[arm + "_rot"], vals[arm + "_aab"] = arm_metrics(rows, images)
        vals["BASE_rot"], vals["BASE_aab"] = base_rot, base_aab
        for name, (a, b) in {"finetune": ("HBBPBD", "BASE"), "format_switch": ("ENVELOPE", "HBBPBD"),
                             "orientation": ("OBB", "ENVELOPE"), "total": ("OBB", "BASE")}.items():
            seed["steps"][name] = {"rotIoU": cluster_ci(vals[a + "_rot"] - vals[b + "_rot"], images),
                                   "aabbIoU": cluster_ci(vals[a + "_aab"] - vals[b + "_aab"], images)}
        report["seeds"][tag] = seed
        print(tag, {k: round(v["rotIoU"]["mean"], 4) for k, v in seed["arms"].items()},
              {k: round(v["rotIoU"]["mean"], 4) for k, v in seed["steps"].items()}, flush=True)

    def agg(getter):
        xs = np.array([getter(report["seeds"][t]) for t in ("s1", "s2", "s3")])
        return {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)), "values": xs.tolist()}

    summary = {"arms": {}, "steps": {}}
    for arm in ("OBB", "ENVELOPE", "HBBPBD"):
        summary["arms"][arm] = {m: agg(lambda s, m=m, arm=arm: (s["arms"][arm][m]["mean"]
                                                                 if isinstance(s["arms"][arm][m], dict)
                                                                 else s["arms"][arm][m]))
                                for m in ("rotIoU", "aabbIoU", "acc50", "acc75", "angle_mae", "decoded_pct")}
    for step in ("finetune", "format_switch", "orientation", "total"):
        summary["steps"][step] = {m: agg(lambda s, m=m, step=step: s["steps"][step][m]["mean"])
                                  for m in ("rotIoU", "aabbIoU")}
    report["across_seeds"] = summary

    # seed 1 must reproduce the submitted Table 2 / Table 3
    s1 = report["seeds"]["s1"]
    assert round(s1["arms"]["OBB"]["rotIoU"]["mean"], 4) == 0.3124
    assert round(s1["arms"]["OBB"]["aabbIoU"]["mean"], 4) == 0.3441
    assert round(s1["steps"]["orientation"]["rotIoU"]["mean"], 4) == 0.0597
    assert round(s1["steps"]["format_switch"]["rotIoU"]["mean"], 4) == 0.0004
    assert round(s1["steps"]["finetune"]["rotIoU"]["mean"], 4) == 0.1271
    assert round(s1["steps"]["total"]["rotIoU"]["mean"], 4) == 0.1872
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("seed 1 reproduces the submitted Tables 2-3; wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
