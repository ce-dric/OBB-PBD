#!/usr/bin/env python3
"""
Step 1c - block-mask self-leakage probe
---------------------------------------
Hypothesis under test
=====================
``create_block_diff_mask_by_pe_4d(..., causal_attn=False)`` makes the parallel
block **mutually (non-causally) visible**: every coordinate slot attends to every
other slot of the same block. The loss is still the standard shifted
next-token CE (``shift_logits`` / ``shift_labels`` in modeling_qwen2.py), and the
collator keeps the ground-truth coordinate tokens inside ``input_ids``
(``labels = input_ids.clone()``).

Consequence: the logits at slot ``k-1`` -- which are scored against the
ground-truth token of slot ``k`` -- can simply *read* that ground-truth token
through the mutual mask. Teacher forcing would then be near-oracle while real
generation, where those positions hold no ground truth, collapses.

Experiment
==========
Two forward passes on the *same* samples and the *same* weights:

  A. as-is                 -- ground-truth coordinate tokens present in input_ids
  B. coordinate slots overwritten with ``<null>`` -- ground truth removed,
     everything else identical

Read the argmax at the same positions in both passes. If A is accurate and B
collapses, the teacher-forced numbers are leakage, not skill, and the model was
trained on a copy shortcut.

Read-only w.r.t. checkpoints. 1 GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoProcessor, AutoModel
from peft import PeftModel

from obb_pbd.collators import PBDMultiModalCollator
from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.geometry import OrientedBox
from obb_pbd.metrics import angle_error_deg, rotated_iou
from obb_pbd.model_config import OBB_BLOCK_SIZE, assert_block_size, build_obb_config
from obb_pbd.tokens import OBBTokenCodec

MODEL_PATH = "/workspace/models/LocateAnything-3B"
N_COORDS = 6


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)


def read_slots(logits, box_idx: int, coord_start: int) -> list[int] | None:
    """Argmax bins for the six geometry slots, using the shifted-CE alignment."""
    bins = []
    for pos in range(box_idx, box_idx + N_COORDS):
        if pos >= logits.shape[0]:
            return None
        token_id = int(logits[pos].argmax(dim=-1).item())
        bins.append(min(1000, max(0, token_id - coord_start)))
    return bins


def decode(bins, image_size) -> OrientedBox | None:
    try:
        return OBBTokenCodec(bins=1000).decode(tuple(bins), image_size=image_size)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="/workspace/data")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    config = build_obb_config(MODEL_PATH, block_size=OBB_BLOCK_SIZE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer

    box_open = tokenizer.convert_tokens_to_ids("<box>")
    coord_start = tokenizer.convert_tokens_to_ids("<0>")
    null_id = tokenizer.convert_tokens_to_ids("<null>")
    print(f"  <box>={box_open}  <0>={coord_start}  <null>={null_id}")

    base = AutoModel.from_pretrained(
        MODEL_PATH, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    observed = assert_block_size(base, OBB_BLOCK_SIZE, strict=False)
    print(f"  effective language_model block_size = {observed}")
    model = PeftModel.from_pretrained(base, args.checkpoint).to("cuda:0")
    model.eval()
    print(f"  loaded {args.checkpoint}", flush=True)

    dataset = RefGeoOBBDataset(data_dir=args.data_dir, split=args.split, verbose=False)
    collator = PBDMultiModalCollator(processor=processor, ignore_idx=-100)
    dtype = next(model.parameters()).dtype

    hr(f"A = ground truth present in input_ids | B = coord slots -> <null>")

    iou_a, iou_b, ang_a, ang_b = [], [], [], []
    for i in range(min(args.samples, len(dataset))):
        raw = dataset[i]
        gt = dataset.samples[i]["obb"]
        image_size = raw["image"].size

        batch = collator([raw])
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = (value.to(dtype=dtype, device="cuda:0")
                              if torch.is_floating_point(value)
                              else value.to(device="cuda:0"))

        ids = batch["input_ids"][0].tolist()
        if box_open not in ids:
            print(f"--- sample {i}: no <box> in input_ids, skipped")
            continue
        box_idx = ids.index(box_open)

        with torch.no_grad():
            bins_a = read_slots(model(**batch).logits[0], box_idx, coord_start)

        masked = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        masked["input_ids"][0, box_idx + 1: box_idx + 1 + N_COORDS] = null_id
        with torch.no_grad():
            bins_b = read_slots(model(**masked).logits[0], box_idx, coord_start)

        box_a = decode(bins_a, image_size) if bins_a else None
        box_b = decode(bins_b, image_size) if bins_b else None
        ia = rotated_iou(box_a, gt) if box_a else 0.0
        ib = rotated_iou(box_b, gt) if box_b else 0.0
        iou_a.append(ia)
        iou_b.append(ib)

        gt_bins = OBBTokenCodec(bins=1000).encode(gt, image_size=image_size)
        print(f"\n--- sample {i}  image={image_size}")
        print(f"    GT   bins={list(gt_bins)}")
        print(f"    A    bins={bins_a}  IoU={ia:.4f}")
        print(f"    B    bins={bins_b}  IoU={ib:.4f}")
        if box_a:
            ea = angle_error_deg(box_a.theta_deg, gt.theta_deg)
            ang_a.append(ea)
            print(f"    A    theta={box_a.theta_deg:6.1f} (gt {gt.theta_deg:6.1f}) angErr={ea:5.1f}")
        if box_b:
            eb = angle_error_deg(box_b.theta_deg, gt.theta_deg)
            ang_b.append(eb)
            print(f"    B    theta={box_b.theta_deg:6.1f} (gt {gt.theta_deg:6.1f}) angErr={eb:5.1f}")

    hr("SUMMARY")
    n = max(1, len(iou_a))
    print(f"  samples                                : {len(iou_a)}")
    print(f"  mean IoU  A (ground truth in input)    : {sum(iou_a)/n:.4f}")
    print(f"  mean IoU  B (ground truth removed)     : {sum(iou_b)/n:.4f}")
    if ang_a:
        print(f"  mean angle err A                       : {sum(ang_a)/len(ang_a):.2f} deg")
    if ang_b:
        print(f"  mean angle err B                       : {sum(ang_b)/len(ang_b):.2f} deg")
    print("\n  If B collapses towards the real-generation numbers, the parallel block")
    print("  is self-visible and every teacher-forced metric in this project is leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
