#!/usr/bin/env python3
"""
Direct parallel-box decoding evaluator
======================================

Why this exists
---------------
The upstream ``modeling_locateanything.generate()`` MTP path fails on a checkpoint
that memorises 16 samples to loss 0.0000: it emits a structurally valid 8-token block only 18.8% of the time and scores
mIoU 0.0248, while a plain eval-mode forward on the *same weights* with the same
mask-token block reproduces the coordinates at mIoU 0.98
(``scripts/probe_block_leakage.py``). The weights are fine; the upstream
generation wrapper is not.

Rather than keep chasing that wrapper, this evaluator performs the parallel decode
directly, exactly as training defines it:

    prompt ... <|im_start|>assistant\\n   <box>  M M M M M M M

one forward pass, then read the argmax at the six geometry slots. This is what
parallel box decoding *is*: a single step, no autoregressive coordinate loop, so
it is also the fastest possible inference and gives an honest latency number.

Decode failures are scored IoU 0.0, never substituted with the target.

Two model-free / adapter-free rows make the table self-contained:

* ``NAME=base`` runs the untrained upstream weights through this same decoder, so the
  baseline no longer has to come from AVVG_BASE_GENERATE, which used a different metric
  (AABB-IoU) *and* a different, broken code path (generate()).
* ``NAME=hbb_oracle`` predicts the axis-aligned envelope of the ground truth. No weights,
  no GPU. It is the ceiling any horizontal-box method can reach on rotated targets.

Usage:
    python scripts/eval_pbd_direct.py --checkpoint NAME=/path/to/ckpt [...]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

from obb_pbd.collators import PBDMultiModalCollator
from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.eval_window import evaluation_window
from obb_pbd.metrics import aabb_iou, angle_error_deg, obb_to_aabb, rotated_iou
from obb_pbd.model_config import (
    HBB_BLOCK_SIZE,
    HBB_N_COORDS,
    OBB_BLOCK_SIZE,
    OBB_N_COORDS,
    assert_block_size,
    build_obb_config,
)
from obb_pbd.predictions import (
    DatasetIndexView,
    make_prediction_record,
    write_prediction_jsonl,
)
from obb_pbd.splits import load_partition_indices
from obb_pbd.tokens import OBBTokenCodec

MODEL_PATH = "/workspace/models/LocateAnything-3B"
EXPERIMENTS = "/workspace/experiments"

# Passing `--checkpoint NAME=base` scores the untrained upstream weights through
# this same decoder. Without it the only AVVG baseline we had was AVVG_BASE_GENERATE,
# which is AABB-IoU measured through the broken generate() MTP path, so it could not
# be placed next to a trained rotated-IoU number in a table.
BASE_WEIGHTS_SENTINEL = "base"

# `--checkpoint NAME=hbb_oracle` needs no weights at all: it predicts the axis-aligned
# envelope of the ground truth. That is the best any horizontal-box method can do on
# this data, so it is the ceiling the OBB extension has to be judged against.
HBB_ORACLE_SENTINEL = "hbb_oracle"

MODEL_FREE_SENTINELS = (HBB_ORACLE_SENTINEL,)

# One model fits comfortably on one card (peak was 21.3 GiB of 48 GiB during 3-GPU DDP
# training), and this evaluator holds a single checkpoint at a time, so the way to use the
# other two cards is to launch one process per card with a slice of the checkpoint list
# rather than to shard a model. `--device` exists for that.
DEFAULT_DEVICE = "cuda:0"

DROP_KEYS = ("labels", "prompt_end_indices", "position_ids", "sin_pos_indices",
             "cos_pos_indices", "cx_pos_indices", "cy_pos_indices",
             "w_pos_indices", "h_pos_indices")


def move(batch, dtype, device=DEFAULT_DEVICE):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = (value.to(dtype=dtype, device=device)
                        if torch.is_floating_point(value)
                        else value.to(device=device))
        else:
            out[key] = value
    return out


def load_model(checkpoint: str, device: str = DEFAULT_DEVICE, block_size: int = OBB_BLOCK_SIZE):
    config = build_obb_config(MODEL_PATH, block_size=block_size)
    base = AutoModel.from_pretrained(
        MODEL_PATH, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device,
    )
    assert_block_size(base, block_size)
    if checkpoint == BASE_WEIGHTS_SENTINEL:
        # No adapter: the point is to read what the untrained weights put in the six
        # geometry slots, on the same decoder the trained rows use.
        base.eval()
        return base, base
    model = PeftModel.from_pretrained(base, checkpoint).to(device)
    model.eval()
    return base, model


def build_block_inputs(batch, prompt_end: int, box_open: int, mask_id: int,
                       block_size: int = OBB_BLOCK_SIZE):
    """prompt + <box> + (block_size - 1) mask tokens, nothing else."""
    ids = batch["input_ids"][:, :prompt_end]
    suffix = torch.full((ids.shape[0], block_size - 1), mask_id,
                        dtype=ids.dtype, device=ids.device)
    trigger = torch.full((ids.shape[0], 1), box_open,
                         dtype=ids.dtype, device=ids.device)
    new_ids = torch.cat([ids, trigger, suffix], dim=1)

    out = {k: v for k, v in batch.items() if k not in DROP_KEYS}
    out["input_ids"] = new_ids
    mask = out.get("attention_mask")
    if isinstance(mask, torch.Tensor) and mask.dim() == 2:
        out["attention_mask"] = torch.ones_like(new_ids)
    return out, prompt_end


def decode_block_prediction(logits, box_idx: int, codec, image_size, coord_start: int,
                            n_coords: int = OBB_N_COORDS):
    """Read one block's geometry slots and turn them into an OrientedBox.

    This is the whole of parallel box decoding on the read side, factored out so that
    figures are drawn from the same boxes this evaluator scores (Figure 2 is rendered from
    its prediction records).
    A figure produced by a second, parallel implementation of this arithmetic would not
    be evidence for the table.

    ``n_coords`` selects the codec method: 6 -> ``codec.decode`` (cx,cy,w,h,sin,cos),
    4 -> ``codec.decode_corners`` (x1,y1,x2,y2, the matched HBB-PBD baseline). ``logits``
    is the ``[seq, vocab]`` slice for a single sequence. Returns ``None`` when the block
    runs past the end of the sequence or the codec rejects the bins, which the caller
    must score as IoU 0.0 rather than substituting the target.
    """
    bins = []
    for pos in range(box_idx, box_idx + n_coords):
        if pos >= logits.shape[0]:
            return None, []
        token_id = int(logits[pos].argmax(dim=-1).item())
        bins.append(min(1000, max(0, token_id - coord_start)))

    decode_fn = codec.decode_corners if n_coords == 4 else codec.decode
    try:
        return decode_fn(tuple(bins), image_size=image_size), bins
    except Exception:
        return None, bins


def evaluate(model, dataset, collator, codec, args, box_open, mask_id, coord_start,
            checkpoint_name: str, block_size: int = OBB_BLOCK_SIZE,
            n_coords: int = OBB_N_COORDS):
    ious, aabb_ious, angles, latencies = [], [], [], []
    prediction_records = []
    decoded = 0
    start, stop = evaluation_window(len(dataset), args.start_index, args.max_eval)
    dtype = next(model.parameters()).dtype

    for index in range(start, stop):
        raw = dataset[index]
        target = dataset.samples[index]["obb"]
        image_size = raw["image"].size

        batch = move(collator([raw]), dtype, args.device)
        prompt_end = int(batch["prompt_end_indices"][0].item())
        inputs, box_idx = build_block_inputs(batch, prompt_end, box_open, mask_id,
                                             block_size=block_size)

        started = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        latencies.append((time.perf_counter() - started) * 1000.0)

        prediction, bins = decode_block_prediction(logits, box_idx, codec, image_size,
                                                   coord_start, n_coords=n_coords)
        if prediction is not None:
            decoded += 1

        if prediction is None:
            ious.append(0.0)
            aabb_ious.append(0.0)
        else:
            ious.append(rotated_iou(prediction, target))
            aabb_ious.append(aabb_iou(prediction, target))
            angles.append(angle_error_deg(prediction.theta_deg, target.theta_deg))

        source_index = (dataset.source_index(index)
                        if hasattr(dataset, "source_index") else index)
        prediction_records.append(make_prediction_record(
            sample_index=source_index,
            sample=dataset.samples[index],
            target=target,
            prediction=prediction,
            checkpoint=checkpoint_name,
            bins=bins,
        ))

        completed = index - start + 1
        if args.progress and completed % args.progress == 0:
            print(f"      {completed}/{stop - start} running mIoU="
                  f"{sum(ious) / len(ious):.4f}", flush=True)

    return summarise(ious, aabb_ious, angles, latencies, decoded), prediction_records


def evaluate_hbb_oracle(dataset, args):
    """The horizontal-box ceiling: predict the axis-aligned envelope of the target.

    Needs no model and no image, so it runs on CPU. Its rotated IoU is exactly
    area(OBB)/area(AABB) per sample, which turns the over-coverage of a horizontal box
    into a score directly comparable with the trained rows.
    Latency is left at zero because there is nothing to time.
    """
    ious, aabb_ious, angles = [], [], []
    start, stop = evaluation_window(len(dataset), args.start_index, args.max_eval)
    for index in range(start, stop):
        target = dataset.samples[index]["obb"]
        prediction = obb_to_aabb(target)
        ious.append(rotated_iou(prediction, target))
        aabb_ious.append(aabb_iou(prediction, target))
        angles.append(angle_error_deg(prediction.theta_deg, target.theta_deg))
    return summarise(ious, aabb_ious, angles, [], len(ious))


def summarise(ious, aabb_ious, angles, latencies, decoded):
    count = max(1, len(ious))
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    return {
        "samples": len(ious),
        "decoded_pct": 100.0 * decoded / count,
        "mIoU": sum(ious) / count,
        "median_IoU": statistics.median(ious) if ious else 0.0,
        "mIoU_aabb": sum(aabb_ious) / count,
        "median_IoU_aabb": statistics.median(aabb_ious) if aabb_ious else 0.0,
        "acc25": sum(1 for v in ious if v >= 0.25) / count,
        "acc50": sum(1 for v in ious if v >= 0.5) / count,
        "acc75": sum(1 for v in ious if v >= 0.75) / count,
        "acc50_aabb": sum(1 for v in aabb_ious if v >= 0.5) / count,
        "angle_mae": statistics.mean(angles) if angles else float("nan"),
        "latency_ms": mean_latency,
        "boxes_per_sec": (1000.0 / mean_latency) if mean_latency else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-eval", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0,
                        help="First sample in the split to evaluate; useful for a "
                             "disjoint checkpoint-selection slice")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="NAME=/path/to/checkpoint, or NAME=base for the untrained "
                             "upstream weights, or NAME=hbb_oracle for the horizontal-box "
                             "ceiling (no model, CPU only)")
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--subset", default="avvg",
                        help="RefGeo subset; avvg is the only oriented one whose "
                             "images ship with RefGeo")
    parser.add_argument("--target-mode", default="obb",
                        choices=("obb", "hbb_envelope", "hbb_corner"),
                        help="Must match how every --checkpoint in this invocation was "
                             "trained: obb/hbb_envelope use the 6-slot/8-token OBB "
                             "block; hbb_corner uses the original 4-slot upstream "
                             "x1/y1/x2/y2 block (block_size=6). Mixing checkpoints "
                             "trained under different modes in one run is not "
                             "supported -- run hbb_corner checkpoints separately.")
    parser.add_argument("--corner-filter", default="auto", choices=("auto", "on", "off"),
                        help="Whether the dataset drops the ~9.8%% train / ~4%% test "
                             "records whose envelope isn't corner-encodable. 'auto' "
                             "(default) ties this to --target-mode==hbb_corner, which "
                             "is what training does and is correct for scoring an "
                             "hbb_corner checkpoint against a record-aligned GT. But "
                             "auto makes hbb_corner's test[start:start+max_eval] a "
                             "DIFFERENT sample of images than every other arm's same "
                             "slice (a report slice picks "
                             "up an extra image, selection slice diverges entirely "
                             "past index 400) -- not comparable in one table. Pass "
                             "'off' to force the unfiltered dataset (matching "
                             "baseline/hbb_envelope/obb's records exactly) when "
                             "cross-arm comparison is the point; decode failures on "
                             "the un-encodable ~4%% are then scored 0.0 like any other "
                             "decode failure, which is the fair way to charge the "
                             "4-slot format for what it structurally cannot express.")
    parser.add_argument("--out", default=f"{EXPERIMENTS}/PBD_DIRECT_BENCHMARK")
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="cuda:N. Launch one process per card with a slice of the\n                             checkpoint list to use all three GPUs")
    parser.add_argument("--split-manifest", default="",
                        help="Train-derived image split manifest for fit/validation evaluation")
    parser.add_argument("--partition", choices=("fit", "validation"), default=None)
    parser.add_argument("--predictions-out", default="",
                        help="Write per-record JSONL; requires exactly one checkpoint")
    args = parser.parse_args()

    if not args.checkpoint:
        print("  nothing to evaluate: pass --checkpoint NAME=/path")
        return 1
    if args.predictions_out and len(args.checkpoint) != 1:
        parser.error("--predictions-out requires exactly one --checkpoint")
    if bool(args.split_manifest) != bool(args.partition):
        parser.error("--split-manifest and --partition must be supplied together")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    box_open = tokenizer.convert_tokens_to_ids("<box>")
    coord_start = tokenizer.convert_tokens_to_ids("<0>")
    mask_id = getattr(tokenizer, "convert_tokens_to_ids")("<mask>")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        from obb_pbd.collators import DEFAULT_MASK_TOKEN_ID
        mask_id = DEFAULT_MASK_TOKEN_ID

    # Ground truth is always the TRUE oriented box, regardless of --target-mode --
    # do NOT pass target_mode here. Every arm (baseline, obb, hbb_envelope,
    # hbb_corner) must be scored against the same GT for rotIoU/aabbIoU to be
    # comparable in one table (see run_hbb_control_eval.sh's header comment,
    # which established this for hbb_envelope; hbb_corner follows the same rule).
    # --target-mode here only selects the MODEL's block structure (below), i.e.
    # how to read its prediction, never what the target is.
    #
    # hbb_corner IS the one exception to "the dataset doesn't need to know
    # target_mode": its training set additionally drops ~9.8% of records whose
    # envelope isn't corner-encodable (dataset.py's corner-frame check), so
    # "record index N" in a plain obb-mode dataset is NOT the same underlying
    # record as "record index N" in the training run's hbb_corner-mode dataset
    # once any earlier record differs between the two filters -- found via
    # scripts/debug_hbb_corner_positions.py: without this,
    # index-misaligned (wrong-record) ground truth silently produced near-zero
    # IoU even for a checkpoint that had perfectly memorized its actual targets.
    # require_corner_encodable=True reproduces hbb_corner's exact filtered
    # record sequence while keeping target_mode="obb" (true rotated geometry).
    if args.corner_filter == "auto":
        require_corner_encodable = (args.target_mode == "hbb_corner")
    else:
        require_corner_encodable = (args.corner_filter == "on")
    dataset = RefGeoOBBDataset(data_dir=args.data_dir, split=args.split,
                               subset=args.subset, verbose=False,
                               require_corner_encodable=require_corner_encodable)
    if args.split_manifest:
        partition_indices = load_partition_indices(
            dataset, args.split_manifest, args.partition
        )
        dataset = DatasetIndexView(dataset, partition_indices)
    # n_geometry_slots only affects training-time masking; at eval time
    # build_block_inputs() rebuilds the block from scratch and DROP_KEYS discards
    # the collator's own pos_indices, so a fixed 6-slot collator is safe to reuse
    # for prompt_end_indices regardless of target_mode (verified: the block-token
    # bounds check inside PBDMultiModalCollator simply fails closed for a 4-slot
    # block, leaving pos_indices at -1, which nothing here reads).
    collator = PBDMultiModalCollator(processor=processor, ignore_idx=-100)
    codec = OBBTokenCodec(bins=1000)

    block_size = HBB_BLOCK_SIZE if args.target_mode == "hbb_corner" else OBB_BLOCK_SIZE
    n_coords = HBB_N_COORDS if args.target_mode == "hbb_corner" else OBB_N_COORDS

    print("=" * 100)
    print("  Direct parallel-box decoding (single forward, failures scored 0)")
    print("=" * 100)
    print(f"  split={args.split}  target_mode={args.target_mode}  "
          f"block_size={block_size}  n_coords={n_coords}  "
          f"corner_filter={args.corner_filter} (require_corner_encodable="
          f"{require_corner_encodable})  dataset_len={len(dataset)}  "
          f"<box>={box_open}  mask={mask_id}  <0>={coord_start}", flush=True)

    results = {}
    output_records = None
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if path in MODEL_FREE_SENTINELS:
            print(f"\n  --- {name} (no model) ---", flush=True)
            metrics = evaluate_hbb_oracle(dataset, args)
        elif path == BASE_WEIGHTS_SENTINEL:
            print(f"\n  --- {name} (untrained upstream weights) ---", flush=True)
            base, model = load_model(path, args.device, block_size=block_size)
            try:
                metrics, records = evaluate(
                    model, dataset, collator, codec, args,
                    box_open, mask_id, coord_start, checkpoint_name=name,
                    block_size=block_size, n_coords=n_coords,
                )
                if args.predictions_out:
                    output_records = records
            finally:
                del model, base
                torch.cuda.empty_cache()
        else:
            if not path or not Path(path).exists():
                print(f"  skipping {name}: {path} missing")
                continue
            print(f"\n  --- {name} ---", flush=True)
            base, model = load_model(path, args.device, block_size=block_size)
            try:
                metrics, records = evaluate(
                    model, dataset, collator, codec, args,
                    box_open, mask_id, coord_start, checkpoint_name=name,
                    block_size=block_size, n_coords=n_coords,
                )
                if args.predictions_out:
                    output_records = records
            finally:
                del model, base
                torch.cuda.empty_cache()
        metrics["checkpoint"] = path
        results[name] = metrics
        print(f"      mIoU={metrics['mIoU']:.4f} median={metrics['median_IoU']:.4f} "
              f"aabbIoU={metrics['mIoU_aabb']:.4f} "
              f"Acc@.25={100 * metrics['acc25']:.1f}% Acc@.5={100 * metrics['acc50']:.1f}% "
              f"Acc@.75={100 * metrics['acc75']:.1f}% angMAE={metrics['angle_mae']:.2f} "
              f"decoded={metrics['decoded_pct']:.1f}% {metrics['boxes_per_sec']:.2f} BPS",
              flush=True)

    header = (f"{'model':<28}{'mIoU':>8}{'med':>8}{'aabbIoU':>9}{'A@.25':>8}{'A@.5':>8}"
              f"{'A@.75':>8}{'angMAE':>8}{'dec%':>8}{'BPS':>7}")
    lines = ["=" * 108,
             "  Direct parallel-box decoding (single forward, failures scored 0)",
             "  mIoU is rotated IoU; aabbIoU is the axis-aligned envelope IoU, the metric "
             "AVVG_BASE_GENERATE used",
             "=" * 108, header, "-" * 108]
    for name, metric in results.items():
        lines.append(
            f"{name:<28}{metric['mIoU']:>8.4f}{metric['median_IoU']:>8.4f}"
            f"{metric['mIoU_aabb']:>9.4f}"
            f"{100 * metric['acc25']:>7.1f}%{100 * metric['acc50']:>7.1f}%"
            f"{100 * metric['acc75']:>7.1f}%{metric['angle_mae']:>8.2f}"
            f"{metric['decoded_pct']:>7.1f}%{metric['boxes_per_sec']:>7.2f}"
        )
    lines.append("=" * 108)
    report = "\n".join(lines)
    print("\n" + report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(report + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2) + "\n",
        encoding="utf-8")
    if args.predictions_out:
        if output_records is None:
            raise RuntimeError("no prediction records were produced")
        write_prediction_jsonl(args.predictions_out, output_records)
        print(f"  wrote {len(output_records):,} predictions to {args.predictions_out}")
    print(f"\n  wrote {out.with_suffix('.txt')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
