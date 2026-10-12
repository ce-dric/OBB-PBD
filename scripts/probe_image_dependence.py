#!/usr/bin/env python3
"""
Diagnostic: is the OBB block actually conditioned on the image?
===============================================================

Why this exists
---------------
A low score has two very different explanations: the recipe is too weak, or the
coordinate slots do not use the image at all. An open attention mask (see
``probe_block_visibility.py``) is not proof that the image features are used,
and a small overfit does not settle it either: every sample carries its own
referring expression through
``Locate a single instance that matches the following description: {question}.``
(``dataset.py``), so a few samples are separable from text alone and can be
memorised without ever reading a pixel.

Experiment
----------
Same weights, same test samples, four conditions:

  matched          image i + prompt i          (the ordinary evaluation)
  image_shuffled   image perm(i) + prompt i    (image destroyed, text intact)
  prompt_shuffled  image i + prompt perm(i)    (text destroyed, image intact)
  mean_box         no model at all: the train-split mean OBB, every time

Reading
-------
  matched ~= image_shuffled           -> hypothesis B. The image is unused.
  matched ~= prompt_shuffled          -> the referring expression is unused too.
  matched ~= mean_box                 -> the model only learned a location prior.
  matched >> both shuffles and mean   -> hypothesis A. The path works, undertrained.

``bin_std`` is the per-slot standard deviation of the predicted bins across
samples. If it collapses towards 0 the model emits one constant box regardless of
input, which is the signature of a prior-only solution.

Read-only w.r.t. checkpoints. 1 GPU, no training.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

from obb_pbd.collators import DEFAULT_MASK_TOKEN_ID, PBDMultiModalCollator
from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.image_dependence import (
    intervention_variants,
    seeded_derangement,
    summarise,
)
from obb_pbd.metrics import angle_error_deg, rotated_iou
from obb_pbd.model_config import (
    OBB_BLOCK_SIZE,
    OBB_N_COORDS,
    assert_block_size,
    build_obb_config,
)
from obb_pbd.tokens import OBBTokenCodec

MODEL_PATH = "/workspace/models/LocateAnything-3B"
EXPERIMENTS = "/workspace/experiments"

DROP_KEYS = ("labels", "prompt_end_indices", "position_ids", "sin_pos_indices",
             "cos_pos_indices", "cx_pos_indices", "cy_pos_indices",
             "w_pos_indices", "h_pos_indices")

CONDITIONS = ("matched", "image_shuffled", "prompt_shuffled")


def hr(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100, flush=True)


def move(batch, dtype, device="cuda:0"):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = (value.to(dtype=dtype, device=device)
                        if torch.is_floating_point(value)
                        else value.to(device=device))
        else:
            out[key] = value
    return out


def build_block_inputs(batch, prompt_end: int, box_open: int, mask_id: int):
    """prompt + <box> + (block_size - 1) mask tokens, exactly as eval_pbd_direct."""
    ids = batch["input_ids"][:, :prompt_end]
    suffix = torch.full((ids.shape[0], OBB_BLOCK_SIZE - 1), mask_id,
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


def read_bins(logits, box_idx: int, coord_start: int) -> list[int] | None:
    """Argmax bins for the six geometry slots under the shifted-CE alignment."""
    bins = []
    for pos in range(box_idx, box_idx + OBB_N_COORDS):
        if pos >= logits.shape[0]:
            return None
        token_id = int(logits[pos].argmax(dim=-1).item())
        bins.append(min(1000, max(0, token_id - coord_start)))
    return bins


def mean_box_baseline(args, codec, test_dataset):
    """Train-split mean OBB in normalised space, scored against every test target."""
    train = RefGeoOBBDataset(data_dir=args.data_dir, split="train",
                             subset=args.subset,
                             max_samples=args.prior_samples, verbose=False)
    if len(train) == 0:
        print("  mean_box: train split unavailable, skipped")
        return None

    cx, cy, w, h, sin2t, cos2t = [], [], [], [], [], []
    import math
    for sample in train.samples:
        try:
            width, height = Image.open(sample["img_path"]).size
        except Exception:
            continue
        box = sample["obb"]
        cx.append(box.cx / width)
        cy.append(box.cy / height)
        w.append(box.w / width)
        h.append(box.h / height)
        angle = math.radians(2.0 * box.theta_deg)
        sin2t.append(math.sin(angle))
        cos2t.append(math.cos(angle))

    if not cx:
        print("  mean_box: no usable train samples, skipped")
        return None

    bins = (
        codec._encode_unit(min(1.0, max(0.0, statistics.mean(cx)))),
        codec._encode_unit(min(1.0, max(0.0, statistics.mean(cy)))),
        codec._encode_unit(min(1.0, max(0.0, statistics.mean(w)))),
        codec._encode_unit(min(1.0, max(0.0, statistics.mean(h)))),
        codec._encode_signed(min(1.0, max(-1.0, statistics.mean(sin2t)))),
        codec._encode_signed(min(1.0, max(-1.0, statistics.mean(cos2t)))),
    )
    print(f"  mean_box prior from {len(cx):,} train samples -> bins {list(bins)}",
          flush=True)

    limit = (len(test_dataset) if args.max_eval == 0
             else min(args.max_eval, len(test_dataset)))
    ious, angles, rows, image_ids = [], [], [], []
    for index in range(limit):
        sample = test_dataset.samples[index]
        image_ids.append(sample["image_id"])
        target = sample["obb"]
        try:
            image_size = Image.open(sample["img_path"]).size
            prediction = codec.decode(bins, image_size=image_size)
        except Exception:
            ious.append(0.0)
            continue
        ious.append(rotated_iou(prediction, target))
        angles.append(angle_error_deg(prediction.theta_deg, target.theta_deg))
        rows.append(list(bins))

    result = summarise(
        ious, angles, rows, len(rows), limit, image_ids=image_ids,
        n_boot=args.n_boot, bootstrap_seed=args.bootstrap_seed)
    result["bins"] = list(bins)
    return result


def load_model(checkpoint: str):
    config = build_obb_config(MODEL_PATH, block_size=OBB_BLOCK_SIZE)
    base = AutoModel.from_pretrained(
        MODEL_PATH, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    assert_block_size(base, OBB_BLOCK_SIZE)
    model = PeftModel.from_pretrained(base, checkpoint).to("cuda:0")
    model.eval()
    return base, model


def evaluate(model, dataset, collator, codec, args, box_open, mask_id,
             coord_start, permutation):
    dtype = next(model.parameters()).dtype
    limit = (len(dataset) if args.max_eval == 0
             else min(args.max_eval, len(dataset)))

    state = {name: {"ious": [], "angles": [], "rows": [], "decoded": 0,
                    "image_ids": []}
             for name in CONDITIONS}
    records = []

    for index in range(limit):
        raw = dataset[index]
        target = dataset.samples[index]["obb"]
        image_size = raw["image"].size          # target geometry lives in this frame
        other = dataset[permutation[index]]
        sample = dataset.samples[index]
        other_sample = dataset.samples[permutation[index]]

        variants = intervention_variants(raw, other)

        record = {
            "index": index,
            "record_id": sample["record_id"],
            "image_id": sample["image_id"],
            "image_path": sample["img_path"],
            "shuffle_index": permutation[index],
            "shuffle_image_id": other_sample["image_id"],
            "shuffle_record_id": other_sample["record_id"],
            "conditions": {},
        }
        for name in CONDITIONS:
            batch = move(collator([variants[name]]), dtype)
            prompt_end = int(batch["prompt_end_indices"][0].item())
            inputs, box_idx = build_block_inputs(batch, prompt_end, box_open, mask_id)
            with torch.no_grad():
                logits = model(**inputs).logits[0]

            bins = read_bins(logits, box_idx, coord_start)
            bucket = state[name]
            bucket["image_ids"].append(sample["image_id"])
            prediction = None
            if bins is not None and len(bins) == OBB_N_COORDS:
                try:
                    # decode in the TARGET frame so IoU stays comparable across
                    # conditions even when the shuffled image has other dimensions
                    prediction = codec.decode(tuple(bins), image_size=image_size)
                    bucket["decoded"] += 1
                    bucket["rows"].append(bins)
                except Exception:
                    prediction = None

            if prediction is None:
                bucket["ious"].append(0.0)
                angle = None
            else:
                bucket["ious"].append(rotated_iou(prediction, target))
                angle = angle_error_deg(prediction.theta_deg, target.theta_deg)
                bucket["angles"].append(angle)
            record["conditions"][name] = {
                "decoded": prediction is not None,
                "rotated_iou": bucket["ious"][-1],
                "angle_error_deg": angle,
                "predicted_bins": bins,
            }

        records.append(record)

        if args.progress and (index + 1) % args.progress == 0:
            running = "  ".join(
                f"{name}={sum(state[name]['ious']) / (index + 1):.4f}"
                for name in CONDITIONS)
            print(f"      {index + 1}/{limit}  {running}", flush=True)

    results = {
        name: summarise(
            bucket["ious"], bucket["angles"], bucket["rows"],
            bucket["decoded"], limit, image_ids=bucket["image_ids"],
            n_boot=args.n_boot, bootstrap_seed=args.bootstrap_seed)
        for name, bucket in state.items()
    }
    return results, records


def render(results: dict) -> str:
    header = (f"{'condition':<28}{'mIoU':>8}{'ci95_lo':>9}{'ci95_hi':>9}{'med':>8}"
              f"{'A@.25':>8}{'A@.5':>8}{'A@.75':>8}{'angMAE':>8}{'dec%':>8}")
    lines = ["=" * 100,
             "  Image / prompt dependence of the OBB block (failures scored 0)",
             "=" * 100, header, "-" * 100]
    for name, metric in results.items():
        low, high = metric.get("mIoU_ci95", [float("nan"), float("nan")])
        lines.append(
            f"{name:<28}{metric['mIoU']:>8.4f}{low:>9.4f}{high:>9.4f}"
            f"{metric['median_IoU']:>8.4f}{100 * metric['acc25']:>7.1f}%"
            f"{100 * metric['acc50']:>7.1f}%{100 * metric['acc75']:>7.1f}%"
            f"{metric['angle_mae']:>8.2f}"
            f"{metric['decoded_pct']:>7.1f}%"
        )
    lines.append("-" * 100)
    lines.append("  predicted bin mean / std across samples "
                 "[cx, cy, w, h, sin2t, cos2t]")
    for name, metric in results.items():
        mean = ", ".join(f"{v:7.1f}" for v in metric.get("bin_mean", []))
        std = ", ".join(f"{v:7.1f}" for v in metric.get("bin_std", []))
        lines.append(f"    {name:<30} mean [{mean}]")
        lines.append(f"    {'':<30}  std [{std}]")
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="/workspace/experiments/obb_pbd_checkpoints/checkpoint_epoch_5")
    parser.add_argument("--name", default="OBB_checkpoint_epoch_5")
    parser.add_argument("--max-eval", type=int, default=200)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--subset", default="avvg",
                        help="RefGeo subset; avvg is the only oriented one whose "
                             "images ship with RefGeo")
    parser.add_argument("--prior-samples", type=int, default=3000,
                        help="train samples used to build the mean_box prior")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--expected-samples", type=int, default=0)
    parser.add_argument("--expected-images", type=int, default=0)
    parser.add_argument("--skip-mean-box", action="store_true")
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument("--out", default=f"{EXPERIMENTS}/OBB_FULL_TEST_IMAGE_DEPENDENCE")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    box_open = tokenizer.convert_tokens_to_ids("<box>")
    coord_start = tokenizer.convert_tokens_to_ids("<0>")
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        mask_id = DEFAULT_MASK_TOKEN_ID

    dataset = RefGeoOBBDataset(data_dir=args.data_dir, split=args.split,
                               subset=args.subset, verbose=False)
    collator = PBDMultiModalCollator(processor=processor, ignore_idx=-100)
    codec = OBBTokenCodec(bins=1000)

    limit = (len(dataset) if args.max_eval == 0
             else min(args.max_eval, len(dataset)))
    if limit < 2:
        print("  need at least 2 samples to shuffle")
        return 1
    if args.expected_samples and limit != args.expected_samples:
        raise SystemExit(
            f"expected {args.expected_samples} samples, found {limit}")
    n_images = len({sample["image_id"] for sample in dataset.samples[:limit]})
    if args.expected_images and n_images != args.expected_images:
        raise SystemExit(f"expected {args.expected_images} images, found {n_images}")

    # derangement: no sample keeps its own image / prompt
    permutation = seeded_derangement(limit, seed=args.seed)

    hr("Image / prompt dependence probe")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  split={args.split}  samples={limit}  block_size={OBB_BLOCK_SIZE}  "
          f"<box>={box_open}  mask={mask_id}  <0>={coord_start}", flush=True)

    if not Path(args.checkpoint).exists():
        print(f"  ABORT: checkpoint missing at {args.checkpoint}")
        return 1

    prior = None if args.skip_mean_box else mean_box_baseline(args, codec, dataset)

    base, model = load_model(args.checkpoint)
    try:
        results, records = evaluate(model, dataset, collator, codec, args,
                                    box_open, mask_id, coord_start, permutation)
    finally:
        del model, base
        torch.cuda.empty_cache()

    if prior is not None:
        results["mean_box (no model)"] = prior

    report = render(results)
    print("\n" + report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(report + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"config": vars(args), "n_images": n_images,
                    "results": results}, indent=2) + "\n",
        encoding="utf-8")
    predictions_path = out.with_suffix(".predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"\n  wrote {out.with_suffix('.txt')}, {out.with_suffix('.json')}, "
          f"and {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
