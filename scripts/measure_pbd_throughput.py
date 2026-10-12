#!/usr/bin/env python3
"""
Throughput of parallel box decoding, measured on both axes that matter
=====================================================================

What it measures
----------------
Two different claims need two different measurements (Section 4.4):

Axis 1, the cost of the extension
    HBB emits four coordinates in a block_size 6 layout; OBB emits six geometry slots in a
    block_size 8 layout. Both are a single forward pass, so the question is whether OBB is
    measurably slower.

Axis 2, the benefit of parallel decoding
    One forward pass against emitting the same six coordinates one token at a time with a
    KV cache, which is the axis LocateAnything's 12.7 BPS claim lives on. Upstream
    ``generate()`` cannot be used for this, and
    ``LocateAnythingForConditionalGeneration.forward()`` takes ``pixel_values`` as a
    required positional argument and feeds it into MoonViT unconditionally, so a cached
    decode step through it cannot skip the vision tower. Passing the image again on every
    step would re-encode it six times and inflate the apparent speedup.

    Instead, the prefill goes through the full model once (MoonViT runs exactly once, as in
    real generation), and every subsequent decode step drives the causal language model
    returned by ``resolve_language_model`` directly with the embedding of the previous
    greedy token and the carried KV cache. This is standard cached autoregressive decoding
    of the same six geometry tokens plus the closing token, i.e. the same number of output
    tokens as the parallel block. ``--only`` and ``--merge`` add a row to an existing
    result file without re-timing the others.

Honest reporting of the absolute numbers matters as much as the ratios. These are measured
at batch size 1, on 1024-pixel imagery, with sdpa attention rather than flash-attn, because
that is the configuration everything else in this repository was measured in. Absolute BPS
is therefore not comparable to a number obtained with batching or flash-attn; the ratios
are the result.

    python scripts/measure_pbd_throughput.py --checkpoint /workspace/experiments/obb_pbd_checkpoints/checkpoint_epoch_5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))

from transformers import AutoModel, AutoProcessor

from eval_pbd_direct import DEFAULT_DEVICE, MODEL_PATH, load_model, move
from obb_pbd.collators import PBDMultiModalCollator
from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.model_config import (
    HBB_BLOCK_SIZE,
    OBB_BLOCK_SIZE,
    OBB_N_COORDS,
    assert_block_size,
    build_obb_config,
)

EXPERIMENTS = "/workspace/experiments"

# Coordinates per block for each layout. HBB is `<box> x1 y1 x2 y2 </box>`, so block_size 6
# carries four; OBB is `<box> cx cy w h sin2t cos2t </box>`, so block_size 8 carries six.
HBB_N_COORDS = HBB_BLOCK_SIZE - 2

# Discarded before timing. The first pass through a freshly loaded model pays for cuDNN
# autotuning and lazy CUDA module loading, which would otherwise be attributed to the layout
# under test.
WARMUP_SAMPLES = 5


def time_single_forward(model, batch, prompt_end, block_size, box_open, mask_id, device):
    """One forward pass over prompt + <box> + (block_size - 1) mask tokens.

    This is parallel box decoding: the whole block is resolved in one step.
    """
    ids = batch["input_ids"][:, :prompt_end]
    trigger = torch.full((ids.shape[0], 1), box_open, dtype=ids.dtype, device=ids.device)
    suffix = torch.full((ids.shape[0], block_size - 1), mask_id, dtype=ids.dtype,
                        device=ids.device)
    new_ids = torch.cat([ids, trigger, suffix], dim=1)

    inputs = {k: v for k, v in batch.items()
              if k not in ("labels", "prompt_end_indices", "position_ids",
                           "sin_pos_indices", "cos_pos_indices", "cx_pos_indices",
                           "cy_pos_indices", "w_pos_indices", "h_pos_indices")}
    inputs["input_ids"] = new_ids
    if isinstance(inputs.get("attention_mask"), torch.Tensor):
        inputs["attention_mask"] = torch.ones_like(new_ids)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        model(**inputs)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def resolve_causal_lm(model):
    """The Qwen2 *ForCausalLM* (with lm_head) inside PEFT/AutoModel wrapping."""
    candidate, seen = model, 0
    while candidate is not None and seen < 8:
        seen += 1
        inner = getattr(candidate, "language_model", None)
        if inner is not None:
            return inner
        nxt = None
        for attr in ("module", "base_model", "model"):
            value = getattr(candidate, attr, None)
            if value is not None and value is not candidate:
                nxt = value
                break
        candidate = nxt
    raise RuntimeError("could not locate language_model inside the loaded model")


def time_sequential_forwards(model, batch, prompt_end, n_coords, box_open, mask_id, device):
    """What the same coordinates cost one token at a time, with a KV cache.

    The comparison has to be like for like: same model, same prompt, same number of
    coordinates produced. Only the number of forward passes changes. The cache is carried
    across steps exactly as autoregressive generation would, so this is not an artificially
    slow strawman that re-encodes the prompt every step.
    """
    ids = batch["input_ids"][:, :prompt_end]
    trigger = torch.full((ids.shape[0], 1), box_open, dtype=ids.dtype, device=ids.device)
    prefix = torch.cat([ids, trigger], dim=1)

    inputs = {k: v for k, v in batch.items()
              if k not in ("labels", "prompt_end_indices", "position_ids",
                           "sin_pos_indices", "cos_pos_indices", "cx_pos_indices",
                           "cy_pos_indices", "w_pos_indices", "h_pos_indices")}
    inputs["input_ids"] = prefix
    if isinstance(inputs.get("attention_mask"), torch.Tensor):
        inputs["attention_mask"] = torch.ones_like(prefix)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    lm = resolve_causal_lm(model)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        cache = out.past_key_values
        if cache is None:
            raise RuntimeError("prefill returned no KV cache; sequential timing would be invalid")
        past_len = prefix.shape[1]
        for _ in range(n_coords):
            # Greedy next token from the previous step, fed to the language model only: the
            # image is already in the cache from the prefill, so re-running MoonViT here would
            # inflate the sequential cost and flatter the parallel speedup. input_ids (not
            # inputs_embeds) keeps the backbone on its inference mask path -- with
            # input_ids=None it builds the training block mask, which ignores the cache --
            # matching upstream generate()'s autoregressive step.
            step_ids = out.logits[:, -1:].argmax(dim=-1)
            attention_mask = torch.ones((prefix.shape[0], past_len + 1), dtype=torch.long,
                                        device=prefix.device)
            position_ids = torch.full((prefix.shape[0], 1), past_len, dtype=torch.long,
                                      device=prefix.device)
            out = lm(input_ids=step_ids, attention_mask=attention_mask,
                     position_ids=position_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            past_len += 1
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def load_layout(checkpoint: str, block_size: int, device: str):
    """Load the model at a given block size, with the adapter only where it applies.

    The trained adapter was fitted at block_size 8. Timing the HBB layout therefore uses
    base weights: the point of axis 1 is the cost of the layout, not the quality of the
    output, and loading a block_size 8 adapter into a block_size 6 model would be wrong.
    """
    if block_size == OBB_BLOCK_SIZE and checkpoint:
        _, model = load_model(checkpoint, device)
        return model
    config = build_obb_config(MODEL_PATH, block_size=block_size)
    model = AutoModel.from_pretrained(
        MODEL_PATH, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=device)
    assert_block_size(model, block_size, strict=False)
    model.eval()
    return model


def summarise(name, latencies, n_coords, note):
    mean = statistics.mean(latencies)
    return {
        "condition": name,
        "coords_per_block": n_coords,
        "samples": len(latencies),
        "latency_ms_mean": mean,
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "boxes_per_sec": 1000.0 / mean if mean else 0.0,
        "note": note,
    }


def render(results):
    lines = ["=" * 104,
             "  Parallel box decoding throughput, batch size 1, 1024 px imagery, sdpa "
             "attention",
             "  Quote the ratios, not the absolute BPS: no batching and no flash-attn.",
             "=" * 104,
             f"{'condition':<34}{'coords':>7}{'ms mean':>10}{'ms med':>9}{'ms p95':>9}"
             f"{'BPS':>8}",
             "-" * 104]
    for metric in results:
        lines.append(
            f"{metric['condition']:<34}{metric['coords_per_block']:>7d}"
            f"{metric['latency_ms_mean']:>10.2f}{metric['latency_ms_median']:>9.2f}"
            f"{metric['latency_ms_p95']:>9.2f}{metric['boxes_per_sec']:>8.2f}")
    lines.append("-" * 104)
    for metric in results:
        lines.append(f"    {metric['condition']:<32} {metric['note']}")

    by_name = {m["condition"]: m for m in results}
    obb = by_name.get("OBB parallel (block_size 8)")
    hbb = by_name.get("HBB parallel (block_size 6)")
    seq = by_name.get("OBB sequential (6 forwards)")
    lines.append("-" * 104)
    if obb and hbb:
        ratio = obb["latency_ms_mean"] / hbb["latency_ms_mean"]
        lines.append(f"  axis 1, cost of the extension: OBB is {ratio:.3f}x the HBB "
                     f"latency while emitting {obb['coords_per_block']} coordinates "
                     f"instead of {hbb['coords_per_block']}")
    if obb and seq:
        speedup = seq["latency_ms_mean"] / obb["latency_ms_mean"]
        lines.append(f"  axis 2, benefit of parallel decoding: one forward is "
                     f"{speedup:.2f}x faster than {OBB_N_COORDS} sequential forwards for "
                     f"the same six coordinates")
    lines.append("=" * 104)
    return "\n".join(lines)


def write_results(out_path, args, results):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(render(results) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"config": vars(args), "results": results}, indent=2) + "\n",
        encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="",
                        help="PEFT checkpoint for the OBB rows. Empty uses base weights, "
                             "which is fine for timing but note it in the paper")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default="avvg")
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--out", default=f"{EXPERIMENTS}/PBD_THROUGHPUT")
    parser.add_argument("--only", default="",
                        help="substring filter on condition names, so a single row can "
                             "be re-measured without paying for the others again")
    parser.add_argument("--merge", default="",
                        help="path to an existing PBD_THROUGHPUT.json whose rows are "
                             "kept for conditions not measured in this run")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    box_open = tokenizer.convert_tokens_to_ids("<box>")
    mask_id = tokenizer.convert_tokens_to_ids("<mask>")
    if mask_id is None or mask_id == tokenizer.unk_token_id:
        from obb_pbd.collators import DEFAULT_MASK_TOKEN_ID
        mask_id = DEFAULT_MASK_TOKEN_ID

    dataset = RefGeoOBBDataset(data_dir=args.data_dir, split=args.split,
                               subset=args.subset, verbose=False)
    collator = PBDMultiModalCollator(processor=processor, ignore_idx=-100)
    limit = min(args.samples + WARMUP_SAMPLES, len(dataset))

    print(f"  timing {limit - WARMUP_SAMPLES} samples after {WARMUP_SAMPLES} warmup, "
          f"device={args.device}", flush=True)

    conditions = [
        ("OBB parallel (block_size 8)", OBB_BLOCK_SIZE, OBB_N_COORDS, "parallel",
         "one forward, six geometry slots. This is what the paper measures as PBD."),
        ("HBB parallel (block_size 6)", HBB_BLOCK_SIZE, HBB_N_COORDS, "parallel",
         "one forward, four coordinates, upstream layout, base weights."),
        ("OBB sequential (6 forwards)", OBB_BLOCK_SIZE, OBB_N_COORDS, "sequential",
         "one full-model prefill (image + prompt + <box>), then one cached language-model "
         "step per output token (six geometry tokens + </box>), greedy, no vision tower."),
    ]

    previous = {}
    if args.merge and Path(args.merge).exists():
        # a crash in one condition should not force re-timing the ones that succeeded
        loaded = json.loads(Path(args.merge).read_text())
        previous = {row["condition"]: row for row in loaded.get("results", [])}
        print(f"  merging {len(previous)} previously measured conditions", flush=True)

    results = []
    for name, block_size, n_coords, mode, note in conditions:
        if args.only and args.only.lower() not in name.lower():
            if name in previous:
                results.append(previous[name])
            continue
        print(f"\n  --- {name} ---", flush=True)
        model = load_layout(args.checkpoint, block_size, args.device)
        dtype = next(model.parameters()).dtype
        latencies = []
        try:
            for index in range(limit):
                batch = move(collator([dataset[index]]), dtype, args.device)
                prompt_end = int(batch["prompt_end_indices"][0].item())
                if mode == "parallel":
                    elapsed = time_single_forward(model, batch, prompt_end, block_size,
                                                  box_open, mask_id, args.device)
                else:
                    elapsed = time_sequential_forwards(model, batch, prompt_end, n_coords,
                                                       box_open, mask_id, args.device)
                if index >= WARMUP_SAMPLES:
                    latencies.append(elapsed)
        finally:
            del model
            torch.cuda.empty_cache()

        metric = summarise(name, latencies, n_coords, note)
        results.append(metric)
        print(f"      {metric['latency_ms_mean']:.2f} ms mean, "
              f"{metric['boxes_per_sec']:.2f} BPS", flush=True)

        # Persist after every condition. The first version of this script lost a completed
        # 100-sample measurement of both parallel rows because the sequential row crashed
        # before anything was written.
        write_results(args.out, args, results)

    print("\n" + render(results))
    out = write_results(args.out, args, results)
    print(f"\n  wrote {out.with_suffix('.txt')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
