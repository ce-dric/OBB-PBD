#!/usr/bin/env python3
"""
OBB-PBD training
================

Fine-tunes LocateAnything-3B on AVVG with the six-slot oriented block
``<box> cx cy w h sin2θ cos2θ </box>`` (Section 3.4 and Table 1 of the paper).
All runs in the paper are launched through ``scripts/run_obb_pbd.sh`` and
``scripts/run_hbb_controls.sh``, which fix the split, seeds and flags.

Trainable parameters
--------------------
* LoRA on the language model projections (``LLM_TARGETS``).
* LoRA on the last ``--vision-blocks`` MoonViT blocks (``wqkv``, ``wo``, ``fc1``,
  ``fc2``). The upstream helper ``wrap_backbone_lora`` targets
  ``self_attn.q_proj`` etc., which do not exist in MoonViT, so it would silently
  adapt nothing; the targets are listed explicitly here instead.
* The multimodal projector ``mlp1``, fully trainable via ``modules_to_save`` so
  it is written into the adapter. ``forward()`` does not wrap the vision tower
  in ``no_grad`` (only ``generate()`` does), so gradients reach it.

``report_trainable`` checks that the vision and projector parameters are
actually trainable before any GPU time is spent.

Losses
------
Cross-entropy on the coordinate tokens plus three auxiliary terms
(``obb_pbd.losses.OBBLossWrapper``): a continuous spatial refinement loss on the
center and size (``--lambda-center``), a circle term that pushes (sin2θ, cos2θ)
onto the unit circle (``--lambda-circle``; two independent softmaxes cannot
learn sin² + cos² = 1 on their own) and a direction term with a near-square
gate (``--lambda-direction``). The defaults 2.5, 0.25 and 0.5 are the paper's
λ_spatial, λ_circle and λ_direction.

Target modes (matched controls, Section 4.2)
--------------------------------------------
``--target-mode`` selects what the same recipe is trained on:

    hbb_corner    original four-slot block <box> x1 y1 x2 y2 </box> (HBB-PBD);
                  requires --lambda-circle 0 --lambda-direction 0
    hbb_envelope  six-slot block on each box's axis-aligned envelope, so the
                  angle is always 0 or 90 degrees (HBB-envelope)
    obb           six-slot block on the oriented box (OBB-PBD, default)

The arms differ only in ``RefGeoOBBDataset(target_mode=...)`` and the block
size, so hbb_corner -> hbb_envelope -> obb separates the block-format step from
the orientation step.

Usage:
    torchrun --nproc_per_node=3 scripts/train.py --epochs 5
    torchrun --nproc_per_node=3 scripts/train.py --max-steps 12  # smoke
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

import peft.utils.save_and_load as peft_sl

# transformers/peft compatibility shim: _maybe_shard_state_dict_for_tp imports
# EmbeddingParallel, which this transformers build does not export. It fires
# whenever a process group exists. We hold one full replica per rank, so
# sharding is a no-op and the function mutates in place anyway.
peft_sl._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None

from obb_pbd.collators import PBDMultiModalCollator
from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.losses import OBBLossWrapper
from obb_pbd.model_config import (
    HBB_BLOCK_SIZE,
    OBB_BLOCK_SIZE,
    assert_block_size,
    build_obb_config,
)
from obb_pbd.splits import (
    resolve_training_partition,
)
from obb_pbd.training_history import write_training_history

MODEL_PATH = "/workspace/models/LocateAnything-3B"
SAVE_DIR = "/workspace/runs/obb_pbd_checkpoints"

#: language model attention + MLP
LLM_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]

#: MoonViT block projections, as actually named in model.safetensors.index.json:
#: vision_model.encoder.blocks.N.{wqkv,wo} and ...blocks.N.mlp.{fc0,fc1}.
#: Note fc0/fc1, not fc1/fc2 -- targeting "mlp.fc2" matches nothing and PEFT does
#: not complain as long as some other target hits, which is why
#: assert_vision_coverage() below counts the adapted modules.
VISION_BLOCK_LEAVES = ["wqkv", "wo", "mlp.fc0", "mlp.fc1"]

#: the multimodal projector, nn.Sequential(LayerNorm, Linear, GELU, Linear)
PROJECTOR_MODULE = "mlp1"

POS_KEYS = (
    "sin_pos_indices",
    "cos_pos_indices",
    "cx_pos_indices",
    "cy_pos_indices",
    "w_pos_indices",
    "h_pos_indices",
)


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(local)
        return rank, local, world, True
    return 0, 0, 1, False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset", default="avvg",
                        help="RefGeo subset. avvg is the only one that is both "
                             "genuinely oriented (97.85%) and ships its own "
                             "imagery; dior_rsvg is 100%% axis-aligned and "
                             "geochat's images are a separate download")
    parser.add_argument("--target-mode", default="obb",
                        choices=("obb", "hbb_envelope", "hbb_corner"),
                        help="obb (default) trains on the polygon's OBB."
                             " hbb_envelope trains the same recipe on the "
                             "axis-aligned envelope instead (theta forced to 0/90) as "
                             "a fine-tuned-HBB control. hbb_corner trains the same "
                             "envelope but tokenized as the original 4-slot upstream "
                             "x1/y1/x2/y2 block (block_size=6) instead of the 6-slot "
                             "block the other two modes use -- pass --lambda-circle 0 "
                             "--lambda-direction 0 with it; see the module docstring")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--projector-lr-scale", type=float, default=0.1,
                        help="mlp1 is trained fully, not through LoRA, so it gets "
                             "a smaller step than the adapters")
    parser.add_argument("--lambda-center", type=float, default=2.5)
    parser.add_argument("--lambda-circle", type=float, default=0.25)
    parser.add_argument("--lambda-direction", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--vision-mode", default="both",
                        choices=("none", "lora", "projector", "both"),
                        help="none freezes the vision path; lora adapts MoonViT blocks only; "
                             "projector trains mlp1 only; both (default) trains both")
    parser.add_argument("--vision-blocks", type=int, default=6,
                        help="adapt only the last N MoonViT blocks. Blocks below the "
                             "cut have no trainable parameter, so their activations "
                             "are never retained -- this is the main VRAM control.")
    parser.add_argument("--freeze-vision", action="store_true",
                        help="alias for --vision-mode none")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stop after N steps per epoch; 0 = full epoch")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--split-manifest", default="",
                        help="Image-disjoint train-derived split manifest. When set, "
                             "only --train-partition indices enter the optimizer.")
    parser.add_argument("--train-partition", default="fit", choices=("fit",),
                        help="Training is deliberately restricted to the fit partition.")
    parser.add_argument("--partition-by-image-ids", action="store_true",
                        help="Derive manifest partitions by canonical image-ID membership "
                             "after target geometry or record filtering.")
    parser.add_argument("--seed", type=int, default=20260812,
                        help="Recorded Python/PyTorch seed.")
    parser.add_argument("--direction-gate", default="normalized",
                        choices=("normalized", "physical"),
                        help="Near-square gate of the direction loss: 'normalized' (the "
                             "paper's recipe) uses (w/W)/(h/H); 'physical' uses the "
                             "box's physical aspect ratio w/h.")
    parser.add_argument("--sampler-seed", type=int, default=0,
                        help="DistributedSampler shuffle seed. Seed 1 in the paper used "
                             "the PyTorch default (0); seeds 2-3 set this to vary "
                             "data order as well as initialization.")
    return parser.parse_args()


def count_vision_blocks(model) -> int:
    """How many MoonViT encoder blocks this checkpoint has."""
    highest = -1
    for name, _ in model.named_modules():
        match = re.search(r"vision_model\.encoder\.blocks\.(\d+)$", name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def build_vision_targets(model, last_n: int, is_main: bool) -> list[str]:
    """Suffix targets for the last ``last_n`` MoonViT blocks.

    PEFT matches a target either exactly or as a dotted suffix, so
    ``blocks.21.wqkv`` selects ``vision_model.encoder.blocks.21.wqkv`` and nothing
    in the language model. Restricting the depth matters for memory as much as for
    optimisation: a block with no trainable parameter takes an input that does not
    require grad, so autograd never retains its intermediates.
    """
    total = count_vision_blocks(model)
    if total == 0:
        raise RuntimeError("no vision_model.encoder.blocks.* found in this model")
    first = max(0, total - last_n) if last_n > 0 else 0
    targets = []
    for index in range(first, total):
        targets += [f"blocks.{index}.{leaf}" for leaf in VISION_BLOCK_LEAVES]
    if is_main:
        print(f"  vision blocks     : {total} total, adapting {first}..{total - 1}",
              flush=True)
    return targets


def assert_vision_coverage(model, expected: int, is_main: bool):
    """Every requested vision module must actually carry a LoRA layer.

    PEFT silently accepts a target name that matches nothing as long as at least
    one other target matches, so a typo costs capacity without any error. The
    first version of this script asked for ``mlp.fc2``, which does not exist
    (MoonViT uses fc0/fc1), and quietly adapted 3 modules per block instead of 4.
    """
    adapted = {
        name.rsplit(".lora_A", 1)[0]
        for name, _ in model.named_modules()
        if "vision_model" in name and name.endswith(".lora_A")
    }
    if is_main:
        print(f"  vision LoRA layers: {len(adapted)} (expected {expected})",
              flush=True)
    if len(adapted) != expected:
        raise RuntimeError(
            f"expected {expected} adapted vision modules, found {len(adapted)}. "
            "A target name in VISION_BLOCK_LEAVES does not exist in this "
            "checkpoint; check model.safetensors.index.json for the real names."
        )


def section_of(name: str) -> str:
    if "vision_model" in name:
        return "vision_model"
    if PROJECTOR_MODULE in name:
        return "mlp1 (projector)"
    if "language_model" in name:
        return "language_model"
    return "other"


def report_trainable(model, is_main: bool):
    """Group trainable parameters by backbone section.

    vision_model and mlp1 must appear here with non-zero counts. If they do not,
    the vision path is frozen and there is no reason to spend the GPU hours.
    """
    counts, totals = {}, {}
    for name, param in model.named_parameters():
        key = section_of(name)
        totals[key] = totals.get(key, 0) + param.numel()
        if param.requires_grad:
            counts[key] = counts.get(key, 0) + param.numel()
    if is_main:
        print("  trainable parameters by section:", flush=True)
        for key in sorted(totals):
            trainable = counts.get(key, 0)
            share = 100.0 * trainable / totals[key] if totals[key] else 0.0
            print(f"    {key:<20} {trainable:>12,} / {totals[key]:>13,}  "
                  f"({share:5.2f}%)", flush=True)
    return counts


def report_grad_norms(model, is_main: bool, tag: str):
    """Per-section gradient norms, printed once after the first backward.

    An open attention mask is not proof that a tensor is used, and a trainable
    flag is not proof that a gradient arrives. This is the check that the vision
    tower is actually in the graph.
    """
    sums = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = section_of(name)
        sums[key] = sums.get(key, 0.0) + float(param.grad.detach().float().pow(2).sum())
    if is_main:
        print(f"  [{tag}] gradient L2 by section:", flush=True)
        for key in sorted(sums):
            print(f"    {key:<20} {sums[key] ** 0.5:.6f}", flush=True)
        missing = [k for k in ("vision_model", "mlp1 (projector)") if k not in sums]
        if missing:
            print(f"    WARNING: no gradient reached {missing}", flush=True)


def main():
    args = parse_args()
    rank, local_rank, world_size, is_ddp = setup_ddp()
    is_main = rank == 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        print("=" * 78, flush=True)
        print("  OBB-PBD training", flush=True)
        print("=" * 78, flush=True)
        print(f"  world_size        : {world_size}", flush=True)
        print(f"  lambda_center     : {args.lambda_center}", flush=True)
        print(f"  lambda_circle     : {args.lambda_circle}", flush=True)
        print(f"  lambda_direction  : {args.lambda_direction}", flush=True)
        print(f"  learning rate     : {args.lr} "
              f"(projector x{args.projector_lr_scale})", flush=True)
        print(f"  lora r            : {args.lora_r}", flush=True)
        vision_desc = "none" if args.freeze_vision else args.vision_mode
        print(f"  vision adaptation : {vision_desc}", flush=True)
        print(f"  target mode       : {args.target_mode}", flush=True)
        print(f"  epochs            : {args.epochs}  grad_accum {args.grad_accum}",
              flush=True)
        print(f"  seed              : {args.seed}", flush=True)
        print(f"  sampler seed      : {args.sampler_seed}", flush=True)
        print(f"  direction gate    : {args.direction_gate}", flush=True)
        if args.max_steps:
            print(f"  SMOKE TEST        : {args.max_steps} steps/epoch", flush=True)

    # hbb_corner is the only mode that trains the original 4-slot upstream block
    # (block_size=6); obb and hbb_envelope both use the 6-slot/8-token OBB block.
    block_size = HBB_BLOCK_SIZE if args.target_mode == "hbb_corner" else OBB_BLOCK_SIZE
    n_geometry_slots = 4 if args.target_mode == "hbb_corner" else 6
    if is_main:
        print(f"  block size        : {block_size} ({n_geometry_slots} geometry slots)",
              flush=True)
        if args.target_mode == "hbb_corner" and (args.lambda_circle > 0 or args.lambda_direction > 0):
            print(f"  WARNING: --target-mode hbb_corner has no orientation slot; "
                  f"lambda_circle={args.lambda_circle} and lambda_direction="
                  f"{args.lambda_direction} will be silently skipped by OBBLossWrapper "
                  f"(sin/cos_pos_indices are None for this mode) rather than applied. "
                  f"Pass --lambda-circle 0 --lambda-direction 0 to make that explicit.",
                  flush=True)

    config = build_obb_config(MODEL_PATH, block_size=block_size)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    base_model = AutoModel.from_pretrained(
        MODEL_PATH, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    observed = assert_block_size(base_model, block_size)
    if is_main:
        print(f"  [block] language_model.model.block_size = {observed} (verified)",
              flush=True)

    mode = "none" if args.freeze_vision else args.vision_mode
    adapt_vision_blocks = mode in ("lora", "both")
    train_projector = mode in ("projector", "both")

    targets = list(LLM_TARGETS)
    vision_targets = []
    if adapt_vision_blocks:
        vision_targets = build_vision_targets(base_model, args.vision_blocks, is_main)
        targets += vision_targets
    modules_to_save = [PROJECTOR_MODULE] if train_projector else None

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        target_modules=targets,
        modules_to_save=modules_to_save,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(local_rank)

    trainable_by_section = report_trainable(model, is_main)
    if vision_targets:
        assert_vision_coverage(model, len(vision_targets), is_main)
    required = []
    if adapt_vision_blocks:
        required.append("vision_model")
    if train_projector:
        required.append("mlp1 (projector)")
    for section in required:
        if trainable_by_section.get(section, 0) == 0:
            raise RuntimeError(
                f"no trainable parameters in {section}: the LoRA targets or "
                f"modules_to_save={modules_to_save} did not match this checkpoint. "
                "Aborting before spending GPU hours."
            )

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    assert_block_size(model, block_size)

    dataset = RefGeoOBBDataset(data_dir="/workspace/data", split="train",
                               subset=args.subset, verbose=is_main,
                               target_mode=args.target_mode)
    source_dataset_records = len(dataset)
    partition_indices, partition_provenance = resolve_training_partition(
        dataset, args.split_manifest or None, args.train_partition,
        args.partition_by_image_ids,
    )
    split_manifest_sha256 = partition_provenance["split_manifest_sha256"] or ""
    if args.split_manifest:
        dataset = Subset(dataset, partition_indices)
        if is_main:
            print(f"  partition=fit     : {len(dataset):,} records from "
                  f"{args.split_manifest}", flush=True)
            print(f"  manifest sha256   : {split_manifest_sha256}", flush=True)
            if args.partition_by_image_ids:
                print("  partition source  : canonical image IDs", flush=True)
    if args.limit_samples > 0:
        dataset = Subset(dataset, list(range(min(args.limit_samples, len(dataset)))))
        if is_main:
            print(f"  [overfit] restricted to {len(dataset)} samples", flush=True)
    collator = PBDMultiModalCollator(processor=processor, ignore_idx=-100,
                                     emit_image_aspect=args.direction_gate == "physical",
                                     n_geometry_slots=n_geometry_slots)
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, seed=args.sampler_seed)
               if is_ddp else None)
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, shuffle=(sampler is None),
        collate_fn=collator, num_workers=2, pin_memory=True,
    )

    coord_start = processor.tokenizer.convert_tokens_to_ids("<0>")
    if coord_start is None or coord_start == processor.tokenizer.unk_token_id:
        raise RuntimeError("tokenizer does not know the <0> coordinate token")
    if is_main:
        print(f"  coord vocab start : {coord_start}", flush=True)
        print(f"  dataset           : {len(dataset):,} samples, "
              f"{len(dataloader):,} steps/rank/epoch", flush=True)

    # mlp1 is trained in full rather than through an adapter, so it takes a
    # smaller step than the LoRA matrices.
    projector_params, adapter_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (projector_params if PROJECTOR_MODULE in name else adapter_params).append(param)
    groups = [{"params": adapter_params, "lr": args.lr}]
    if projector_params:
        groups.append({"params": projector_params,
                       "lr": args.lr * args.projector_lr_scale})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)

    loss_wrapper = OBBLossWrapper(
        lambda_circle=args.lambda_circle,
        lambda_center=args.lambda_center,
        lambda_direction=args.lambda_direction,
        coord_vocab_start_idx=coord_start,
    )

    history = []
    inner = model.module if is_ddp else model
    grads_reported = False

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        running = 0.0
        counted = 0
        failures = 0
        epoch_started = time.time()

        for step, batch in enumerate(dataloader):
            if args.max_steps and step >= args.max_steps:
                break

            pos = {key: batch.pop(key, None) for key in POS_KEYS}
            image_aspect = batch.pop("image_aspect", None)
            dtype = next(model.parameters()).dtype
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = (value.to(dtype=dtype, device=local_rank)
                                  if torch.is_floating_point(value)
                                  else value.to(device=local_rank))

            try:
                outputs = model(**batch)
                loss, metrics = loss_wrapper(
                    outputs,
                    labels=batch.get("labels"),
                    sin_pos_indices=pos["sin_pos_indices"],
                    cos_pos_indices=pos["cos_pos_indices"],
                    cx_pos_indices=pos["cx_pos_indices"],
                    cy_pos_indices=pos["cy_pos_indices"],
                    w_pos_indices=pos["w_pos_indices"],
                    h_pos_indices=pos["h_pos_indices"],
                    image_aspect=image_aspect,
                )
                (loss / args.grad_accum).backward()
            except Exception as exc:
                failures += 1
                optimizer.zero_grad()
                if is_main and failures <= 5:
                    print(f"    step {step} failed: {type(exc).__name__}: {exc}",
                          flush=True)
                continue

            if not grads_reported:
                report_grad_norms(model, is_main, "step 1")
                if is_main:
                    peak = torch.cuda.max_memory_allocated(local_rank) / 1024 ** 3
                    print(f"  [vram] peak after first backward: {peak:.2f} GiB",
                          flush=True)
                grads_reported = True

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running += float(loss.item())
            counted += 1

            if is_main and (step == 0 or (step + 1) % args.log_every == 0):
                print(
                    f"  ep{epoch} step {step + 1}/{len(dataloader)} "
                    f"loss {loss.item():.4f} "
                    f"(ce {metrics['loss_ce']:.4f} "
                    f"centre {metrics.get('loss_center', 0.0):.4f} "
                    f"dir {metrics.get('loss_direction', 0.0):.4f} "
                    f"circle {metrics.get('loss_circle', 0.0):.4f})",
                    flush=True,
                )

        avg = running / max(1, counted)
        if is_main:
            mins = (time.time() - epoch_started) / 60.0
            peak = torch.cuda.max_memory_allocated(local_rank) / 1024 ** 3
            print(f"  epoch {epoch} done | avg loss {avg:.4f} | {counted} steps | "
                  f"{failures} failures | {mins:.1f} min | peak {peak:.2f} GiB",
                  flush=True)

            if epoch % args.save_every == 0 or epoch == args.epochs:
                ckpt = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}")
                inner.save_pretrained(ckpt)
                print(f"  saved {ckpt}", flush=True)

            history.append({"epoch": epoch, "avg_loss": avg, "steps": counted,
                            "failures": failures, "minutes": mins,
                            "peak_gib": peak})
            write_training_history(
                args.save_dir,
                {
                    "lambda_center": args.lambda_center,
                    "lambda_circle": args.lambda_circle,
                    "lambda_direction": args.lambda_direction,
                    "lr": args.lr,
                    "projector_lr_scale": args.projector_lr_scale,
                    "lora_r": args.lora_r,
                    "vision_mode": args.vision_mode,
                    "vision_blocks": args.vision_blocks,
                    "grad_accum": args.grad_accum,
                    "vision_adapted": not args.freeze_vision,
                    "lora_targets": targets,
                    "modules_to_save": modules_to_save,
                    "block_size": block_size,
                    "n_geometry_slots": n_geometry_slots,
                    "coord_vocab_start_idx": coord_start,
                    "target_mode": args.target_mode,
                    "seed": args.seed,
                    "sampler_seed": args.sampler_seed,
                    "direction_gate": args.direction_gate,
                    "source_dataset_records": source_dataset_records,
                    "training_records": len(dataset),
                    "history": history,
                },
                partition_provenance,
            )

        if is_ddp:
            dist.barrier()

    if is_main:
        print("Training complete.", flush=True)
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
