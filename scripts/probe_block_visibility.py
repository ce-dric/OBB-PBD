#!/usr/bin/env python3
"""
Step 1d - training-time block visibility probe
----------------------------------------------
``modeling_qwen2.py`` builds the training attention mask as::

    x0_len = find_prefix_seq_length_by_pe(position_ids)
    create_block_diff_mask_by_pe_4d(block_size, x0_len_list=x0_len, position_ids=..., causal_attn=...)

``find_prefix_seq_length_by_pe`` returns the index where the position ids *drop*,
and ``-1`` when they never drop. ``create_block_diff_mask_by_pe_4d`` then treats
``q_idx < x0_len`` as the causal prefix and everything else as parallel-block
tokens that may attend the prefix.

If x0_len comes back as -1 during training, the prefix region is empty and the
coordinate slots can only attend inside their own 8-token window: no image, no
prompt. Teacher forcing hid this because the answer was inside that window.

This probe intercepts both functions on a real training-mode forward and reports
x0_len plus, for the cx slot, how many keys are visible and whether any of them
belong to the prompt/image prefix.

Read-only. 1 GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModel, AutoProcessor

from obb_pbd.collators import PBDMultiModalCollator
from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.model_config import OBB_BLOCK_SIZE, build_obb_config

MODEL_PATH = "/workspace/models/LocateAnything-3B"

captured: dict = {}


def install_probes():
    """Wrap the two mask helpers wherever transformers loaded the remote code."""
    targets = [m for name, m in list(sys.modules.items())
               if m is not None
               and hasattr(m, "create_block_diff_mask_by_pe_4d")
               and hasattr(m, "find_prefix_seq_length_by_pe")]
    if not targets:
        raise RuntimeError("mask helpers not found in sys.modules")

    for module in targets:
        original_prefix = module.find_prefix_seq_length_by_pe
        original_mask = module.create_block_diff_mask_by_pe_4d

        def probed_prefix(pe, _orig=original_prefix):
            out = _orig(pe)
            captured.setdefault("x0_len", []).append(out.tolist())
            captured.setdefault("pe_shape", pe.shape)
            captured.setdefault("pe_head", pe[0, :6].tolist())
            captured.setdefault("pe_tail", pe[0, -10:].tolist())
            drops = (pe[0, 1:] < pe[0, :-1]).nonzero().flatten().tolist()
            captured.setdefault("pe_drop_positions", drops[:10])
            return out

        def probed_mask(block_size, x0_len_list, position_ids, causal_attn=False, _orig=original_mask):
            float_mask, bool_mask = _orig(block_size, x0_len_list, position_ids, causal_attn)
            captured.setdefault("block_size", block_size)
            captured.setdefault("causal_attn", causal_attn)
            captured.setdefault("x0_len_used", x0_len_list.tolist())
            captured.setdefault("bool_mask", bool_mask)
            return float_mask, bool_mask

        module.find_prefix_seq_length_by_pe = probed_prefix
        module.create_block_diff_mask_by_pe_4d = probed_mask
    return len(targets)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    config = build_obb_config(MODEL_PATH, block_size=OBB_BLOCK_SIZE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    print(f"  probes installed on {install_probes()} module(s)", flush=True)

    tokenizer = processor.tokenizer
    box_open = tokenizer.convert_tokens_to_ids("<box>")

    dataset = RefGeoOBBDataset(data_dir=args.data_dir, split=args.split, verbose=False)
    collator = PBDMultiModalCollator(processor=processor, ignore_idx=-100)
    batch = collator([dataset[0]])
    dtype = next(model.parameters()).dtype
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = (value.to(dtype=dtype, device="cuda:0")
                          if torch.is_floating_point(value)
                          else value.to(device="cuda:0"))

    ids = batch["input_ids"][0].tolist()
    box_idx = ids.index(box_open)
    seq_len = len(ids)
    prompt_end = int(batch["prompt_end_indices"][0].item())

    for key in ("labels", "prompt_end_indices", "sin_pos_indices", "cos_pos_indices",
                "cx_pos_indices", "cy_pos_indices", "w_pos_indices", "h_pos_indices"):
        batch.pop(key, None)

    model.train()
    with torch.no_grad():
        model(**batch)

    print("\n" + "=" * 78)
    print("  TRAINING-MODE BLOCK MASK")
    print("=" * 78)
    print(f"  sequence length        : {seq_len}")
    print(f"  prompt ends at         : {prompt_end}")
    print(f"  <box> at               : {box_idx}  (cx slot = {box_idx + 1})")
    print(f"  block_size             : {captured.get('block_size')}")
    print(f"  causal_attn            : {captured.get('causal_attn')}")
    print(f"  position_ids shape     : {captured.get('pe_shape')}")
    print(f"  position_ids head      : {captured.get('pe_head')}")
    print(f"  position_ids tail      : {captured.get('pe_tail')}")
    print(f"  position_id drops at   : {captured.get('pe_drop_positions')}")
    print(f"  x0_len (prefix length) : {captured.get('x0_len', [['n/a']])[0]}")
    print(f"  x0_len passed to mask  : {captured.get('x0_len_used')}")

    mask = captured.get("bool_mask")
    if mask is None:
        print("  !! mask was not captured")
        return 1
    mask = mask[0, 0].to("cpu")
    cx = box_idx + 1
    row = mask[cx]
    visible = int(row.sum().item())
    visible_idx = row.nonzero().flatten().tolist()
    print(f"\n  cx slot (row {cx}) sees {visible} of {seq_len} keys")
    print(f"  visible key indices    : {visible_idx[:16]}{' ...' if len(visible_idx) > 16 else ''}")
    print(f"  sees any prompt/image  : {any(i < prompt_end for i in visible_idx)}")
    print(f"  sees own position      : {bool(row[cx].item())}")
    print(f"  sees sibling slots     : "
          f"{[i for i in visible_idx if box_idx <= i <= box_idx + 7]}")
    print("\n  A coordinate slot that cannot see the prefix cannot be conditioned on the")
    print("  image, and the only learnable signal left is the coordinate prior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
