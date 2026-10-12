#!/usr/bin/env python3
"""One-off diagnostic (2026-08-08): dump argmax tokens around the <box> block for a
memorized hbb_corner checkpoint, to distinguish a read-offset bug (eval reads the
wrong positions) from a training-side bug (the model never learned the right
positions), per the advisor's suggested cheap discriminator. Not part of the
regular pipeline -- kept in scripts/ for reproducibility, not wired into any
run_*.sh launcher.
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

from obb_pbd.collators import PBDMultiModalCollator
from obb_pbd.dataset import RefGeoOBBDataset
from obb_pbd.model_config import HBB_BLOCK_SIZE, assert_block_size, build_obb_config
from obb_pbd.tokens import OBBTokenCodec
from eval_pbd_direct import build_block_inputs, move

MODEL_PATH = "/workspace/models/LocateAnything-3B"
CHECKPOINT = "/workspace/experiments/hbb_corner_memtest/checkpoint_epoch_20"
SAMPLE_INDEX = 0

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer
box_open = tokenizer.convert_tokens_to_ids("<box>")
box_close = tokenizer.convert_tokens_to_ids("</box>")
coord_start = tokenizer.convert_tokens_to_ids("<0>")
mask_id = tokenizer.convert_tokens_to_ids("<mask>")
if mask_id is None or mask_id == tokenizer.unk_token_id:
    from obb_pbd.collators import DEFAULT_MASK_TOKEN_ID
    mask_id = DEFAULT_MASK_TOKEN_ID
print(f"<box>={box_open}  </box>={box_close}  <0>={coord_start}  mask={mask_id}")

dataset = RefGeoOBBDataset(data_dir="/workspace/data", split="train", subset="avvg",
                           verbose=False, target_mode="hbb_corner")
raw = dataset[SAMPLE_INDEX]
target_obb = dataset.samples[SAMPLE_INDEX]["obb"]
image_size = raw["image"].size
codec = OBBTokenCodec(bins=1000)
gt_tokens = codec.encode_corners(target_obb, image_size)
print(f"sample {SAMPLE_INDEX}: target_obb_str = {raw['target_obb_str']!r}")
print(f"ground truth corner bins (x1,y1,x2,y2) = {gt_tokens}")

collator = PBDMultiModalCollator(processor=processor, ignore_idx=-100)
batch = collator([raw])
input_ids = batch["input_ids"][0].tolist()
prompt_end = int(batch["prompt_end_indices"][0].item())
print(f"prompt_end = {prompt_end}, seq_len = {len(input_ids)}")
if box_open in input_ids:
    box_pos_in_full_seq = input_ids.index(box_open)
    print(f"<box> position in the FULL (unmasked, ground-truth-included) "
          f"sequence built by the collator: {box_pos_in_full_seq}")
    window = input_ids[max(0, box_pos_in_full_seq - 2):box_pos_in_full_seq + 8]
    print(f"tokens around <box> in that full sequence: {window}")
    bins = [(t - coord_start) if coord_start <= t < coord_start + 1001 else None
            for t in window]
    print(f"  as coordinate bins (None = not a coord token): {bins}")

# Now build the EVAL-STYLE masked input exactly as build_block_inputs() does.
device = "cuda:0"
config = build_obb_config(MODEL_PATH, block_size=HBB_BLOCK_SIZE)
base = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True,
                                 torch_dtype=torch.bfloat16, device_map=device)
assert_block_size(base, HBB_BLOCK_SIZE)
print(f"config.block_size={config.block_size}  "
      f"config.n_future_tokens={getattr(config, 'n_future_tokens', '?')}  "
      f"text_config.block_size={config.text_config.block_size}  "
      f"text_config.n_future_tokens={getattr(config.text_config, 'n_future_tokens', '?')}")
model = PeftModel.from_pretrained(base, CHECKPOINT).to(device)
model.eval()

dtype = next(model.parameters()).dtype
moved = move(batch, dtype, device)
inputs, box_idx = build_block_inputs(moved, prompt_end, box_open, mask_id,
                                     block_size=HBB_BLOCK_SIZE)
print(f"\neval-style input (via build_block_inputs, same function eval_pbd_direct.py "
      f"uses): prompt_end={prompt_end}, box_idx used for logits read={box_idx}, "
      f"input_ids.shape={tuple(inputs['input_ids'].shape)}")
print(f"input_ids around box_idx: "
      f"{inputs['input_ids'][0, max(0,box_idx-2):box_idx+8].tolist()}")
print(f"inputs keys: {sorted(inputs.keys())}")

with torch.no_grad():
    logits = model(**inputs).logits[0]

print(f"\nargmax token id / coordinate bin at each position box_idx-2 .. box_idx+7:")
for pos in range(max(0, box_idx - 2), box_idx + 8):
    if pos >= logits.shape[0]:
        break
    tok = int(logits[pos].argmax(dim=-1).item())
    bin_val = (tok - coord_start) if coord_start <= tok < coord_start + 1001 else None
    tag = ""
    if pos == box_idx:
        tag = "  <-- box_idx (per current eval code, reads box_idx..box_idx+3 for corners)"
    print(f"  pos={pos:4d}  token_id={tok:7d}  bin={bin_val}{tag}")

print(f"\nground truth corner bins for comparison: {gt_tokens}")

# Decode the prediction and score it against BOTH:
#  (a) the envelope ground truth (target_mode="hbb_corner" dataset, what training
#      actually targets) -- this is what the argmax-match above already confirms.
#  (b) the TRUE original rotated OBB (target_mode="obb" default dataset) -- this is
#      what eval_pbd_direct.py actually scores against (see run_hbb_control_eval.sh:
#      "Ground truth is the original OBB regardless of target_mode"). A perfect
#      envelope prediction is NOT expected to score near 1.0 rotIoU here -- the GT
#      AABB-envelope reference row in results.tex caps at rotIoU 0.5050 on average
#      (Table 2), so a single elongated/steeply-rotated sample could legitimately
#      score much lower, even near 0, with a geometrically PERFECT envelope
#      prediction. This block distinguishes "expected geometric mismatch" from an
#      actual remaining bug.
from obb_pbd.metrics import rotated_iou, aabb_iou

pred_bins = tuple(
    int(logits[pos].argmax(dim=-1).item()) - coord_start
    for pos in range(box_idx, box_idx + 4)
)
prediction = codec.decode_corners(pred_bins, image_size=image_size)
print(f"\ndecoded prediction (axis-aligned): {prediction}")

envelope_iou = rotated_iou(prediction, target_obb)
print(f"rotIoU vs envelope ground truth (target_mode=hbb_corner's own target): "
      f"{envelope_iou:.4f}  (expect ~1.0 given the exact bin match above)")

true_dataset = RefGeoOBBDataset(data_dir="/workspace/data", split="train", subset="avvg",
                                verbose=False, require_corner_encodable=True)
# default target_mode="obb": TRUE rotated GT, but filtered to the SAME record
# sequence as the hbb_corner dataset above (the fix for the index-misalignment
# bug this whole script was written to catch).
true_target = true_dataset.samples[SAMPLE_INDEX]["obb"]
print(f"TRUE rotated GT obb for this sample: {true_target}")
true_rot_iou = rotated_iou(prediction, true_target)
true_aabb_iou = aabb_iou(prediction, true_target)
print(f"rotIoU vs TRUE rotated GT (what eval_pbd_direct.py actually scores): "
      f"{true_rot_iou:.4f}")
print(f"aabbIoU vs TRUE rotated GT: {true_aabb_iou:.4f}")

from obb_pbd.metrics import obb_to_aabb
theoretical_best_envelope = obb_to_aabb(true_target)
ceiling_iou = rotated_iou(theoretical_best_envelope, true_target)
print(f"GT AABB-envelope reference ceiling for THIS sample (best any axis-aligned "
      f"box can do against this specific rotated GT): {ceiling_iou:.4f}")
