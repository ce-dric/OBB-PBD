from __future__ import annotations

import torch
import numpy as np
from typing import Any, Sequence
from PIL import Image


#: ``generate_utils.get_token_ids_from_config`` resolves the block placeholder as
#: ``getattr(text_config, 'text_mask_token_id', 151676)``. Training must place the
#: same token in the parallel block, otherwise the block is filled with ground
#: truth at train time and with mask tokens at inference time.
DEFAULT_MASK_TOKEN_ID = 151676

#: geometry slots between ``<box>`` and ``</box>``: cx, cy, w, h, sin2t, cos2t.
#: This is the default for ``PBDMultiModalCollator(n_geometry_slots=...)``; the
#: matched 4-slot HBB-PBD baseline (``dataset.py``'s ``target_mode="hbb_corner"``)
#: passes ``n_geometry_slots=4`` for the original ``x1 y1 x2 y2`` block instead.
N_GEOMETRY_SLOTS = 6


class PBDMultiModalCollator:
    """
    Data Collator designed specifically for Native 8-Token OBB Parallel Box Decoding in multi-GPU (DDP) training.
    Uses native Qwen2-VL vision processors, dynamic padding, -100 label masking, and sin/cos index localization.

    Block input masking (``mask_block_inputs=True``, default)
    --------------------------------------------------------
    ``create_block_diff_mask_by_pe_4d`` is called with ``causal_attn=False``, so
    the parallel block is **mutually visible**: every slot attends to every other
    slot of the same block, itself included. The training objective is still the
    shifted next-token CE of ``modeling_qwen2.py``, which scores the logits at
    slot ``k-1`` against the ground-truth token of slot ``k``.

    Leaving the ground-truth coordinate tokens inside ``input_ids`` therefore lets
    the model read the very token it is asked to predict, and it learns a copy
    shortcut instead of image-conditioned box regression. Measured on E6 ep1 and
    E7 ep5 (``scripts/probe_block_leakage.py``): teacher-forced mIoU 0.41 / 0.46
    with the ground truth present, 0.0000 for both once the slots are overwritten,
    matching the real-generation mIoU of ~0.02 and ~0.0001.

    ``modeling_locateanything.generate()`` fills those same slots with
    ``default_mask_token_id`` (151676), so training now does the same: the block
    inputs become mask tokens while ``labels`` keep the ground-truth coordinates.
    Position ids are untouched -- the ``position_ids[-n_future:] -= 1`` seen in
    generation only compensates for the duplicated trigger token that generation
    appends to reuse the KV cache, which does not exist here.
    """

    def __init__(
        self,
        processor,
        ignore_idx: int = -100,
        mask_block_inputs: bool = True,
        mask_token_id: int | None = None,
        emit_position_ids: bool = True,
        block_pe_offset: int = 0,
        mask_box_close: bool = True,
        n_geometry_slots: int = N_GEOMETRY_SLOTS,
        emit_image_aspect: bool = False,
    ):
        self.processor = processor
        # Training-only: per-sample image width/height, consumed (and popped) by the
        # physical-aspect direction gate. Off by default so eval batches are unchanged.
        self.emit_image_aspect = emit_image_aspect
        self.tokenizer = processor.tokenizer
        self.ignore_idx = ignore_idx
        self.mask_block_inputs = mask_block_inputs
        self.mask_token_id = DEFAULT_MASK_TOKEN_ID if mask_token_id is None else mask_token_id
        self.emit_position_ids = emit_position_ids
        self.block_pe_offset = block_pe_offset
        self.mask_box_close = mask_box_close
        if n_geometry_slots not in (4, 6):
            raise ValueError(
                f"n_geometry_slots must be 4 (x1,y1,x2,y2 -- hbb_corner) or 6 "
                f"(cx,cy,w,h,sin,cos -- obb/hbb_envelope); got {n_geometry_slots}"
            )
        self.n_geometry_slots = n_geometry_slots

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor | list]:
        texts = []
        all_images = []
        all_videos = []
        image_aspects = []

        for s in samples:
            img = s["image"]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif hasattr(img, "convert"):
                img = img.convert("RGB")
            image_aspects.append(img.width / img.height)

            prompt = s["prompt"].strip()
            target = s["target_obb_str"].strip()

            messages_user = [
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
            ]
            text_prompt = self.processor.py_apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)
            full_conv = text_prompt + target
            texts.append(full_conv)

            images, videos = self.processor.process_vision_info(messages_user)
            if images is not None:
                all_images.extend(images)
            if videos is not None:
                all_videos.extend(videos)

        # Batch encode via multimodal processor with prepared visual assets
        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Ensure numpy arrays (like image_grid_hws in Qwen2/LocateAnything) become tensors
        for k, v in list(inputs.items()):
            if isinstance(v, np.ndarray):
                inputs[k] = torch.from_numpy(v)

        # Generate label masks (ignore everything before and up to 'assistant\n').
        # labels is cloned BEFORE the block inputs are overwritten, so it keeps the
        # ground-truth coordinate tokens as targets.
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        sin_pos_indices = []
        cos_pos_indices = []
        cx_pos_indices = []
        cy_pos_indices = []
        w_pos_indices = []
        h_pos_indices = []

        assistant_marker = "<|im_start|>assistant\n"
        marker_ids = self.tokenizer.encode(assistant_marker, add_special_tokens=False)
        marker_len = len(marker_ids)

        box_open_id = self.tokenizer.encode("<box>", add_special_tokens=False)[0]
        prompt_end_indices = []
        seq_len = input_ids.shape[1]
        position_ids = (torch.arange(seq_len, dtype=torch.long)
                        .unsqueeze(0)
                        .repeat(input_ids.shape[0], 1))

        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()

            # Mask out prompt part with -100
            mask_end_idx = 0
            for i in range(len(seq) - marker_len + 1):
                if seq[i : i + marker_len] == marker_ids:
                    mask_end_idx = i + marker_len
                    break
            labels[b, :mask_end_idx] = self.ignore_idx
            prompt_end_indices.append(mask_end_idx)

            # Replace padding tokens with -100
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                labels[b][labels[b] == pad_id] = self.ignore_idx

            # Locate the geometry coordinate tokens between <box> and </box>.
            # 6-slot (obb / hbb_envelope): <box> <cx> <cy> <w> <h> <sin> <cos> </box>
            # 4-slot (hbb_corner):         <box> <x1> <y1> <x2> <y2> </box> -- x1/y1/
            #   x2/y2 are carried in the same cx_idx/cy_idx/w_idx/h_idx slots (they
            #   are just the first 4 coordinate positions either way); sin_idx/
            #   cos_idx stay -1 because there is no orientation slot to find.
            cx_idx, cy_idx, w_idx, h_idx, sin_idx, cos_idx = -1, -1, -1, -1, -1, -1
            if box_open_id in seq:
                idx = seq.index(box_open_id)
                if idx + self.n_geometry_slots + 1 < len(seq):
                    cx_idx = idx + 1
                    cy_idx = idx + 2
                    w_idx = idx + 3
                    h_idx = idx + 4
                    last_coord_idx = idx + self.n_geometry_slots
                    if self.n_geometry_slots == 6:
                        sin_idx = idx + 5
                        cos_idx = idx + 6

                    # Mirror generation: the block carries mask tokens, not ground
                    # truth. generate() forwards [<box> duplicate, mask x N-1], so
                    # the </box> slot is a mask token there too. The block is
                    # mutually visible, so leaving a real </box> in training changes
                    # the context every slot attends to; measured effect (6-slot)
                    # was a valid block 0.0% of the time at generation while an
                    # eval-mode forward on the same weights scored IoU 0.98.
                    if self.mask_block_inputs:
                        end = last_coord_idx + 2 if self.mask_box_close else last_coord_idx + 1
                        input_ids[b, cx_idx:end] = self.mask_token_id

                    # NOTE on position ids. modeling_qwen2.py derives the prefix
                    # length as find_prefix_seq_length_by_pe(position_ids), i.e. the
                    # first index where the position ids DROP, and returns -1 when
                    # they never do. A plain arange never drops, so x0_len came back
                    # as -1, `q_idx < x0_len` was False everywhere, and the
                    # coordinate slots saw only their own 8-token window: 7 of 894
                    # keys, zero prompt or image tokens
                    # (scripts/probe_block_visibility.py).
                    #
                    # Manufacturing that drop here by shifting the answer region
                    # down (block_pe_offset=2) does restore prefix visibility, but it
                    # also changes the block's rotary positions relative to
                    # generate(), which numbers the block box_pe .. box_pe+7. A gate
                    # run trained that way memorised 16 samples to loss 0.0000 and
                    # then emitted a valid block 0.0% of the time. The position ids
                    # must therefore stay a plain arange, and the prefix boundary is
                    # instead recovered inside the model by the backbone patch
                    # (patches/LocateAnything-3B, U5), which falls back to the first
                    # mask token when the position ids do not drop.
                    if self.emit_position_ids and self.block_pe_offset:
                        position_ids[b, cx_idx:] -= self.block_pe_offset

            cx_pos_indices.append(cx_idx)
            cy_pos_indices.append(cy_idx)
            w_pos_indices.append(w_idx)
            h_pos_indices.append(h_idx)
            sin_pos_indices.append(sin_idx)
            cos_pos_indices.append(cos_idx)

        inputs["input_ids"] = input_ids
        inputs["labels"] = labels
        if self.emit_position_ids:
            inputs["position_ids"] = position_ids
        inputs["cx_pos_indices"] = torch.tensor(cx_pos_indices, dtype=torch.long)
        inputs["cy_pos_indices"] = torch.tensor(cy_pos_indices, dtype=torch.long)
        inputs["w_pos_indices"] = torch.tensor(w_pos_indices, dtype=torch.long)
        if self.emit_image_aspect:
            inputs["image_aspect"] = torch.tensor(image_aspects, dtype=torch.float32)
        inputs["h_pos_indices"] = torch.tensor(h_pos_indices, dtype=torch.long)
        if self.n_geometry_slots == 6:
            inputs["sin_pos_indices"] = torch.tensor(sin_pos_indices, dtype=torch.long)
            inputs["cos_pos_indices"] = torch.tensor(cos_pos_indices, dtype=torch.long)
        else:
            # No orientation slot exists in the 4-slot corner block. Emitting None
            # (rather than an all -1 tensor) means OBBLossWrapper's explicit
            # ``is not None`` gates skip the circle/direction losses outright, so a
            # forgotten ``--lambda-circle 0``/``--lambda-direction 0`` flag cannot
            # silently index logits at position -1/-2 and corrupt training.
            inputs["sin_pos_indices"] = None
            inputs["cos_pos_indices"] = None
        # End of "<|im_start|>assistant\n": generation must start here, otherwise the
        # model is asked to continue *after* the ground-truth block instead of
        # producing it.
        inputs["prompt_end_indices"] = torch.tensor(prompt_end_indices, dtype=torch.long)
        return dict(inputs)
