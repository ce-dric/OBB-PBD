# LocateAnything-3B remote-code patch

Unified diff between the remote code of [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) at revision `c32291ca5e996f5a7a485845b4f57a233936bba0` and the patched files used for all results in the paper. Distributed under the NVIDIA License (see the repository `LICENSE`).

| File | Purpose |
|---|---|
| `combined.diff` | all changes below in one file; apply with `patch -p1` inside the downloaded model folder |
| `modeling_qwen2.py.diff` | **U5** (the silent defect): when the training-time position IDs give no prefix length, recover it from the mask-token region so the coordinate slots see the image and prompt. **U4**: route the teacher-forced evaluation path (`input_ids=None`) to the training-mode mask. **U1**: remove the hard-coded four-slot block bound. **U2**: return a single `CausalLMOutputWithPast` in training mode |
| `modeling_locateanything.py.diff` | **U3**: `.clone()` after the embedding lookup and reshape, so the in-place multimodal token injection does not write into an autograd view; plus `inputs_embeds`/`**kwargs` in the forward signature for PEFT compatibility |
| `generate_utils.py.diff` | replaces hard-coded four-coordinate slices with a block-length-derived count, so a six-slot block can pass through `generate()`; the sequential `generate()` path is still not fully validated for six slots, and all reported results use the direct decoder |
| `mask_sdpa_utils.py.diff` | empty (no change needed); listed for completeness |
| `SOURCE_FILE_HASHES.txt` | SHA-256 of each file before (`pristine=`) and after (`patched=`) the patch |

After patching, clear the `trust_remote_code` cache (`~/.cache/huggingface/modules/transformers_modules/LocateAnything_hyphen_3B`) so that the patched files are loaded.
