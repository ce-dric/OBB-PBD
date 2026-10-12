"""Single source of truth for OBB-PBD block geometry and config plumbing.

Why this module exists
----------------------
``modeling_qwen2.py`` resolves the parallel-block width from the *text* config::

    self.block_size = getattr(config, 'block_size', 6)   # config == text_config

Setting ``block_size`` on the top-level ``LocateAnythingConfig`` therefore has
**no effect**. Measured on the shipped checkpoint::

    config.block_size = 8                  -> language_model.model.block_size == 6
    config.text_config.block_size = 8      -> language_model.model.block_size == 8

E4/E5/E6 set only the top-level attribute and were consequently trained with an
8-token OBB block behind a 6-wide attention mask, placing the ``sin2theta`` and
``cos2theta`` slots outside the parallel window. Every evaluation and
visualisation script carried the same defect.

Use :func:`build_obb_config` everywhere instead of hand-rolling the config, and
call :func:`assert_block_size` right after the model is instantiated so a silent
regression becomes a loud failure.
"""

from __future__ import annotations

__all__ = [
    "OBB_BLOCK_SIZE",
    "OBB_N_COORDS",
    "OBB_N_FUTURE_TOKENS",
    "HBB_BLOCK_SIZE",
    "HBB_N_COORDS",
    "COORD_VOCAB_START_IDX",
    "apply_obb_block_size",
    "build_obb_config",
    "resolve_language_model",
    "effective_block_size",
    "assert_block_size",
]

#: ``<box> cx cy w h sin2theta cos2theta </box>``
OBB_BLOCK_SIZE = 8

#: geometry slots inside the block (block length minus the two delimiters)
OBB_N_COORDS = OBB_BLOCK_SIZE - 2

#: value to pass as ``generate(n_future_tokens=...)``.
#:
#: ``_sample_token_in_mtp`` slices ``logits[:, -n_future_tokens:, :]`` and the
#: patched ``generate_utils`` derives the coordinate count as
#: ``probs.size(0) - 2``. The whole block must therefore be inside the window,
#: so this equals the block size, not the number of coordinates.
OBB_N_FUTURE_TOKENS = OBB_BLOCK_SIZE

#: original upstream layout, ``<box> x1 y1 x2 y2 </box>``
HBB_BLOCK_SIZE = 6

#: geometry slots inside the HBB block (block length minus the two delimiters).
#: Used by the matched 4-slot HBB-PBD baseline (``dataset.py``'s
#: ``target_mode="hbb_corner"``), not by the ``hbb_envelope`` control, which
#: still uses the 6-slot/8-token OBB block with a degenerate axis-aligned angle.
HBB_N_COORDS = HBB_BLOCK_SIZE - 2

# Token id of `<0>`, the first of the 1000 coordinate bins. Do not write 151669 here:
# that is `</box>`, and using it shifts every coordinate by eight bins. That defect was
# found in losses.py and lived on as a stale default in an earlier trainer, so the value now has
# exactly one home. Verified against the shipped tokenizer:
#   tokenizer.convert_tokens_to_ids("<0>") == 151677
COORD_VOCAB_START_IDX = 151677


def apply_obb_block_size(config, block_size: int = OBB_BLOCK_SIZE):
    """Write ``block_size`` at every level that any component might read.

    Kept free of ``transformers`` imports so it can be unit tested with a stub.

    Returns the same config object for chaining.
    """
    if block_size < 3:
        raise ValueError(
            f"block_size must leave room for <box>, at least one slot and </box>; got {block_size}"
        )

    # top level: harmless, and some helper code does read it
    setattr(config, "block_size", block_size)
    setattr(config, "n_future_tokens", block_size - 2)

    # text level: this is the one modeling_qwen2.py actually consumes
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise AttributeError(
            "config has no .text_config; cannot set the block width that "
            "modeling_qwen2.py reads. Refusing to continue silently."
        )
    setattr(text_config, "block_size", block_size)
    setattr(text_config, "n_future_tokens", block_size - 2)

    return config


def build_obb_config(model_path: str, block_size: int = OBB_BLOCK_SIZE, **from_pretrained_kwargs):
    """Load the model config and plumb ``block_size`` through correctly."""
    from transformers import AutoConfig

    from_pretrained_kwargs.setdefault("trust_remote_code", True)
    config = AutoConfig.from_pretrained(model_path, **from_pretrained_kwargs)
    return apply_obb_block_size(config, block_size=block_size)


def resolve_language_model(model):
    """Return the Qwen2 inner model that owns ``block_size``, or ``None``.

    Tolerates PEFT / DDP / ``AutoModel`` wrapping in any order.
    """
    seen = 0
    candidate = model
    while candidate is not None and seen < 8:
        seen += 1
        inner = getattr(candidate, "language_model", None)
        if inner is not None:
            return getattr(inner, "model", inner)
        for attr in ("module", "base_model", "model"):
            nxt = getattr(candidate, attr, None)
            if nxt is not None and nxt is not candidate:
                candidate = nxt
                break
        else:
            return None
    return None


def effective_block_size(model):
    """The block width the model will actually use, or ``None`` if not found."""
    lm = resolve_language_model(model)
    if lm is None:
        return None
    return getattr(lm, "block_size", None)


def assert_block_size(model, expected: int = OBB_BLOCK_SIZE, *, strict: bool = True):
    """Verify the loaded model really uses ``expected`` as its block width.

    Call this immediately after ``from_pretrained``. With ``strict=True`` a
    mismatch raises; otherwise it returns the observed value so the caller can
    warn. Returns the observed block size.
    """
    actual = effective_block_size(model)
    if actual == expected:
        return actual

    message = (
        f"block_size mismatch: language model reports {actual!r}, expected {expected}. "
        "Set it via obb_pbd.model_config.build_obb_config() / apply_obb_block_size(); "
        "assigning config.block_size alone does NOT propagate to text_config, which "
        "is what modeling_qwen2.py reads."
    )
    if strict:
        raise RuntimeError(message)
    return actual
