"""Unit tests for the block_size plumbing helper.

Regression target: E4/E5/E6 set ``config.block_size = 8`` only, which does not
reach ``text_config`` and therefore left the language model at width 6 while the
OBB block is 8 tokens long.
"""

import pytest

from obb_pbd.model_config import (
    HBB_BLOCK_SIZE,
    OBB_BLOCK_SIZE,
    OBB_N_COORDS,
    OBB_N_FUTURE_TOKENS,
    apply_obb_block_size,
    assert_block_size,
    effective_block_size,
    resolve_language_model,
)


class _TextConfig:
    """Stand-in for Qwen2Config; on-disk default is 6."""

    def __init__(self, block_size=HBB_BLOCK_SIZE):
        self.block_size = block_size


class _Config:
    """Stand-in for LocateAnythingConfig."""

    def __init__(self):
        self.text_config = _TextConfig()


class _InnerModel:
    """Stand-in for the Qwen2 model that owns block_size."""

    def __init__(self, block_size):
        self.block_size = block_size


class _LanguageModel:
    def __init__(self, block_size):
        self.model = _InnerModel(block_size)


class _Model:
    def __init__(self, block_size):
        self.language_model = _LanguageModel(block_size)


class _PeftWrapper:
    def __init__(self, inner):
        self.base_model = inner


class _DDPWrapper:
    def __init__(self, inner):
        self.module = inner


def test_constants_are_self_consistent():
    assert OBB_BLOCK_SIZE == 8
    assert OBB_N_COORDS == 6, "cx, cy, w, h, sin2theta, cos2theta"
    # the whole block must sit inside the MTP logits window, so this is the
    # block size and NOT the coordinate count
    assert OBB_N_FUTURE_TOKENS == OBB_BLOCK_SIZE


def test_apply_sets_text_config_not_just_top_level():
    cfg = _Config()
    assert cfg.text_config.block_size == HBB_BLOCK_SIZE

    apply_obb_block_size(cfg)

    assert cfg.text_config.block_size == OBB_BLOCK_SIZE, (
        "text_config is the level modeling_qwen2.py reads; this is the bug that "
        "silently trained E4/E5/E6 with a 6-wide mask"
    )
    assert cfg.block_size == OBB_BLOCK_SIZE


def test_apply_is_idempotent():
    cfg = _Config()
    apply_obb_block_size(cfg)
    apply_obb_block_size(cfg)
    assert cfg.text_config.block_size == OBB_BLOCK_SIZE


def test_apply_supports_hbb_retention_experiments():
    cfg = _Config()
    apply_obb_block_size(cfg, block_size=HBB_BLOCK_SIZE)
    assert cfg.text_config.block_size == HBB_BLOCK_SIZE
    assert cfg.n_future_tokens == HBB_BLOCK_SIZE - 2


def test_apply_rejects_degenerate_block_size():
    with pytest.raises(ValueError):
        apply_obb_block_size(_Config(), block_size=2)


def test_apply_refuses_config_without_text_config():
    class _Bare:
        pass

    with pytest.raises(AttributeError):
        apply_obb_block_size(_Bare())


@pytest.mark.parametrize(
    "wrap",
    [
        lambda m: m,
        _PeftWrapper,
        _DDPWrapper,
        lambda m: _DDPWrapper(_PeftWrapper(m)),
        lambda m: _PeftWrapper(_DDPWrapper(m)),
    ],
)
def test_resolve_language_model_sees_through_wrappers(wrap):
    model = wrap(_Model(OBB_BLOCK_SIZE))
    assert resolve_language_model(model).block_size == OBB_BLOCK_SIZE
    assert effective_block_size(model) == OBB_BLOCK_SIZE


def test_assert_block_size_passes_on_match():
    assert assert_block_size(_Model(OBB_BLOCK_SIZE)) == OBB_BLOCK_SIZE


def test_assert_block_size_raises_on_the_e4_e5_e6_defect():
    model = _Model(HBB_BLOCK_SIZE)  # what E4/E5/E6 actually loaded
    with pytest.raises(RuntimeError) as excinfo:
        assert_block_size(model, expected=OBB_BLOCK_SIZE)
    assert "text_config" in str(excinfo.value)


def test_assert_block_size_non_strict_reports_instead_of_raising():
    model = _Model(HBB_BLOCK_SIZE)
    assert assert_block_size(model, expected=OBB_BLOCK_SIZE, strict=False) == HBB_BLOCK_SIZE


def test_resolve_returns_none_when_absent():
    class _Nothing:
        pass

    assert resolve_language_model(_Nothing()) is None
    assert effective_block_size(_Nothing()) is None
