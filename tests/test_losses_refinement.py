"""Tests for the continuous coordinate refinement loss.

torch is deliberately not a declared dependency of this package: pyproject lists only
numpy, Pillow and shapely, because the local environment is for the geometry and codec
tests and anything needing torch runs inside the CVLab container. This module therefore
skips rather than failing collection, so that the documented local command

    python -m pip install -e ".[dev]" && python -m pytest

works on a machine without torch instead of erroring out before any test runs.
"""
import pytest

torch = pytest.importorskip("torch", reason="the loss operates on torch logits")

import torch.nn as nn  # noqa: E402  imported after the skip guard on purpose
from types import SimpleNamespace  # noqa: E402

from obb_pbd.losses import OBBLossWrapper, compute_continuous_coordinate_loss  # noqa: E402
from obb_pbd.model_config import COORD_VOCAB_START_IDX  # noqa: E402

# compute_continuous_coordinate_loss reads logits at ``pos - 1``, because
# modeling_qwen2.py scores shift_logits[..., :-1] against shift_labels[..., 1:]: the slot at
# position p is predicted by the logits at p - 1. That is defect O2, fixed on 2026-07-30.
# These tests originally wrote the prediction at ``pos`` and therefore did not exercise the
# loss at all. One of them passed anyway, because reading all-zero logits gives a uniform
# distribution whose expected bin is 500, which happened to be the target.
PREDICTION_OFFSET = -1

# The fixtures used to hardcode 151669, which is `</box>` rather than `<0>` -- the exact
# value of defect O3. Any base works for the loss arithmetic, so a reader could copy that
# literal out of here and reintroduce the eight-bin shift. Bind to the real constant
# instead, which also makes these tests exercise the shipped vocabulary layout.
COORD_START = COORD_VOCAB_START_IDX
NUM_BINS = 1001
# Bins occupy COORD_START .. COORD_START + 1000, so the vocabulary has to reach past them.
VOCAB_SIZE = COORD_START + NUM_BINS


def test_continuous_coordinate_loss_zero_error():
    """Verify loss is minimal when soft probabilities peak exactly at ground truth tokens."""
    batch_size, seq_len, vocab_size = 2, 10, VOCAB_SIZE
    coord_start = COORD_START
    
    # Target tokens are coordinates <500> (id COORD_START + 500 == 152177)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    pos = torch.tensor([3, 4], dtype=torch.long)
    target_val = 500
    labels[0, 3] = coord_start + target_val
    labels[1, 4] = coord_start + target_val

    # Logits heavily favouring the target coordinate, at the position that predicts it
    logits = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32)
    logits[0, 3 + PREDICTION_OFFSET, coord_start + target_val] = 50.0
    logits[1, 4 + PREDICTION_OFFSET, coord_start + target_val] = 50.0

    loss = compute_continuous_coordinate_loss(logits, labels, [pos], coord_vocab_start_idx=coord_start)
    assert loss.item() < 1e-4, f"Expected near-zero loss for perfect prediction, got {loss.item()}"


def test_continuous_coordinate_loss_reads_the_shifted_position():
    """A prediction written at ``pos`` instead of ``pos - 1`` must not score as correct.

    This is the guard the previous version of this file lacked. Without it, reverting the O2
    alignment fix would leave every test in this file green.
    """
    vocab_size = VOCAB_SIZE
    coord_start = COORD_START
    labels = torch.full((1, 6), -100, dtype=torch.long)
    pos = torch.tensor([3], dtype=torch.long)
    target_val = 800
    labels[0, 3] = coord_start + target_val

    misplaced = torch.zeros((1, 6, vocab_size), dtype=torch.float32)
    misplaced[0, 3, coord_start + target_val] = 50.0  # the pre-fix convention

    loss = compute_continuous_coordinate_loss(misplaced, labels, [pos],
                                              coord_vocab_start_idx=coord_start)
    # Position 2 is all zeros, so the soft-argmax expectation is bin 500 against a target of
    # 800, which must register as a real error rather than as a perfect hit.
    assert loss.item() > 1e-3, (
        "Writing the prediction at pos rather than pos-1 scored as correct, so the loss is "
        "reading the wrong position again")


def test_continuous_coordinate_loss_directional_gradient():
    """Verify loss provides proportional gradients across discrete bin distances."""
    batch_size, seq_len, vocab_size = 1, 5, VOCAB_SIZE
    coord_start = COORD_START
    labels = torch.full((1, 5), -100, dtype=torch.long)
    pos = torch.tensor([2], dtype=torch.long)
    labels[0, 2] = coord_start + 800  # Ground truth is <800>

    # Case 1: Model predicts near <750>
    logits1 = torch.zeros((1, 5, vocab_size), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits1[0, 2 + PREDICTION_OFFSET, coord_start + 750] = 20.0
    
    # Case 2: Model predicts further away at <200>
    logits2 = torch.zeros((1, 5, vocab_size), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits2[0, 2 + PREDICTION_OFFSET, coord_start + 200] = 20.0

    loss1 = compute_continuous_coordinate_loss(logits1, labels, [pos], coord_vocab_start_idx=coord_start)
    loss2 = compute_continuous_coordinate_loss(logits2, labels, [pos], coord_vocab_start_idx=coord_start)

    assert loss2.item() > loss1.item() * 5.0, "Smooth-L1 loss should penalize distant predictions much more than near ones."


def test_obb_loss_wrapper_full_integration():
    """Verify OBBLossWrapper processes CE, spatial refinement, and circular consistency properly."""
    vocab_size = VOCAB_SIZE
    coord_start = COORD_START
    
    outputs = SimpleNamespace(
        loss=torch.tensor(1.5, dtype=torch.float32),
        logits=torch.randn((2, 10, vocab_size), dtype=torch.float32)
    )
    labels = torch.full((2, 10), -100, dtype=torch.long)
    labels[0, 2] = coord_start + 400
    labels[0, 3] = coord_start + 600
    
    wrapper = OBBLossWrapper(lambda_circle=0.01, lambda_center=2.5, coord_vocab_start_idx=coord_start)
    
    total_loss, metrics = wrapper(
        outputs,
        labels=labels,
        cx_pos_indices=torch.tensor([2, -1]),
        cy_pos_indices=torch.tensor([3, -1]),
        w_pos_indices=torch.tensor([-1, -1]),
        h_pos_indices=torch.tensor([-1, -1]),
        sin_pos_indices=torch.tensor([-1, -1]),
        cos_pos_indices=torch.tensor([-1, -1]),
    )
    
    assert total_loss.item() >= 1.5, "Total loss should include base CE loss plus spatial refinement penalties."
    assert metrics["loss_ce"] == 1.5
    assert "loss_center" in metrics and metrics["loss_center"] > 0.0, "loss_center should be non-zero when spatial indices are passed."
    print("✅ Full OBB Loss Wrapper integration test passed successfully!")
