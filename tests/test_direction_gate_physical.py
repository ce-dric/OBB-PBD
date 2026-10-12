"""Physical-aspect direction gate (gate ablation, Section 3.4). Needs torch; runs on the server."""
import pytest

torch = pytest.importorskip("torch")

from obb_pbd.losses import compute_cosine_directional_loss  # noqa: E402


def _batch(w_level, h_level, start=10, bins=1001):
    # one sample; slots at positions 1..6 = cx, cy, w, h, sin, cos
    labels = torch.full((1, 8), -100, dtype=torch.long)
    labels[0, 3] = start + w_level
    labels[0, 4] = start + h_level
    labels[0, 5] = start + 700  # sin
    labels[0, 6] = start + 300  # cos
    logits = torch.zeros((1, 8, start + bins))
    idx = lambda i: torch.tensor([i])  # noqa: E731
    return dict(logits=logits, labels=labels, sin_pos_indices=idx(5), cos_pos_indices=idx(6),
                w_pos_indices=idx(3), h_pos_indices=idx(4), coord_vocab_start_idx=start)


def test_normalized_gate_drops_box_that_is_physically_elongated():
    # w/W = 0.100, h/H = 0.100 on a 16:9 image: normalized ratio 1.0 (gated out),
    # physical ratio 1.78 (kept).
    batch = _batch(100, 100)
    normalized = compute_cosine_directional_loss(**batch)
    physical = compute_cosine_directional_loss(**batch, image_aspect=torch.tensor([16 / 9]))
    assert normalized.item() == 0.0
    assert physical.item() > 0.0


def test_physical_gate_drops_box_that_is_physically_square():
    # w/W = 0.0563, h/H = 0.100 on a 16:9 image: normalized ratio 1.78 (kept),
    # physical ratio ~1.0 (gated out).
    batch = _batch(56, 100)
    normalized = compute_cosine_directional_loss(**batch)
    physical = compute_cosine_directional_loss(**batch, image_aspect=torch.tensor([16 / 9]))
    assert normalized.item() > 0.0
    assert physical.item() == 0.0
