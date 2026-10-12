"""CPU proof that the shared parallel-box decoder recovers the box it was given.

Why this test is worth its lines
--------------------------------
``decode_block_prediction`` is the single read-side implementation of parallel box
decoding: ``eval_pbd_direct.py`` scores its output, and Figure 2 is rendered
from those predictions. An off-by-eight in the coordinate vocabulary start index is exactly the kind of defect
that already happened once in ``src/obb_pbd/losses.py`` (151669, which is ``</box>``,
instead of 151677), and it would produce a plausible-looking box rather than a crash.

This is pinned with synthetic logits on CPU. No GPU, no model weights, no dataset.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from obb_pbd.geometry import OrientedBox
from obb_pbd.metrics import rotated_iou
from obb_pbd.tokens import OBBTokenCodec

torch = pytest.importorskip("torch", reason="the decoder reads torch logits")

# The real coordinate vocabulary start. 151669 is `</box>`; using it shifts every
# coordinate by eight bins.
COORD_VOCAB_START = 151677
VOCAB_SIZE = COORD_VOCAB_START + 1001
BLOCK_SIZE = 8
N_COORDS = 6
IMAGE_SIZE = (4000, 2250)          # AVVG frames are DJI UAV stills at this size
QUANTISATION_IOU_FLOOR = 0.95


def load_decoder():
    """Import scripts/eval_pbd_direct.py without importing torch-heavy siblings.

    The module imports transformers and peft at module scope, so it is only importable
    where those are installed. Skip rather than fail on a machine that has torch but not
    the full training stack.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_pbd_direct.py"
    spec = importlib.util.spec_from_file_location("eval_pbd_direct", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_pbd_direct"] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as error:                      # pragma: no cover
        pytest.skip(f"eval_pbd_direct needs the training stack: {error}")
    return module


def synthetic_logits(bins, box_idx: int, sequence_length: int):
    """Logits whose argmax at the six geometry slots spells out ``bins``.

    Everything else is left at zero, and the winning token is given a clearly positive
    score, so the test fails if the decoder reads the wrong positions rather than
    accidentally passing on a tie.
    """
    logits = torch.zeros((sequence_length, VOCAB_SIZE), dtype=torch.float32)
    for offset, value in enumerate(bins):
        position = box_idx + offset
        if position >= sequence_length:
            # the truncation case: the block runs off the end of the sequence, which is
            # exactly what the decoder has to report as a failure
            break
        logits[position, COORD_VOCAB_START + value] = 10.0
    return logits


def test_synthetic_logits_round_trip_to_the_encoded_box():
    module = load_decoder()
    codec = OBBTokenCodec(bins=1000)

    box = OrientedBox(cx=1850.0, cy=1120.0, w=310.0, h=140.0, theta_deg=37.5)
    bins = codec.encode(box, image_size=IMAGE_SIZE)
    assert len(bins) == N_COORDS

    box_idx = 12
    logits = synthetic_logits(bins, box_idx, box_idx + BLOCK_SIZE)

    prediction, read_bins = module.decode_block_prediction(
        logits, box_idx, codec, IMAGE_SIZE, COORD_VOCAB_START)

    assert read_bins == list(bins)
    assert prediction is not None
    assert rotated_iou(prediction, box) >= QUANTISATION_IOU_FLOOR
    # the angle survives the sin2theta/cos2theta detour, which is the part that a bin
    # offset would corrupt most visibly
    assert prediction.theta_deg == pytest.approx(box.theta_deg, abs=1.0)


def test_wrong_vocab_start_is_detected_by_the_same_check():
    """Guards the guard: an off-by-eight must not still pass the IoU floor.

    If it did, this test file would be decoration rather than a check.
    """
    module = load_decoder()
    codec = OBBTokenCodec(bins=1000)

    box = OrientedBox(cx=1850.0, cy=1120.0, w=310.0, h=140.0, theta_deg=37.5)
    bins = codec.encode(box, image_size=IMAGE_SIZE)
    box_idx = 12
    logits = synthetic_logits(bins, box_idx, box_idx + BLOCK_SIZE)

    stale_start = 151669                              # `</box>`, the defect that happened
    prediction, _ = module.decode_block_prediction(
        logits, box_idx, codec, IMAGE_SIZE, stale_start)

    assert prediction is None or rotated_iou(prediction, box) < QUANTISATION_IOU_FLOOR


def test_block_running_past_the_sequence_returns_no_box():
    """A truncated block must be reported as a failure, never as a box.

    An earlier evaluator scored decode failures as IoU 1.0 by substituting
    the target. Returning None here is what keeps that from being possible.
    """
    module = load_decoder()
    codec = OBBTokenCodec(bins=1000)

    box = OrientedBox(cx=100.0, cy=100.0, w=40.0, h=20.0, theta_deg=10.0)
    bins = codec.encode(box, image_size=IMAGE_SIZE)
    box_idx = 4
    # only three of the six slots exist in this sequence
    logits = synthetic_logits(bins, box_idx, box_idx + 3)

    prediction, _ = module.decode_block_prediction(
        logits, box_idx, codec, IMAGE_SIZE, COORD_VOCAB_START)
    assert prediction is None
