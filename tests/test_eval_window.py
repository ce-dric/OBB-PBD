import pytest

from obb_pbd.eval_window import evaluation_window


def test_evaluation_window_selects_a_disjoint_slice():
    """A 400-sample selection run must start after the reported 400 samples."""
    assert evaluation_window(dataset_size=6139, start_index=400, max_eval=400) == (400, 800)


def test_evaluation_window_zero_max_eval_runs_to_the_end():
    """Zero keeps the existing convention: evaluate every sample after the offset."""
    assert evaluation_window(dataset_size=6139, start_index=400, max_eval=0) == (400, 6139)


def test_evaluation_window_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="start_index"):
        evaluation_window(dataset_size=100, start_index=-1, max_eval=10)

    with pytest.raises(ValueError, match="start_index"):
        evaluation_window(dataset_size=100, start_index=100, max_eval=10)

    with pytest.raises(ValueError, match="max_eval"):
        evaluation_window(dataset_size=100, start_index=0, max_eval=-1)
