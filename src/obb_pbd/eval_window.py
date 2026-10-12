"""Shared sample-window validation for evaluation scripts."""


def evaluation_window(dataset_size: int, start_index: int = 0,
                      max_eval: int = 0) -> tuple[int, int]:
    """Return the half-open sample range selected by an evaluation run.

    ``max_eval=0`` preserves the evaluator's existing convention of running to
    the end of the split. A non-zero value caps the range without crossing the
    dataset boundary.
    """
    if start_index < 0 or start_index >= dataset_size:
        raise ValueError("start_index must be within the dataset")
    if max_eval < 0:
        raise ValueError("max_eval must be non-negative")

    stop = dataset_size if max_eval == 0 else min(dataset_size, start_index + max_eval)
    return start_index, stop
