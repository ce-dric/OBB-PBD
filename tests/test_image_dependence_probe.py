from __future__ import annotations

import pytest

from obb_pbd.image_dependence import (
    intervention_variants,
    seeded_derangement,
    summarise,
)


def test_seeded_derangement_is_deterministic_and_has_no_fixed_points():
    first = seeded_derangement(100, seed=0)
    second = seeded_derangement(100, seed=0)

    assert first == second
    assert sorted(first) == list(range(100))
    assert all(source != target for target, source in enumerate(first))


def test_interventions_replace_only_the_requested_modality():
    raw = {"image": "image-a", "prompt": "prompt-a", "metadata": "target"}
    shuffled = {
        "image": "image-b",
        "prompt": "prompt-b",
        "metadata": "shuffled",
    }

    variants = intervention_variants(raw, shuffled)

    assert variants["matched"] is raw
    assert variants["image_shuffled"] == {
        "image": "image-b", "prompt": "prompt-a", "metadata": "target"
    }
    assert variants["prompt_shuffled"] == {
        "image": "image-a", "prompt": "prompt-b", "metadata": "target"
    }


def test_summarise_uses_target_image_cluster_bootstrap_and_reports_acc75():
    # Ten correlated records per image make the cluster interval [0, 1].  A
    # per-record bootstrap would instead produce a much narrower interval, so
    # this catches accidental regression to record-level resampling.
    ious = [0.0] * 10 + [1.0] * 10
    image_ids = ["scene-a"] * 10 + ["scene-b"] * 10

    result = summarise(
        ious=ious,
        angles=[10.0, 20.0],
        bin_rows=[[0, 0, 0, 0, 0, 0]] * 20,
        decoded=20,
        total=20,
        image_ids=image_ids,
        n_boot=10_000,
        bootstrap_seed=0,
    )

    assert result["mIoU"] == pytest.approx(0.5)
    assert result["mIoU_ci95"] == pytest.approx([0.0, 1.0])
    assert result["n_images"] == 2
    assert result["acc75"] == pytest.approx(0.5)
    assert result["angle_mae"] == pytest.approx(15.0)


def test_summarise_rejects_misaligned_image_ids():
    with pytest.raises(ValueError, match="image_ids"):
        summarise(
            ious=[0.1, 0.2],
            angles=[10.0, 20.0],
            bin_rows=[],
            decoded=2,
            total=2,
            image_ids=["only-one"],
            n_boot=100,
            bootstrap_seed=0,
        )
