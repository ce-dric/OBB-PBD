"""Repo hygiene guard against the block_size plumbing defect.

``modeling_qwen2.py`` resolves the parallel-block width from ``text_config``.
Any script that sets ``config.block_size`` without also setting
``config.text_config.block_size`` silently runs with the upstream default of 6,
which trains an 8-token OBB block behind a 6-wide attention mask.

This test fails if that pattern reappears anywhere under ``scripts/``.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Files that legitimately contain the bare pattern as documentation / tooling
# rather than as live configuration.
EXEMPT: set[str] = set()

TOP_LEVEL_ASSIGN = "config.block_size"
TEXT_LEVEL_ASSIGN = "config.text_config.block_size"
HELPER_CALL = "build_obb_config"


def _python_files():
    if not SCRIPTS_DIR.is_dir():
        pytest.skip("scripts/ directory not present")
    return sorted(p for p in SCRIPTS_DIR.rglob("*.py") if p.name not in EXEMPT)


def test_scripts_dir_is_present_and_nonempty():
    assert _python_files(), "expected at least one script to scan"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_script_sets_block_size_without_text_config(path):
    source = path.read_text(encoding="utf-8", errors="replace")

    if TOP_LEVEL_ASSIGN not in source:
        return  # script does not touch block_size at all

    # Either it goes through the helper, or it must also set the text level.
    uses_helper = HELPER_CALL in source
    sets_text_level = TEXT_LEVEL_ASSIGN in source

    assert uses_helper or sets_text_level, (
        f"{path.relative_to(REPO_ROOT)} assigns {TOP_LEVEL_ASSIGN} but never "
        f"{TEXT_LEVEL_ASSIGN} and does not call {HELPER_CALL}(). "
        "modeling_qwen2.py reads the block width from text_config, so this "
        "script would silently run with block_size=6 on an 8-token OBB block. "
        "Use obb_pbd.model_config.build_obb_config()."
    )


def test_training_scripts_verify_block_size_at_runtime():
    """Training runs are expensive; they must assert the width before starting."""
    training = sorted(SCRIPTS_DIR.glob("train*.py"))
    if not training:
        pytest.skip("no training scripts found")

    missing = []
    for path in training:
        source = path.read_text(encoding="utf-8", errors="replace")
        if "assert_block_size" not in source and "text_config.block_size" not in source:
            missing.append(path.name)

    assert not missing, (
        "these training scripts neither call assert_block_size() nor set "
        f"text_config.block_size: {missing}"
    )
