from pathlib import Path

from obb_pbd.upstream import audit_upstream


def test_audit_detects_reviewed_hard_codings(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"text_config":{"block_size":6}}', encoding="utf-8")
    (tmp_path / "modeling_locateanything.py").write_text("n_future_tokens: int = 6", encoding="utf-8")
    (tmp_path / "modeling_qwen2.py").write_text("max_n_future_tokens = min(4, self.model.block_size)", encoding="utf-8")
    (tmp_path / "generate_utils.py").write_text("probs[1:5]\nprobs[5, end_target_ids]\nx0[1:5]", encoding="utf-8")
    (tmp_path / "mask_sdpa_utils.py").write_text("def create_block_diff_mask_by_pe_4d(block_size): pass", encoding="utf-8")

    result = audit_upstream(tmp_path)
    assert all(result.values())
