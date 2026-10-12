from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class UpstreamConfig:
    repo_id: str
    revision: str
    files: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "UpstreamConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(value["repo_id"], value["revision"], tuple(value["files"]))


def download_upstream(config: UpstreamConfig, destination: str | Path) -> list[Path]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name in config.files:
        url = f"https://huggingface.co/{config.repo_id}/resolve/{config.revision}/{name}"
        target = root / name
        target.write_bytes(urlopen(url, timeout=60).read())
        written.append(target)
    return written


def audit_upstream(source_dir: str | Path) -> dict[str, bool]:
    root = Path(source_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    generation = (root / "modeling_locateanything.py").read_text(encoding="utf-8")
    qwen = (root / "modeling_qwen2.py").read_text(encoding="utf-8")
    utils = (root / "generate_utils.py").read_text(encoding="utf-8")
    mask = (root / "mask_sdpa_utils.py").read_text(encoding="utf-8")
    return {
        "block_size_is_six": config["text_config"]["block_size"] == 6,
        "generation_future_tokens_is_six": "n_future_tokens: int = 6" in generation,
        "loss_diagnostics_are_four_positions": "min(4, self.model.block_size)" in qwen,
        "decoder_uses_four_coordinate_slice": "probs[1:5]" in utils and "x0[1:5]" in utils,
        "decoder_end_position_is_five": "probs[5, end_target_ids]" in utils,
        "attention_mask_accepts_block_size": "create_block_diff_mask_by_pe_4d" in mask and "block_size" in mask,
    }
