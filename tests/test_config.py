from __future__ import annotations

from pathlib import Path

from stgpt.config import StGPTConfig


def test_config_env_expansion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STGPT_TMP", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text(
        """
case_name: env_case
data:
  mode: synthetic
  output_dir: ${STGPT_TMP}/case
  n_cells: 8
  n_genes: 12
model:
  d_model: 32
  n_heads: 4
  n_layers: 1
  max_genes: 8
  image_size: 32
training:
  batch_size: 4
  max_steps: 2
  output_dir: ${STGPT_TMP}/train
  device: cpu
""",
        encoding="utf-8",
    )
    cfg = StGPTConfig.from_file(config, preset="smoke")
    assert cfg.case_name == "env_case"
    assert str(tmp_path) in cfg.data.output_dir
    assert cfg.training.device == "cpu"
    assert cfg.training.max_steps == 2
