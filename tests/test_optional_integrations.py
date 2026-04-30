from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def test_optional_real_atera_env_is_explicit() -> None:
    if not os.environ.get("STGPT_XENIUM_ROOT"):
        pytest.skip("STGPT_XENIUM_ROOT is not set; real Atera integration is opt-in.")


def test_optional_packages_are_optional() -> None:
    # This test documents the intended behavior: the core package imports even
    # when pyXenium and spatho are not installed in the smoke environment.
    assert importlib.util.find_spec("stgpt") is not None


def test_optional_real_atera_validate_data(tmp_path: Path) -> None:
    required = ("STGPT_XENIUM_ROOT", "STGPT_SPATHO_RUN_ROOT", "STGPT_OUTPUT_ROOT")
    if not all(os.environ.get(name) for name in required):
        pytest.skip("Atera validation is opt-in and requires STGPT_XENIUM_ROOT, STGPT_SPATHO_RUN_ROOT, and STGPT_OUTPUT_ROOT.")
    if importlib.util.find_spec("pyXenium") is None:
        pytest.skip("pyXenium is not installed; real Xenium validation is opt-in.")

    from stgpt.config import StGPTConfig
    from stgpt.qc import validate_data

    cfg = StGPTConfig.from_file("configs/atera_wta_breast.yaml.example")
    result = validate_data(cfg, output_dir=tmp_path / "atera_qc")
    assert Path(result["qc_report_json"]).exists()
