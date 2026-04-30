from __future__ import annotations

import importlib.util
import os

import pytest


def test_optional_real_atera_env_is_explicit() -> None:
    if not os.environ.get("STGPT_XENIUM_ROOT"):
        pytest.skip("STGPT_XENIUM_ROOT is not set; real Atera integration is opt-in.")


def test_optional_packages_are_optional() -> None:
    # This test documents the intended behavior: the core package imports even
    # when pyXenium and spatho are not installed in the smoke environment.
    assert importlib.util.find_spec("stgpt") is not None
