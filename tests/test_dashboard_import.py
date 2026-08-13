"""Smoke test for dashboard imports."""

import importlib.util
from pathlib import Path


def test_dashboard_imports():
    app_path = Path("src/dashboard/app.py")
    spec = importlib.util.spec_from_file_location("dashboard_app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
