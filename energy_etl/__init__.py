"""Energy ETL package scaffold.

This module exposes convenience accessors so scripts such as `run_pipeline.py`
can rely on package-style imports (``from energy_etl import import_batch``).
Modules are resolved lazily to avoid heavy dependencies at import time and to
prevent circular import errors when the package is executed as a script.
"""
from __future__ import annotations

import sys
from importlib import import_module
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Dict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

_MODULE_ALIASES = {
    "import_batch": PROJECT_ROOT / "import_batch.py",
    "append_updates": PROJECT_ROOT / "append_updates.py",
    "build_summary": PROJECT_ROOT / "build_summary.py",
    "summarize_fossil_fuels": PROJECT_ROOT / "summarize_fossil_fuels.py",
    "fix_reserves": PROJECT_ROOT / "fix_reserves.py",
}


def __getattr__(name: str) -> ModuleType:
    if name in _MODULE_ALIASES:
        module = _load_module(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Convenience dictionary for introspection or dynamic access
def modules() -> Dict[str, ModuleType]:
    return {key: getattr(__import__(__name__), key) for key in _MODULE_ALIASES}


def _load_module(name: str) -> ModuleType:
    target = _MODULE_ALIASES[name]
    if isinstance(target, Path) and target.exists():
        loader = SourceFileLoader(name, str(target))
        module = loader.load_module()  # type: ignore[deprecated-decorator]
        sys.modules[f"{__name__}.{name}"] = module
        return module
    # fallback to standard import if alias is string or missing file
    module = import_module(f"{__name__}.{name}")
    return module


# Commonly referenced paths
DATA_DIR = PACKAGE_ROOT / "data"
TEXT_DIR = PACKAGE_ROOT / "Text files"

__all__ = [
    *list(_MODULE_ALIASES),
    "modules",
    "PACKAGE_ROOT",
    "DATA_DIR",
    "TEXT_DIR",
]
