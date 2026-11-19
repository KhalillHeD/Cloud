"""Import the Lambda handlers from scripts (local mode, seeding, tests).

Several Lambdas use the module name app.py, so each is loaded from its file under a
unique name. The agent's sibling modules (prompts, tools, llm) need agent/ on sys.path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "common", ROOT / "agent", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FILES = {
    "producer": ROOT / "producer" / "app.py",
    "processing": ROOT / "processing" / "app.py",
    "forecast": ROOT / "ml" / "inference" / "app.py",
    "api": ROOT / "api" / "app.py",
    "agent": ROOT / "agent" / "handler.py",
}
_LOADED: dict = {}


def load(name: str):
    if name not in _LOADED:
        spec = importlib.util.spec_from_file_location(f"component_{name}", FILES[name])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _LOADED[name] = mod
    return _LOADED[name]
