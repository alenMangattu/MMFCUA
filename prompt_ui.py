"""Minimal local prompt UI."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPONENTS_DIR = ROOT / "components"
BUILD_DIR = ROOT / ".build"

SOURCES = [
    COMPONENTS_DIR / "PromptInputApp.swift",
]


def _needs_rebuild(binary: Path) -> bool:
    if not binary.exists():
        return True

    binary_mtime = binary.stat().st_mtime
    return any(source.stat().st_mtime > binary_mtime for source in SOURCES)


def _build_binary() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    binary = BUILD_DIR / "prompt_input"

    missing = [source for source in SOURCES if not source.exists()]
    if missing:
        paths = "\n".join(f"- {source}" for source in missing)
        raise FileNotFoundError(f"Missing prompt UI source files:\n{paths}")

    if _needs_rebuild(binary):
        module_cache = BUILD_DIR / "module-cache"
        module_cache.mkdir(parents=True, exist_ok=True)
        command = [
            "swiftc",
            "-parse-as-library",
            "-module-cache-path",
            str(module_cache),
            *map(str, SOURCES),
            "-o",
            str(binary),
        ]
        env = os.environ.copy()
        env["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
        subprocess.run(command, check=True, env=env)

    return binary


def gettextfromui() -> str:
    """Show the overlay, wait for Enter, and return the submitted text."""
    completed = subprocess.run(
        [str(_build_binary())],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


get_text_from_ui = gettextfromui
