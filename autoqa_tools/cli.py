from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _run_script(script: Path, cwd: Path, extra_paths: list[Path] | None = None) -> None:
    if not script.is_file():
        raise SystemExit(f"Missing entrypoint: {script}")

    os.chdir(cwd)
    _prepend_sys_path(cwd)
    _prepend_sys_path(REPO_ROOT)
    for path in extra_paths or []:
        _prepend_sys_path(path)

    runpy.run_path(str(script), run_name="__main__")


def macro() -> None:
    """Run the original PyQt5 macro automation application."""
    app_dir = REPO_ROOT / "QA-Auto"
    _run_script(app_dir / "main.py", app_dir)


def llm() -> None:
    """Run the LLM-assisted video bug triage UI."""
    app_dir = REPO_ROOT / "AutoQA_llm"
    _run_script(app_dir / "main.py", app_dir)


def yolo() -> None:
    """Run the YOLO live detection UI."""
    _run_script(REPO_ROOT / "detect_live.py", REPO_ROOT)


def check() -> None:
    """Verify that uv-installed launch targets can resolve their entrypoints."""
    checks = {
        "repo": REPO_ROOT,
        "macro": REPO_ROOT / "QA-Auto" / "main.py",
        "llm": REPO_ROOT / "AutoQA_llm" / "main.py",
        "yolo": REPO_ROOT / "detect_live.py",
        "root_readme": REPO_ROOT / "README.md",
    }
    missing = [name for name, path in checks.items() if not path.exists()]
    if missing:
        for name in missing:
            print(f"[missing] {name}: {checks[name]}")
        raise SystemExit(1)

    print("AutonomousQA uv setup check passed.")
    for name, path in checks.items():
        print(f"[ok] {name}: {path}")
