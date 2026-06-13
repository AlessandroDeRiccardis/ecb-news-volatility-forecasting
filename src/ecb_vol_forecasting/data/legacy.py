"""Traceable execution of archived raw-data stages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_legacy_script(repo_root: Path, filename: str, args: list[str] | None = None) -> None:
    """Run an archived script from the repository root."""
    script = repo_root / "legacy" / "original_scripts" / filename
    if not script.exists():
        raise FileNotFoundError(f"Archived stage not found: {script}")
    subprocess.run(
        [sys.executable, str(script), *(args or [])],
        cwd=repo_root,
        check=True,
    )
