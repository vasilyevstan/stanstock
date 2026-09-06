from __future__ import annotations

import subprocess
from pathlib import Path


def clean_git_revision(project_root: Path) -> str:
    """Return HEAD for a clean checkout or fail before observed evidence is created."""
    status = _git(project_root, "status", "--porcelain", "--untracked-files=normal")
    if status.stdout.strip():
        raise ValueError(
            "Automated observed refreshes require a clean Git worktree. "
            "Commit, stash, or remove local changes before retrying."
        )
    revision = _git(project_root, "rev-parse", "HEAD").stdout.strip()
    if len(revision) != 40:
        raise ValueError("Git did not return a full 40-character HEAD revision")
    return revision


def _git(project_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Git revision validation could not be completed") from exc
    if result.returncode != 0:
        raise ValueError("Git revision validation failed") from None
    return result
