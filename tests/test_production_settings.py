from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_SETTINGS_IMPORT = "import stanstock.settings.prod"


def _production_settings_process(secret_key: str | None) -> subprocess.CompletedProcess[str]:
    environment = {
        "DATABASE_URL": "postgresql:///stanstock",
        "DJANGO_DEBUG": "true",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    if secret_key is not None:
        environment["DJANGO_SECRET_KEY"] = secret_key

    return subprocess.run(
        [sys.executable, "-c", PRODUCTION_SETTINGS_IMPORT],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "secret_key",
    [None, "", " \t\n"],
    ids=["missing", "empty", "whitespace"],
)
def test_production_rejects_blank_secret_key(secret_key: str | None) -> None:
    result = _production_settings_process(secret_key)

    assert result.returncode != 0
    assert (
        "ImproperlyConfigured: DJANGO_SECRET_KEY is required for production settings"
        in result.stderr
    )


def test_production_preserves_valid_nonblank_secret_key() -> None:
    secret_key = f"  {'x' * 50}  "
    environment = {
        "DATABASE_URL": "postgresql:///stanstock",
        "DJANGO_DEBUG": "true",
        "DJANGO_SECRET_KEY": secret_key,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "from stanstock.settings import prod; "
                "raise SystemExit(0 if prod.SECRET_KEY == os.environ['DJANGO_SECRET_KEY'] else 2)"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail("Production settings rejected or changed a valid nonblank secret")
