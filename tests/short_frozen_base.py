"""Load the frozen short-v1/v2 implementation from the d637 base revision.

The four modules are installed under their real names in dependency order so
imports made while executing the base service cannot resolve to working-tree
config, indicators, or scoring code.  Configuration bytes are also read from
the base object database.  There is deliberately no working-tree fallback.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

BASE_SHA = "d6374eb8a25361eb813e1ca79086a696588e1585"
REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_MODULE_PATHS: tuple[tuple[str, str], ...] = (
    ("stanstock.research.config", "src/stanstock/research/config.py"),
    ("stanstock.research.indicators", "src/stanstock/research/indicators.py"),
    ("stanstock.research.scoring", "src/stanstock/research/scoring.py"),
    ("stanstock.research.service", "src/stanstock/research/service.py"),
)
BASE_CONFIG_PATHS = (
    "config/scoring/us-price-baseline-v1.yml",
    "config/scoring/us-price-baseline-v2.yml",
)
FIXED_COLLABORATOR_PATHS = (
    "src/stanstock/research/scenarios.py",
    "src/stanstock/research/explanations.py",
    "src/stanstock/research/types.py",
    "src/stanstock/data/asof.py",
    "pyproject.toml",
    "uv.lock",
)


class BaseRevisionUnavailableError(RuntimeError):
    """The exact d637 object needed for frozen evidence is unavailable."""


@dataclass(frozen=True, slots=True)
class BaseShortModules:
    base_sha: str
    source_sha256: dict[str, str]
    config: ModuleType
    indicators: ModuleType
    scoring: ModuleType
    service: ModuleType


def read_base_bytes(path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{BASE_SHA}:{path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise BaseRevisionUnavailableError(
            f"Base revision object {BASE_SHA}:{path} is not readable"
        ) from error


def base_sources_available() -> bool:
    try:
        base_source_checksums()
    except BaseRevisionUnavailableError:
        return False
    return True


def base_source_checksums() -> dict[str, str]:
    paths = [
        *(path for _name, path in BASE_MODULE_PATHS),
        *BASE_CONFIG_PATHS,
        *FIXED_COLLABORATOR_PATHS,
    ]
    return {path: hashlib.sha256(read_base_bytes(path)).hexdigest() for path in paths}


def write_base_config(path: Path, repository_path: str) -> None:
    """Write one exact base config to a pytest-owned temporary path."""
    path.write_bytes(read_base_bytes(repository_path))


@contextmanager
def base_short_modules() -> Iterator[BaseShortModules]:
    """Temporarily bind every frozen module and restore every prior binding."""
    originals = {name: sys.modules.get(name) for name, _path in BASE_MODULE_PATHS}
    loaded: dict[str, ModuleType] = {}
    checksums: dict[str, str] = {}
    try:
        for name, path in BASE_MODULE_PATHS:
            source = read_base_bytes(path)
            checksums[path] = hashlib.sha256(source).hexdigest()
            spec = importlib.util.spec_from_loader(name, loader=None, origin=f"{BASE_SHA}:{path}")
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(REPO_ROOT / path)
            sys.modules[name] = module
            exec(compile(source, f"<{BASE_SHA}:{path}>", "exec"), module.__dict__)
            loaded[name] = module
        yield BaseShortModules(
            base_sha=BASE_SHA,
            source_sha256=checksums,
            config=loaded["stanstock.research.config"],
            indicators=loaded["stanstock.research.indicators"],
            scoring=loaded["stanstock.research.scoring"],
            service=loaded["stanstock.research.service"],
        )
    finally:
        for name, _path in BASE_MODULE_PATHS:
            original = originals[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
