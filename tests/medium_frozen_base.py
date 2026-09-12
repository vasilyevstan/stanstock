"""Load frozen medium-v1 code exclusively from the sealed base revision.

The loader installs the changed modules and their directly used first-party
runtime dependencies under their real names, in dependency order. All
source/config bytes come from Git objects; there is deliberately no
working-tree fallback.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

BASE_SHA = "d0e08a720c383a65ff6d7fa99bf8ac1be03f5155"
BASE_TREE = "74a758ec9ebac8ad2d88628cd1b950e30cf3de35"
REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_MODULE_PATHS: tuple[tuple[str, str], ...] = (
    ("stanstock.research.forecast_config", "src/stanstock/research/forecast_config.py"),
    ("stanstock.research.medium_forecasts", "src/stanstock/research/medium_forecasts.py"),
    ("stanstock.research.service", "src/stanstock/research/service.py"),
    ("stanstock.web.templatetags.stanstock", "src/stanstock/web/templatetags/stanstock.py"),
)
# Django has already registered these model classes before this test harness
# runs. Re-executing either module would create conflicting ORM identities, so
# they remain the fixed schema/fixture boundary. Every other first-party module
# imported directly by the four frozen modules is executed from base bytes.
BASE_ORM_IDENTITY_PATHS = {
    "src/stanstock/data/models.py",
    "src/stanstock/research/models.py",
}
BASE_RUNTIME_MODULE_PATHS: tuple[tuple[str, str], ...] = (
    ("stanstock.core.revision", "src/stanstock/core/revision.py"),
    ("stanstock.core.verification_types", "src/stanstock/core/verification_types.py"),
    ("stanstock.data.provider_policy", "src/stanstock/data/provider_policy.py"),
    ("stanstock.data.sec_config", "src/stanstock/data/sec_config.py"),
    ("stanstock.research.affordability", "src/stanstock/research/affordability.py"),
    ("stanstock.research.config", "src/stanstock/research/config.py"),
    ("stanstock.research.eligibility", "src/stanstock/research/eligibility.py"),
    ("stanstock.research.forecast_config", "src/stanstock/research/forecast_config.py"),
    ("stanstock.research.forecasting", "src/stanstock/research/forecasting.py"),
    ("stanstock.research.long_forecast_config", "src/stanstock/research/long_forecast_config.py"),
    ("stanstock.research.provenance", "src/stanstock/research/provenance.py"),
    ("stanstock.research.refresh_evidence", "src/stanstock/research/refresh_evidence.py"),
    ("stanstock.research.timing", "src/stanstock/research/timing.py"),
    ("stanstock.research.types", "src/stanstock/research/types.py"),
    ("stanstock.data.assets", "src/stanstock/data/assets.py"),
    ("stanstock.research.explanations", "src/stanstock/research/explanations.py"),
    ("stanstock.research.fundamentals", "src/stanstock/research/fundamentals.py"),
    ("stanstock.research.indicators", "src/stanstock/research/indicators.py"),
    ("stanstock.research.scenarios", "src/stanstock/research/scenarios.py"),
    ("stanstock.data.asof", "src/stanstock/data/asof.py"),
    ("stanstock.data.etfs", "src/stanstock/data/etfs.py"),
    ("stanstock.research.scoring", "src/stanstock/research/scoring.py"),
    ("stanstock.research.under10", "src/stanstock/research/under10.py"),
    ("stanstock.research.long_forecasts", "src/stanstock/research/long_forecasts.py"),
    ("stanstock.research.medium_forecasts", "src/stanstock/research/medium_forecasts.py"),
    ("stanstock.research.service", "src/stanstock/research/service.py"),
    ("stanstock.web.templatetags.stanstock", "src/stanstock/web/templatetags/stanstock.py"),
)
BASE_CONFIG_PATHS = (
    "config/forecasts/us-price-medium-v1.yml",
    "config/scoring/us-price-baseline-v2.yml",
    "config/forecasts/us-sec-long-v2.yml",
)
NON_MODULE_COLLABORATORS = (
    "pyproject.toml",
    "uv.lock",
    "templates/base.html",
)


class BaseRevisionUnavailableError(RuntimeError):
    """The exact sealed base object needed for frozen evidence is unavailable."""


@dataclass(frozen=True, slots=True)
class BaseMediumModules:
    base_sha: str
    base_tree: str
    source_sha256: dict[str, str]
    forecast_config: ModuleType
    medium_forecasts: ModuleType
    service: ModuleType
    template_tags: ModuleType


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


def _base_path_exists(path: str) -> bool:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{BASE_SHA}:{path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _module_path(module: str) -> str | None:
    if not module.startswith("stanstock."):
        return None
    stem = f"src/{module.replace('.', '/')}"
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if _base_path_exists(candidate):
            return candidate
    return None


def imported_repository_dependencies() -> tuple[str, ...]:
    """Mechanically enumerate local imports of every changed base module."""
    dependencies: set[str] = set()
    for _name, path in BASE_MODULE_PATHS:
        tree = ast.parse(read_base_bytes(path))
        for node in ast.walk(tree):
            module_names: list[str] = []
            if isinstance(node, ast.Import):
                module_names.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names.append(node.module)
            for module_name in module_names:
                dependency = _module_path(module_name)
                if dependency is not None:
                    dependencies.add(dependency)
    changed_paths = {path for _name, path in BASE_MODULE_PATHS}
    return tuple(sorted(dependencies - changed_paths))


def base_source_checksums() -> dict[str, str]:
    paths = (
        *(path for _name, path in BASE_MODULE_PATHS),
        *BASE_CONFIG_PATHS,
        *imported_repository_dependencies(),
        *NON_MODULE_COLLABORATORS,
    )
    return {
        path: hashlib.sha256(read_base_bytes(path)).hexdigest() for path in dict.fromkeys(paths)
    }


def base_sources_available() -> bool:
    try:
        base_source_checksums()
    except BaseRevisionUnavailableError:
        return False
    return True


def write_base_config(path: Path, repository_path: str) -> None:
    path.write_bytes(read_base_bytes(repository_path))


@contextmanager
def base_medium_modules() -> Iterator[BaseMediumModules]:
    expected_runtime_paths = {
        *(path for _name, path in BASE_MODULE_PATHS),
        *imported_repository_dependencies(),
    } - BASE_ORM_IDENTITY_PATHS
    if {path for _name, path in BASE_RUNTIME_MODULE_PATHS} != expected_runtime_paths:
        raise AssertionError("Sealed medium runtime dependency bindings are incomplete")
    originals = {name: sys.modules.get(name) for name, _path in BASE_RUNTIME_MODULE_PATHS}
    fixed_models = sys.modules.get("stanstock.research.models")
    if fixed_models is None:
        raise AssertionError("Django research model identities must be loaded before sealing")
    original_scenario_from_document = fixed_models.scenario_from_document
    loaded: dict[str, ModuleType] = {}
    checksums: dict[str, str] = {}
    try:
        for name, path in BASE_RUNTIME_MODULE_PATHS:
            source = read_base_bytes(path)
            checksums[path] = hashlib.sha256(source).hexdigest()
            spec = importlib.util.spec_from_loader(name, loader=None, origin=f"{BASE_SHA}:{path}")
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(REPO_ROOT / path)
            sys.modules[name] = module
            exec(compile(source, f"<{BASE_SHA}:{path}>", "exec"), module.__dict__)
            loaded[name] = module
        fixed_models.scenario_from_document = loaded[
            "stanstock.research.forecasting"
        ].scenario_from_document
        yield BaseMediumModules(
            base_sha=BASE_SHA,
            base_tree=BASE_TREE,
            source_sha256=checksums,
            forecast_config=loaded["stanstock.research.forecast_config"],
            medium_forecasts=loaded["stanstock.research.medium_forecasts"],
            service=loaded["stanstock.research.service"],
            template_tags=loaded["stanstock.web.templatetags.stanstock"],
        )
    finally:
        fixed_models.scenario_from_document = original_scenario_from_document
        for name, _path in reversed(BASE_RUNTIME_MODULE_PATHS):
            original = originals[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
