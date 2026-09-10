from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "stanstock"


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


@pytest.mark.parametrize("domain", ["research", "simulation"])
def test_domain_logic_does_not_import_raw_provider_modules(domain: str) -> None:
    violations: list[str] = []
    for path in (SOURCE_ROOT / domain).rglob("*.py"):
        imported_modules = _imported_modules(path)
        if any(
            module == "stanstock.data.providers" or module.startswith("stanstock.data.providers.")
            for module in imported_modules
        ):
            violations.append(str(path.relative_to(SOURCE_ROOT)))

    assert violations == []


def test_verification_types_is_a_zero_domain_dependency_leaf() -> None:
    """`core.verification_types` must never import Django, a `stanstock`
    domain package, Polars, or `decimal` -- it is the verification graph's
    leaf module every domain validator and orchestrator build on."""
    modules = _imported_modules(SOURCE_ROOT / "core" / "verification_types.py")
    forbidden = (
        "django",
        "polars",
        "decimal",
        "stanstock.data",
        "stanstock.research",
        "stanstock.portfolio",
    )
    assert not [m for m in modules if any(m == f or m.startswith(f + ".") for f in forbidden)]


def test_refresh_validation_does_not_import_heavy_research_or_live_us_modules() -> None:
    """`research.refresh_validation` must stay safe for `core` to import in
    a later slice: it must never import `data.live_us` (the US-only daily
    job orchestrator) or `research.service` (the full analysis/prediction
    writer), which would create a heavy or cyclical dependency. It uses the
    cycle-safe `research.timing` leaf and `research.refresh_evidence`'s own
    pure helpers instead."""
    modules = _imported_modules(SOURCE_ROOT / "research" / "refresh_validation.py")
    forbidden = ("stanstock.data.live_us", "stanstock.research.service")
    assert not [m for m in modules if any(m == f or m.startswith(f + ".") for f in forbidden)]
