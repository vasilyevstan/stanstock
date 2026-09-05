from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "stanstock"


@pytest.mark.parametrize("domain", ["research", "simulation"])
def test_domain_logic_does_not_import_raw_provider_modules(domain: str) -> None:
    violations: list[str] = []
    for path in (SOURCE_ROOT / domain).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            if any(
                module == "stanstock.data.providers"
                or module.startswith("stanstock.data.providers.")
                for module in imported_modules
            ):
                violations.append(str(path.relative_to(SOURCE_ROOT)))

    assert violations == []
