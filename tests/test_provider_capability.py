"""Split/reverse-split capability refusal.

`split_event_capability` is the only place the codebase answers "can a
verified split event be obtained?". No reviewed corporate-actions source is
integrated for any provider, so the answer is always no -- and the refusal
must say *why* without inventing an entitlement, a source, or an inference
path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from stanstock.data.provider_policy import (
    CAPABILITY_NO_REVIEWED_SOURCE,
    CAPABILITY_PLAN_NOT_ENTITLED,
    CAPABILITY_UNAVAILABLE,
    SPLIT_EVENT_CAPABILITY,
    normalized_provider_plan,
    split_event_capability,
)

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "stanstock"


@pytest.mark.parametrize("plan", ["basic", "Basic", "BASIC", "  basic  "])
def test_recorded_twelve_data_basic_plan_is_not_entitled(plan: str) -> None:
    assert split_event_capability("twelve_data", plan) == (
        CAPABILITY_UNAVAILABLE,
        CAPABILITY_PLAN_NOT_ENTITLED,
    )


@pytest.mark.parametrize(
    ("provider", "plan"),
    [
        ("twelve_data", None),
        ("twelve_data", ""),
        ("twelve_data", "   "),
        ("twelve_data", "basic-plus"),
        ("twelve_data", "pro"),
        ("twelve_data", "ultra"),
        ("twelve_data", "grow"),
        ("twelve_data", "custom"),
        ("twelve_data", "{'plan': 'basic'}"),
        ("sec", "basic"),
        ("synthetic_demo", "basic"),
        ("synthetic_demo", None),
        ("stooq", None),
        ("", None),
    ],
)
def test_every_other_provider_or_plan_has_no_reviewed_source(
    provider: str,
    plan: str | None,
) -> None:
    assert split_event_capability(provider, plan) == (
        CAPABILITY_UNAVAILABLE,
        CAPABILITY_NO_REVIEWED_SOURCE,
    )


def test_no_branch_can_return_an_available_or_verified_state() -> None:
    providers = ("twelve_data", "sec", "synthetic_demo", "stooq", "unknown", "")
    plans = (None, "", "basic", "pro", "grow", "ultra", "custom", "enterprise", "???")

    statuses = {
        split_event_capability(provider, plan)[0] for provider in providers for plan in plans
    }
    reasons = {
        split_event_capability(provider, plan)[1] for provider in providers for plan in plans
    }

    assert statuses == {CAPABILITY_UNAVAILABLE}
    assert reasons == {CAPABILITY_PLAN_NOT_ENTITLED, CAPABILITY_NO_REVIEWED_SOURCE}
    assert "verified" not in statuses
    assert "available" not in statuses


def test_capability_name_and_plan_normalization_are_explicit() -> None:
    assert SPLIT_EVENT_CAPABILITY == "corporate_actions_splits"
    assert normalized_provider_plan(None) is None
    assert normalized_provider_plan("") is None
    assert normalized_provider_plan("   ") is None
    assert normalized_provider_plan(" Basic ") == "basic"
    assert normalized_provider_plan("PRO") == "pro"


@pytest.mark.django_db
def test_capability_helper_performs_no_database_query() -> None:
    with CaptureQueriesContext(connection) as captured:
        split_event_capability("twelve_data", "basic")
        split_event_capability("twelve_data", None)
        split_event_capability("sec", "pro")

    assert list(captured.captured_queries) == []


def test_capability_helper_makes_no_network_or_orm_call_in_its_source() -> None:
    module = ast.parse(
        (SOURCE_ROOT / "data" / "provider_policy.py").read_text(encoding="utf-8"),
        filename="provider_policy.py",
    )
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "split_event_capability"
    )
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }

    assert called <= {"strip", "lower", "normalized_provider_plan"}
    assert "objects" not in ast.dump(function)
    assert "fetch" not in ast.dump(function)


def test_no_split_event_model_or_migration_exists() -> None:
    sources = list(SOURCE_ROOT.rglob("*.py"))
    model_sources = [path for path in sources if path.name == "models.py"]
    migration_sources = [path for path in sources if path.parent.name == "migrations"]

    for path in (*model_sources, *migration_sources):
        text = path.read_text(encoding="utf-8")
        assert "SplitEvent" not in text
        assert "splitevent" not in text
        assert "CorporateAction" not in text
        assert "corporate_actions_splits" not in text
