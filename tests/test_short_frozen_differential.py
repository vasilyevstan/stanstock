"""Frozen d637 execution evidence for short price-baseline v1 and v2."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid5

import polars as pl
import pytest

from short_frozen_base import (
    BASE_CONFIG_PATHS,
    BASE_MODULE_PATHS,
    BASE_SHA,
    FIXED_COLLABORATOR_PATHS,
    BaseRevisionUnavailableError,
    base_short_modules,
    base_source_checksums,
    base_sources_available,
    write_base_config,
)
from stanstock.data.assets import AssetStore
from stanstock.data.models import (
    Company,
    DataAsset,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.research.models import AnalysisRun, Prediction
from stanstock.research.service import analyze_listing

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "short_v1_v2_frozen_base_payloads.json"
WRITE_GOLDEN_ENV = "STANSTOCK_WRITE_SHORT_FROZEN_GOLDEN"
DECISION_TIME = datetime(2026, 8, 31, 20, tzinfo=UTC)
TARGET_DATE = DECISION_TIME.date()
NAMESPACE = UUID("ef09f640-fd30-480f-9e2d-73686f727476")
RUN_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def _det(*parts: str) -> UUID:
    return uuid5(NAMESPACE, "/".join(parts))


def _frame(kind: str) -> pl.DataFrame:
    case = kind.rsplit("-", 1)[-1]
    rows = {"success": 400, "avoid": 2, "hold": 130}[case]
    dates = [TARGET_DATE - timedelta(days=rows - index - 1) for index in range(rows)]
    if case == "avoid":
        closes = [20.0, 40.0]
    else:
        closes = [40.0 + 0.08 * index + 0.35 * ((index % 11) - 5) / 5 for index in range(rows)]
    return pl.DataFrame(
        {
            "date": dates,
            "open": [close - 0.1 for close in closes],
            "high": [close + 0.4 for close in closes],
            "low": [close - 0.5 for close in closes],
            "close": closes,
            "volume": [2_000_000 + 10_000 * (index % 7) for index in range(rows)],
        }
    )


def _benchmark(kind: str) -> pl.DataFrame:
    frame = _frame(kind)
    return frame.with_columns(
        (35.0 + pl.int_range(pl.len()) * 0.045 + (pl.int_range(pl.len()) % 13) * 0.02)
        .cast(pl.Float64)
        .alias("close")
    )


def _fixture(store: AssetStore, kind: str) -> tuple[Listing, UniverseSnapshot, str]:
    company = Company.objects.create(id=_det("company", kind), name=f"{kind} Co", country="US")
    security = Security.objects.create(
        id=_det("security", kind), company=company, name=f"{kind} Common"
    )
    listing = Listing.objects.create(
        id=_det("listing", kind),
        security=security,
        ticker=f"F{kind[:3].upper()}",
        provider_symbol=f"F{kind[:3].upper()}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug=f"short-frozen-{kind}",
        name=f"Short frozen {kind}",
        config_version="short-frozen-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        id=_det("snapshot", kind),
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="1" * 64,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing, eligible=True)
    benchmark_subject = f"B{kind.replace('-', '').upper()}"
    for subject, frame in (
        (listing.ticker, _frame(kind)),
        (benchmark_subject, _benchmark(kind)),
    ):
        stored = store.write_frame(f"short-frozen/{kind}/{subject}.parquet", frame)
        DataAsset.objects.create(
            id=_det("asset", kind, subject),
            provider="synthetic_demo",
            kind="price_history",
            subject=subject,
            relative_path=stored.relative_path,
            sha256=stored.sha256,
            retrieved_at=DECISION_TIME,
            available_at=DECISION_TIME,
            period_start=frame["date"][0],
            period_end=frame["date"][-1],
            metadata={
                "return_definition": "split_adjusted_price_return",
                "dividends_included": False,
            },
        )
    return listing, snapshot, benchmark_subject


def _normalize(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize(item, f"{path}.{key}" if path else key) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize(item, f"{path}[]") for item in value]
    if isinstance(value, tuple):
        return [_normalize(item, f"{path}[]") for item in value]
    if isinstance(value, UUID):
        return "<uuid>"
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if path.endswith(("model_version", "prediction_version")) and isinstance(value, str):
        return RUN_SUFFIX.sub("-<run>", value)
    return value


def _capture(run: AnalysisRun) -> dict[str, Any]:
    analysis = run.stocks.get()
    predictions = list(Prediction.objects.filter(analysis=analysis).order_by("horizon"))
    return _normalize(
        {
            "run": {
                "generated_at": run.generated_at,
                "data_cutoff": run.data_cutoff,
                "target_date": run.target_date,
                "issued_on_time": run.issued_on_time,
                "config_version": run.config_version,
                "config_hash": run.config_hash,
                "code_revision": run.code_revision,
                "status": run.status,
            },
            "analysis": {
                field: getattr(analysis, field)
                for field in (
                    "current_price",
                    "daily_change",
                    "overall_score",
                    "recommendation",
                    "risk_score",
                    "risk_class",
                    "confidence",
                    "confidence_status",
                    "component_scores",
                    "forecast_scenarios",
                    "short_scenario",
                    "medium_scenario",
                    "long_scenario",
                    "reasons",
                    "risks",
                    "data_quality",
                )
            },
            "predictions": [
                {
                    field: getattr(prediction, field)
                    for field in (
                        "generated_at",
                        "target_date",
                        "issued_on_time",
                        "horizon",
                        "evidence_role",
                        "evidence_grade",
                        "source_mode",
                        "price_provider",
                        "price_subject",
                        "price_at_prediction",
                        "bear_return",
                        "base_return",
                        "bull_return",
                        "probability_positive",
                        "confidence",
                        "confidence_status",
                        "insufficiency_reason",
                        "recommendation",
                        "overall_score",
                        "component_scores",
                        "model_version",
                        "method_version",
                        "config_hash",
                        "data_cutoff",
                        "source_assets",
                        "calculation",
                        "code_revision",
                    )
                }
                for prediction in predictions
            ],
        }
    )


def _run_payloads(
    tmp_path: Path,
    *,
    analyze: Any,
    load_config: Any,
) -> dict[str, Any]:
    """Execute with one explicit revision, independent of the outer suite."""
    environment_key = "STANSTOCK_CODE_REVISION"
    previous_revision = os.environ.get(environment_key)
    os.environ[environment_key] = "f" * 40
    try:
        return _run_payloads_at_fixed_revision(
            tmp_path,
            analyze=analyze,
            load_config=load_config,
        )
    finally:
        if previous_revision is None:
            os.environ.pop(environment_key, None)
        else:
            os.environ[environment_key] = previous_revision


def _run_payloads_at_fixed_revision(
    tmp_path: Path,
    *,
    analyze: Any,
    load_config: Any,
) -> dict[str, Any]:
    store = AssetStore(tmp_path)
    payloads: dict[str, Any] = {}
    for config_repo_path in BASE_CONFIG_PATHS:
        label = Path(config_repo_path).stem.rsplit("-", 1)[-1]
        config_path = tmp_path / f"{label}.yml"
        write_base_config(config_path, config_repo_path)
        config = load_config(config_path)
        cases: dict[str, Any] = {}
        for kind in ("success", "avoid", "hold"):
            listing, snapshot, benchmark_subject = _fixture(store, f"{label}-{kind}")
            persisted = analyze(
                listing=listing,
                universe_snapshot=snapshot,
                decision_time=DECISION_TIME,
                target_date=TARGET_DATE,
                provider="synthetic_demo",
                benchmark_subject=benchmark_subject,
                store=store,
                config_path=config_path,
                sample_support={"short": 200},
            )
            cases[kind] = _capture(persisted.run)
        malformed_path = tmp_path / f"{label}-malformed.yml"
        malformed_path.write_text("version: broken\n", encoding="utf-8")
        try:
            load_config(malformed_path)
        except Exception as error:  # exact historical exception is payload evidence
            malformed = {"type": type(error).__name__, "message": str(error)}
        else:  # pragma: no cover - a malformed config must never succeed
            malformed = {"type": "<none>", "message": ""}
        payloads[label] = {
            "effective_hash": hashlib.sha256(
                json.dumps(
                    config.raw,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode()
            ).hexdigest(),
            "cases": cases,
            "malformed_config": malformed,
        }
    payloads["outcome_truth_table"] = {
        "buy": {"negative": False, "zero": False, "positive": True},
        "avoid": {"negative": True, "zero": True, "positive": False},
        "hold": {"below": False, "bear": True, "inside": True, "bull": True, "above": False},
    }
    return payloads


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _first_difference(left: Any, right: Any, path: str = "payloads") -> str:
    if type(left) is not type(right):
        return f"{path}: types differ ({type(left).__name__} != {type(right).__name__})"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys differ ({sorted(left)} != {sorted(right)})"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths differ ({len(left)} != {len(right)})"
        for index, item in enumerate(left):
            difference = _first_difference(item, right[index], f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if left == right else f"{path}: {left!r} != {right!r}"


def _assert_matches_golden(payloads: dict[str, Any]) -> None:
    golden = _golden()["payloads"]
    assert payloads == golden, _first_difference(payloads, golden)


@pytest.mark.django_db
def test_frozen_short_v1_v2_payloads_match_committed_base_golden(tmp_path: Path) -> None:
    payloads = _run_payloads(
        tmp_path,
        analyze=analyze_listing,
        load_config=importlib.import_module("stanstock.research.config").load_scoring_config,
    )
    _assert_matches_golden(payloads)


def test_committed_short_golden_pins_exact_base_objects() -> None:
    golden = _golden()
    assert golden["base_sha"] == BASE_SHA
    assert golden["generated_from"] == "base_revision_execution"
    assert golden["normalized_paths"] == [
        "run.id (not captured)",
        "analysis.id (not captured)",
        "prediction.id (not captured)",
        "predictions[].model_version UUID suffix",
        "predictions[].calculation.prediction_version UUID suffix",
    ]
    assert golden["source_sha256"] == {
        "src/stanstock/research/config.py": (
            "6c7328b71450e9c302fa400c4d087356ac1e3b805182c72f8155346463c1ce12"
        ),
        "src/stanstock/research/indicators.py": (
            "bb0efc599155481b6d1eb42d8624e6f1380131cbbcb42e5e33d5493ed699d406"
        ),
        "src/stanstock/research/scoring.py": (
            "ac6696d0e50f006496b5a23771326c1608592f2e45d3f4386c6a16265b8c7a07"
        ),
        "src/stanstock/research/service.py": (
            "bad15874181f40c08a8bd86438b6d3345028987b11e804d5c1d0d0b481cb87c3"
        ),
    }


@pytest.mark.skipif(not base_sources_available(), reason="d637 base objects unavailable")
def test_committed_short_golden_hashes_match_local_base_objects() -> None:
    golden = _golden()
    checksums = base_source_checksums()
    assert golden["all_base_sha256"] == checksums


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="d637 base objects unavailable")
def test_frozen_short_payloads_match_live_base_execution(tmp_path: Path) -> None:
    with base_short_modules() as base:
        payloads = _run_payloads(
            tmp_path,
            analyze=base.service.analyze_listing,
            load_config=base.config.load_scoring_config,
        )
    _assert_matches_golden(payloads)


@pytest.mark.django_db
def test_short_golden_regeneration_refuses_without_base_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import short_frozen_base

    monkeypatch.setattr(
        short_frozen_base,
        "read_base_bytes",
        lambda path: (_ for _ in ()).throw(BaseRevisionUnavailableError(path)),
    )
    target = tmp_path / "golden.json"
    with pytest.raises(BaseRevisionUnavailableError):
        regenerate_golden(target, tmp_path / "work")
    assert not target.exists()


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="d637 base objects unavailable")
def test_head_output_mutation_fails_complete_golden_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import stanstock.research.service as live_service

    original = live_service.compute_listing_analysis

    def mutate(**kwargs: Any) -> Any:
        result = original(**kwargs)
        result.data_quality["frozen_negative_control"] = True
        return result

    monkeypatch.setattr(live_service, "compute_listing_analysis", mutate)
    payloads = _run_payloads(
        tmp_path,
        analyze=live_service.analyze_listing,
        load_config=importlib.import_module("stanstock.research.config").load_scoring_config,
    )
    assert payloads != _golden()["payloads"]


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="d637 base objects unavailable")
def test_head_service_monkeypatch_cannot_contaminate_base_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import stanstock.research.service as live_service

    monkeypatch.setattr(
        live_service,
        "analyze_listing",
        lambda **_kwargs: pytest.fail("base regeneration executed head service"),
    )
    target = tmp_path / "regenerated.json"

    regenerate_golden(target, tmp_path / "work")

    assert target.read_bytes() == GOLDEN_PATH.read_bytes()


def test_frozen_outcome_truth_table_matches_current_unchanged_semantics() -> None:
    from stanstock.research.models import Recommendation
    from stanstock.research.outcomes import _success

    def prediction(recommendation: str) -> SimpleNamespace:
        return SimpleNamespace(
            recommendation=recommendation,
            bear_return=-0.1,
            bull_return=0.1,
        )

    actual = {
        "buy": {
            "negative": _success(prediction(Recommendation.BUY), -0.01),
            "zero": _success(prediction(Recommendation.BUY), 0.0),
            "positive": _success(prediction(Recommendation.BUY), 0.01),
        },
        "avoid": {
            "negative": _success(prediction(Recommendation.AVOID), -0.01),
            "zero": _success(prediction(Recommendation.AVOID), 0.0),
            "positive": _success(prediction(Recommendation.AVOID), 0.01),
        },
        "hold": {
            "below": _success(prediction(Recommendation.HOLD), -0.11),
            "bear": _success(prediction(Recommendation.HOLD), -0.1),
            "inside": _success(prediction(Recommendation.HOLD), 0.0),
            "bull": _success(prediction(Recommendation.HOLD), 0.1),
            "above": _success(prediction(Recommendation.HOLD), 0.11),
        },
    }

    assert actual == _golden()["payloads"]["outcome_truth_table"]


@pytest.mark.django_db
@pytest.mark.skipif(
    os.environ.get(WRITE_GOLDEN_ENV) != "1",
    reason=f"set {WRITE_GOLDEN_ENV}=1 to regenerate",
)
def test_regenerate_short_golden(tmp_path: Path) -> None:  # pragma: no cover
    regenerate_golden(GOLDEN_PATH, tmp_path)


def regenerate_golden(path: Path, work_path: Path) -> dict[str, Any]:
    checksums = base_source_checksums()
    work_path.mkdir(parents=True, exist_ok=True)
    with base_short_modules() as base:
        payloads = _run_payloads(
            work_path,
            analyze=base.service.analyze_listing,
            load_config=base.config.load_scoring_config,
        )
        assert base.source_sha256 == {
            source_path: checksums[source_path] for _name, source_path in BASE_MODULE_PATHS
        }
    document = {
        "base_sha": BASE_SHA,
        "generated_by": "tests/test_short_frozen_differential.py",
        "generated_from": "base_revision_execution",
        "source_sha256": {
            source_path: checksums[source_path] for _name, source_path in BASE_MODULE_PATHS
        },
        "config_byte_sha256": {path: checksums[path] for path in BASE_CONFIG_PATHS},
        "fixed_collaborator_sha256": {path: checksums[path] for path in FIXED_COLLABORATOR_PATHS},
        "all_base_sha256": checksums,
        "normalized_paths": [
            "run.id (not captured)",
            "analysis.id (not captured)",
            "prediction.id (not captured)",
            "predictions[].model_version UUID suffix",
            "predictions[].calculation.prediction_version UUID suffix",
        ],
        "payloads": payloads,
    }
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), default=str) + "\n",
        encoding="utf-8",
    )
    return document
