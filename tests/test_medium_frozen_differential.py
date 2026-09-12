"""Complete base-object differential for frozen ``us-price-medium-v1``."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import os
import re
import sys
import zlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import polars as pl
import pytest
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from medium_frozen_base import (
    BASE_CONFIG_PATHS,
    BASE_MODULE_PATHS,
    BASE_RUNTIME_MODULE_PATHS,
    BASE_SHA,
    BASE_TREE,
    NON_MODULE_COLLABORATORS,
    BaseRevisionUnavailableError,
    base_medium_modules,
    base_source_checksums,
    base_sources_available,
    imported_repository_dependencies,
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
from stanstock.research.models import Prediction

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "medium_v1_frozen_base_payloads.json"
WRITE_GOLDEN_ENV = "STANSTOCK_WRITE_MEDIUM_FROZEN_GOLDEN"
TARGET_DATE = date(2026, 9, 4)
GENERATED_AT = datetime(2026, 9, 5, 1, tzinfo=UTC)
NAMESPACE = UUID("e384d298-2933-4b90-bf12-6d656469756d")
UUID_TEXT = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _det(*parts: str) -> UUID:
    return uuid5(NAMESPACE, "/".join(parts))


@dataclass(frozen=True, slots=True)
class _IdentityAliases:
    by_uuid: Mapping[str, str]
    run_aliases: Mapping[str, str]


EMPTY_ALIASES = _IdentityAliases(by_uuid={}, run_aliases={})


def _normalize(
    value: Any,
    aliases: _IdentityAliases = EMPTY_ALIASES,
    path: str = "",
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize(
                item,
                aliases,
                f"{path}.{key}" if path else str(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, aliases, f"{path}[]") for item in value]
    if isinstance(value, UUID):
        return _identity_alias(str(value), aliases)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Frozen capture cannot normalize a non-finite float")
        # Numerical libraries can expose platform-specific binary tails even
        # when the meaningful result is identical. Fifteen significant decimal
        # digits are stable across those float round trips while retaining
        # materially different captured values.
        normalized = float(format(value, ".15g"))
        return 0.0 if normalized == 0.0 else normalized
    if isinstance(value, str):
        normalized = UUID_TEXT.sub(
            lambda match: _identity_alias(match.group(0), aliases),
            value,
        )
        for run_id, alias in aliases.run_aliases.items():
            run_hex = run_id.replace("-", "")
            normalized = normalized.replace(run_hex, alias)
            if path.endswith(("model_version", "prediction_version")):
                normalized = re.sub(
                    rf"-{re.escape(run_hex[:8])}$",
                    f"-{alias}",
                    normalized,
                )
        for filename in ("not_mapping.yml", "schema.yml", "broken_yaml.yml"):
            if filename in normalized:
                normalized = re.sub(r"\S*/" + re.escape(filename), f"<path>/{filename}", normalized)
        return normalized
    return value


def _identity_alias(value: str, aliases: _IdentityAliases) -> str:
    try:
        return aliases.by_uuid[value]
    except KeyError:
        raise AssertionError(f"Unregistered UUID in frozen evidence: {value}") from None


def _semantic_identity_aliases(
    *,
    result: Any,
    snapshot: UniverseSnapshot,
    listings: list[Listing],
    predictions: list[Prediction],
    assets: list[DataAsset],
) -> _IdentityAliases:
    """Derive stable aliases from authoritative fixture/model identities."""

    run = result.run
    run_label = f"<run:{snapshot.universe.slug}:{run.target_date.isoformat()}:{run.config_version}>"
    by_uuid: dict[str, str] = {
        str(run.id): run_label,
        str(snapshot.id): (
            f"<snapshot:{snapshot.universe.slug}:{snapshot.as_of_date.isoformat()}:"
            f"{snapshot.grade}>"
        ),
    }
    for listing in listings:
        by_uuid[str(listing.id)] = (
            f"<listing:{listing.exchange_mic}:{listing.provider_symbol or listing.ticker}>"
        )
    for asset in assets:
        if (
            asset.provider == "stanstock"
            and asset.kind == "medium_forecast_panel"
            and asset.subject == str(run.id)
        ):
            alias = (
                f"<panel-asset:{snapshot.universe.slug}:"
                f"{run.target_date.isoformat()}:{run.config_version}>"
            )
        else:
            alias = f"<source-asset:{asset.provider}:{asset.kind}:{asset.subject}>"
        if alias in by_uuid.values():
            raise AssertionError(f"Frozen fixture has duplicate semantic asset identity: {alias}")
        by_uuid[str(asset.id)] = alias
    for prediction in predictions:
        listing_alias = by_uuid[str(prediction.listing_id)].removeprefix("<").removesuffix(">")
        alias = (
            f"<prediction:{listing_alias}:{prediction.horizon}:"
            f"{prediction.evidence_role}:{prediction.method_version}>"
        )
        if alias in by_uuid.values():
            raise AssertionError(
                f"Frozen fixture has duplicate semantic prediction identity: {alias}"
            )
        by_uuid[str(prediction.id)] = alias
    return _IdentityAliases(
        by_uuid=by_uuid,
        run_aliases={str(run.id): run_label},
    )


def _sessions(count: int) -> list[date]:
    calendar = get_calendar("XNYS")
    target = calendar.date_to_session(TARGET_DATE)
    first = calendar.session_offset(target, -(count - 1))
    return [session.date() for session in calendar.sessions_in_range(first, target)]


def _listing(ticker: str) -> Listing:
    company = Company.objects.create(
        id=_det("company", ticker),
        name=f"{ticker} Company",
        country="US",
    )
    security = Security.objects.create(
        id=_det("security", ticker),
        company=company,
        name=f"{ticker} Common",
    )
    return Listing.objects.create(
        id=_det("listing", ticker),
        security=security,
        ticker=ticker,
        exchange_mic="XNAS",
        provider_symbol=ticker,
        currency="USD",
        region=Region.US,
    )


def _asset(
    store: AssetStore,
    *,
    subject: str,
    sessions: list[date],
    slope: float,
) -> DataAsset:
    closes = [40.0 + index * slope + ((index % 17) - 8) * 0.04 for index in range(len(sessions))]
    frame = pl.DataFrame(
        {
            "date": sessions,
            "close": closes,
            "volume": [2_000_000 + index for index in range(len(sessions))],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    path = f"medium-frozen/{subject}.parquet"
    stored = store.write_frame(path, frame)
    return DataAsset.objects.create(
        id=_det("asset", subject),
        provider="synthetic",
        kind="price_history",
        subject=subject,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=GENERATED_AT,
        available_at=GENERATED_AT,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )


def _panel_record(
    *,
    listing_id: str,
    anchor: date,
    value: float | None,
    current: bool = False,
    state: int = 1,
) -> dict[str, object]:
    return {
        "horizon": "6m",
        "anchor_date": anchor,
        "label_end_date": None if current else anchor,
        "is_forecast": current,
        "cohort_id": f"6m:{anchor.isoformat()}",
        "listing_id": listing_id,
        "ticker": listing_id,
        "price_asset_id": listing_id,
        "relative_momentum": 0.1,
        "drawdown": -0.1,
        "volatility": 0.2,
        "market_trend": 0.05,
        "market_volatility": 0.2,
        "relative_momentum_bucket": state,
        "drawdown_bucket": state,
        "volatility_bucket": state,
        "market_trend_bucket": state,
        "market_volatility_bucket": state,
        "close_vs_sma_50": 0.05,
        "close_vs_sma_200": 0.1,
        "downside_volatility": 0.1,
        "average_dollar_volume": 10_000_000.0,
        "forward_return": value,
        "benchmark_forward_return": None if value is None else value / 2,
        "relative_forward_return": None if value is None else value / 2,
        "eligible": True,
        "insufficiency_reason": "",
    }


def _formula_cases(module: Any, config: Any) -> dict[str, Any]:
    small_horizons = {
        name: replace(
            horizon,
            minimum_raw_matches=1,
            minimum_effective_cohorts=1,
            minimum_distinct_listings=1,
            shrinkage_prior_cohorts=1,
            probability_minimum_effective_cohorts=1,
            probability_minimum_distinct_listings=1,
            probability_minimum_calendar_span_days=1,
            probability_minimum_distinct_market_regimes=1,
        )
        for name, horizon in config.horizons.items()
    }
    small = replace(
        config,
        horizons=small_horizons,
        calibration=replace(
            config.calibration,
            minimum_training_cohorts=1,
            minimum_test_cohorts=1,
            maximum_baseline_mae_ratio=10,
            maximum_brier_score=1,
        ),
    )
    rows = [
        _panel_record(
            listing_id=f"L{index % 3}",
            anchor=date(2018, 1, 2) + timedelta(days=180 * index),
            value=(-0.2, 0.1, 0.4)[index % 3],
            state=index % 2,
        )
        for index in range(6)
    ]
    rows.append(_panel_record(listing_id="current", anchor=TARGET_DATE, value=None, current=True))
    panel = pl.DataFrame(rows, schema=module.PANEL_SCHEMA, orient="row")
    success = module.build_medium_forecasts(panel, small)["current"]["6m"]

    missing_rows = [row for row in rows if not row["is_forecast"]]
    missing_panel = pl.DataFrame(missing_rows, schema=module.PANEL_SCHEMA, orient="row")
    missing = module.build_medium_forecasts(missing_panel, small)

    current_missing = deepcopy(rows[-1])
    current_missing["eligible"] = False
    current_missing["insufficiency_reason"] = "frozen missing current inputs"
    ineligible_panel = pl.DataFrame(
        [*rows[:-1], current_missing],
        schema=module.PANEL_SCHEMA,
        orient="row",
    )
    ineligible = module.build_medium_forecasts(ineligible_panel, small)["current"]["6m"]

    restricted = replace(
        small,
        horizons={
            **small.horizons,
            "6m": replace(small.horizons["6m"], minimum_raw_matches=99),
        },
    )
    support_missing = module.build_medium_forecasts(panel, restricted)["current"]["6m"]
    calibration_rows: list[dict[str, object]] = []
    first_anchor = date(2018, 1, 2)
    for cohort_index in range(13):
        anchor = first_anchor + timedelta(days=182 * cohort_index)
        for listing_index in range(30):
            row = _panel_record(
                listing_id=f"cal-{listing_index:02d}",
                anchor=anchor,
                value=0.1,
                state=9,
            )
            row["market_trend_bucket"] = cohort_index % 3
            row["market_volatility_bucket"] = cohort_index % 3
            calibration_rows.append(row)
    calibration_rows.append(
        _panel_record(
            listing_id="published",
            anchor=TARGET_DATE,
            value=None,
            current=True,
            state=9,
        )
    )
    calibration_panel = pl.DataFrame(
        calibration_rows,
        schema=module.PANEL_SCHEMA,
        orient="row",
    )
    published = module.build_medium_forecasts(calibration_panel, config)["published"]["6m"]
    all_reasons_config = replace(
        config,
        horizons={
            **config.horizons,
            "6m": replace(
                config.horizons["6m"],
                probability_minimum_effective_cohorts=14,
                probability_minimum_distinct_listings=31,
                probability_minimum_calendar_span_days=3_000,
                probability_minimum_distinct_market_regimes=4,
            ),
        },
    )
    all_reasons = module.build_medium_forecasts(calibration_panel, all_reasons_config)["published"][
        "6m"
    ]
    failing_rows = deepcopy(calibration_rows)
    for index, row in enumerate(failing_rows):
        if not row["is_forecast"] and index % 2 == 0:
            row["forward_return"] = -0.1
            row["benchmark_forward_return"] = -0.05
            row["relative_forward_return"] = -0.05
    failing_config = replace(
        config,
        calibration=replace(config.calibration, maximum_brier_score=0.2),
    )
    calibration_failed = module.build_medium_forecasts(
        pl.DataFrame(failing_rows, schema=module.PANEL_SCHEMA, orient="row"),
        failing_config,
    )["published"]["6m"]
    return _normalize(
        {
            "success_scenario": success.scenario.as_dict(),
            "success_scenario_payload": success.scenario_payload(),
            "success_calculation": success.calculation,
            "missing_current_listing_map": missing,
            "ineligible_scenario": ineligible.scenario.as_dict(),
            "ineligible_calculation": ineligible.calculation,
            "support_missing_scenario": support_missing.scenario.as_dict(),
            "support_missing_calculation": support_missing.calculation,
            "probability_published_scenario": published.scenario.as_dict(),
            "probability_published_calculation": published.calculation,
            "all_probability_reasons": all_reasons.scenario.as_dict(),
            "calibration_failed_scenario": calibration_failed.scenario.as_dict(),
            "calibration_failed_calculation": calibration_failed.calculation,
        }
    )


def _parser_cases(config_module: Any, config_path: Path) -> dict[str, Any]:
    config = config_module.load_medium_forecast_config(config_path)
    cases: dict[str, Any] = {
        "default_path_name": config_module.default_medium_forecast_config_path().name,
        "effective_hash": config_module.medium_forecast_config_hash(config),
        "raw": config.raw,
    }
    malformed = {
        "not_mapping": "[]\n",
        "schema": "schema_version: 3\n",
        "broken_yaml": "schema_version: [\n",
    }
    for name, content in malformed.items():
        path = config_path.parent / f"{name}.yml"
        path.write_text(content, encoding="utf-8")
        try:
            config_module.load_medium_forecast_config(path)
        except Exception as error:
            cases[name] = {"type": type(error).__name__, "message": str(error)}
        else:  # pragma: no cover
            cases[name] = {"type": "<none>", "message": ""}
    return _normalize(cases)


def _service_capture(
    service: Any,
    config_module: Any,
    config_path: Path,
    scoring_config_path: Path,
    long_forecast_config_path: Path,
    store: AssetStore,
) -> dict[str, Any]:
    universe = Universe.objects.create(
        slug="medium-frozen",
        name="Medium frozen",
        config_version="frozen-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        id=_det("snapshot"),
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    listing = _listing("FROZEN")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    sessions = _sessions(1500)
    _asset(store, subject="SPY", sessions=sessions, slope=0.03)
    _asset(store, subject=listing.ticker, sessions=sessions, slope=0.025)
    prior_revision = os.environ.get("STANSTOCK_CODE_REVISION")
    original_default = config_module.default_medium_forecast_config_path
    sealed_asof = service.AsOfData
    original_latest_asset = sealed_asof.latest_asset
    original_read_frame = store.read_frame
    selections: list[str] = []
    reads: list[str] = []

    def latest_asset(self: Any, **kwargs: Any) -> DataAsset:
        selections.append(str(kwargs["subject"]))
        return original_latest_asset(self, **kwargs)

    def read_frame(relative_path: str) -> pl.DataFrame:
        reads.append(relative_path)
        return original_read_frame(relative_path)

    os.environ["STANSTOCK_CODE_REVISION"] = "f" * 40
    config_module.default_medium_forecast_config_path = lambda: config_path
    sealed_asof.latest_asset = latest_asset
    store.read_frame = read_frame
    try:
        results = service.analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=GENERATED_AT,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="synthetic",
            benchmark_subject="SPY",
            store=store,
            config_path=scoring_config_path,
            long_forecast_config_path=long_forecast_config_path,
            long_forecast_requested=False,
        )
    finally:
        config_module.default_medium_forecast_config_path = original_default
        sealed_asof.latest_asset = original_latest_asset
        store.read_frame = original_read_frame
        if prior_revision is None:
            os.environ.pop("STANSTOCK_CODE_REVISION", None)
        else:
            os.environ["STANSTOCK_CODE_REVISION"] = prior_revision
    result = results[0]
    predictions = list(Prediction.objects.filter(analysis=result.analysis).order_by("horizon"))
    panel = DataAsset.objects.get(kind="medium_forecast_panel", subject=str(result.run.id))
    assets = list(DataAsset.objects.order_by("provider", "kind", "subject", "id"))
    aliases = _semantic_identity_aliases(
        result=result,
        snapshot=snapshot,
        listings=[listing],
        predictions=predictions,
        assets=assets,
    )
    source_asset = next(asset for asset in assets if asset.kind == "price_history")
    return _normalize(
        {
            "run": {
                field: getattr(result.run, field)
                for field in (
                    "generated_at",
                    "data_cutoff",
                    "target_date",
                    "issued_on_time",
                    "config_version",
                    "config_hash",
                    "code_revision",
                    "status",
                )
            },
            "analysis": {
                field: getattr(result.analysis, field)
                for field in (
                    "current_price",
                    "overall_score",
                    "recommendation",
                    "risk_score",
                    "risk_class",
                    "confidence",
                    "component_scores",
                    "forecast_scenarios",
                    "reasons",
                    "risks",
                    "data_quality",
                )
            },
            "predictions": [
                {
                    field: getattr(prediction, field)
                    for field in (
                        "horizon",
                        "evidence_role",
                        "evidence_grade",
                        "issued_on_time",
                        "bear_return",
                        "base_return",
                        "bull_return",
                        "probability_positive",
                        "confidence",
                        "confidence_status",
                        "insufficiency_reason",
                        "model_version",
                        "method_version",
                        "config_hash",
                        "source_assets",
                        "calculation",
                    )
                }
                for prediction in predictions
            ],
            "panel_metadata": panel.metadata,
            "panel_sha256": panel.sha256,
            "panel_bytes": base64.b64encode(store.read_bytes(panel.relative_path)).decode(),
            "panel_rows": store.read_frame(panel.relative_path).to_dicts(),
            "stock_analysis_6m": result.analysis.six_month_forecast_scenario,
            "stock_analysis_12m": result.analysis.twelve_month_forecast_scenario,
            "source_access": {
                "selections": selections,
                "physical_reads": reads,
            },
            "identity_relationship_probe": {
                "correct_panel_reference": str(panel.id),
                "source_in_panel_role": str(source_asset.id),
                "prediction_references": [str(prediction.id) for prediction in predictions],
            },
        },
        aliases,
    )


def _capture(
    tmp_path: Path,
    *,
    config_module: Any,
    medium_module: Any,
    service_module: Any,
    tags_module: Any,
    asset_store_class: Any = AssetStore,
) -> dict[str, Any]:
    config_path = tmp_path / "us-price-medium-v1.yml"
    scoring_config_path = tmp_path / "us-price-baseline-v2.yml"
    long_forecast_config_path = tmp_path / "us-sec-long-v2.yml"
    write_base_config(config_path, BASE_CONFIG_PATHS[0])
    write_base_config(scoring_config_path, BASE_CONFIG_PATHS[1])
    write_base_config(long_forecast_config_path, BASE_CONFIG_PATHS[2])
    config = config_module.load_medium_forecast_config(config_path)
    formula_cases = _formula_cases(medium_module, config)
    service_payload = _service_capture(
        service_module,
        config_module,
        config_path,
        scoring_config_path,
        long_forecast_config_path,
        asset_store_class(tmp_path / "assets"),
    )
    return {
        "config": _parser_cases(config_module, config_path),
        "formula_cases": formula_cases,
        "service": service_payload,
        "rendering": {
            "scenario_range": tags_module.scenario_range(service_payload["stock_analysis_6m"]),
            "calibrated_label": tags_module.display_label("empirical_calibrated"),
            "range_only_label": tags_module.display_label("empirical_range_only"),
        },
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _golden_capture() -> dict[str, Any]:
    document = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    compressed = base64.b64decode(document["capture_zlib_base64"])
    return json.loads(zlib.decompress(compressed))


def _assert_matches_golden(capture: dict[str, Any]) -> None:
    expected = _golden_capture()
    assert capture == expected, _first_difference(capture, expected)


def _first_difference(left: Any, right: Any, path: str = "capture") -> str:
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


def test_frozen_float_normalization_is_platform_stable_across_nested_payloads() -> None:
    linux_value = 96.9674184132452
    macos_value = 96.96741841324523

    def nested_capture(value: float) -> dict[str, Any]:
        return {
            "calculation": {"risk": {"beta": value}},
            "scenario": [{"probability_positive": value}],
            "panel_rows": [{"relative_forward_return": value}],
        }

    linux = _normalize(nested_capture(linux_value))
    macos = _normalize(nested_capture(macos_value))

    assert linux == macos
    assert linux["calculation"]["risk"]["beta"] == 96.9674184132452
    assert linux["scenario"][0]["probability_positive"] == 96.9674184132452
    assert linux["panel_rows"][0]["relative_forward_return"] == 96.9674184132452
    assert _normalize(nested_capture(96.9674184132462)) != linux


def test_frozen_float_normalization_preserves_other_scalar_handling() -> None:
    identity = UUID("ca05db92-12eb-4634-aa95-52f5005543e0")
    aliases = _IdentityAliases(by_uuid={str(identity): "<fixture>"}, run_aliases={})

    normalized = _normalize(
        {
            "negative_zero": -0.0,
            "integer": 7,
            "boolean": False,
            "decimal": Decimal("96.9674184132452300"),
            "date": date(2026, 9, 4),
            "identity": identity,
        },
        aliases,
    )

    assert normalized == {
        "negative_zero": 0.0,
        "integer": 7,
        "boolean": False,
        "decimal": "96.9674184132452300",
        "date": "2026-09-04",
        "identity": "<fixture>",
    }
    assert math.copysign(1.0, normalized["negative_zero"]) == 1.0
    assert type(normalized["integer"]) is int
    assert type(normalized["boolean"]) is bool


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_frozen_float_normalization_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(
        ValueError,
        match="^Frozen capture cannot normalize a non-finite float$",
    ):
        _normalize({"nested": [{"value": value}]})


@pytest.mark.django_db
def test_frozen_medium_v1_matches_base_produced_golden(tmp_path: Path) -> None:
    live = _capture(
        tmp_path,
        config_module=importlib.import_module("stanstock.research.forecast_config"),
        medium_module=importlib.import_module("stanstock.research.medium_forecasts"),
        service_module=importlib.import_module("stanstock.research.service"),
        tags_module=importlib.import_module("stanstock.web.templatetags.stanstock"),
    )
    _assert_matches_golden(live)


def test_medium_golden_pins_base_identity_and_dependency_closure() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["base_sha"] == BASE_SHA
    assert golden["base_tree"] == BASE_TREE
    assert golden["generated_from"] == "base_revision_execution"
    assert golden["base_config_sha256"]["config/forecasts/us-price-medium-v1.yml"] == (
        "8c6bbfb7602d3966536675412ea716e3219f03ab578d74ed2a6f214611d93853"
    )
    assert golden["base_config_sha256"]["config/forecasts/us-sec-long-v2.yml"] == (
        "51b0a1c4711321d65616555e9a87892c31ad93bed24d0a7a8dd56252dd46b51e"
    )
    assert len(BASE_RUNTIME_MODULE_PATHS) == 27
    assert len(BASE_CONFIG_PATHS) == 3
    assert set(golden["all_base_sha256"]) == {
        *(path for _name, path in BASE_MODULE_PATHS),
        *BASE_CONFIG_PATHS,
        *imported_repository_dependencies(),
        *NON_MODULE_COLLABORATORS,
    }
    capture_bytes = _canonical(_golden_capture())
    assert len(capture_bytes) == golden["capture_bytes"]
    assert hashlib.sha256(capture_bytes).hexdigest() == golden["capture_sha256"]


def test_frozen_identity_aliases_preserve_panel_and_source_roles() -> None:
    probe = _golden_capture()["service"]["identity_relationship_probe"]
    panel_alias = probe["correct_panel_reference"]
    source_alias = probe["source_in_panel_role"]

    assert panel_alias.startswith("<panel-asset:")
    assert source_alias.startswith("<source-asset:")
    assert panel_alias != source_alias
    assert len(set(probe["prediction_references"])) == len(probe["prediction_references"])
    assert all(alias.startswith("<prediction:") for alias in probe["prediction_references"])
    assert _canonical({"panel_asset_id": panel_alias}) != _canonical(
        {"panel_asset_id": source_alias}
    )


@pytest.mark.skipif(not base_sources_available(), reason="sealed base objects unavailable")
def test_medium_golden_hashes_match_local_base_objects() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["all_base_sha256"] == base_source_checksums()


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="sealed base objects unavailable")
def test_frozen_medium_v1_matches_live_base_execution(tmp_path: Path) -> None:
    with base_medium_modules() as base:
        capture = _capture(
            tmp_path,
            config_module=base.forecast_config,
            medium_module=base.medium_forecasts,
            service_module=base.service,
            tags_module=base.template_tags,
            asset_store_class=base.service.AssetStore,
        )
    _assert_matches_golden(capture)


@pytest.mark.django_db
def test_medium_golden_regeneration_refuses_without_base_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import medium_frozen_base

    monkeypatch.setattr(
        medium_frozen_base,
        "read_base_bytes",
        lambda path: (_ for _ in ()).throw(BaseRevisionUnavailableError(path)),
    )
    target = tmp_path / "golden.json"
    with pytest.raises(BaseRevisionUnavailableError):
        regenerate_golden(target, tmp_path / "work")
    assert not target.exists()


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="sealed base objects unavailable")
def test_head_monkeypatch_cannot_contaminate_base_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    live = importlib.import_module("stanstock.research.medium_forecasts")
    monkeypatch.setattr(
        live,
        "build_medium_forecasts",
        lambda *_args, **_kwargs: pytest.fail("base regeneration executed head code"),
    )
    target = tmp_path / "regenerated.json"
    regenerate_golden(target, tmp_path / "work")
    assert target.read_bytes() == GOLDEN_PATH.read_bytes()


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="sealed base objects unavailable")
def test_live_asof_monkeypatch_cannot_contaminate_base_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    live_asof = importlib.import_module("stanstock.data.asof")
    monkeypatch.setattr(
        live_asof.AsOfData,
        "price_frame_with_diagnostics",
        lambda *_args, **_kwargs: pytest.fail(
            "sealed base regeneration executed the live AsOfData dependency"
        ),
    )
    target = tmp_path / "regenerated.json"
    regenerate_golden(target, tmp_path / "work")
    assert target.read_bytes() == GOLDEN_PATH.read_bytes()


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="sealed base objects unavailable")
def test_worktree_long_config_cannot_contaminate_base_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    live = importlib.import_module("stanstock.research.long_forecast_config")
    worktree_default = live.default_long_forecast_config_path().resolve()
    original_read_text = Path.read_text

    monkeypatch.setattr(
        live,
        "load_long_forecast_config",
        lambda *_args, **_kwargs: pytest.fail(
            "base regeneration executed the worktree long-config reader"
        ),
    )

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.resolve() == worktree_default:
            pytest.fail("base regeneration read the worktree long-config bytes")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    target = tmp_path / "regenerated.json"
    regenerate_golden(target, tmp_path / "work")
    assert target.read_bytes() == GOLDEN_PATH.read_bytes()


@pytest.mark.django_db
@pytest.mark.skipif(not base_sources_available(), reason="sealed base objects unavailable")
def test_live_model_helper_cannot_contaminate_base_regeneration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = importlib.import_module("stanstock.research.models")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("base regeneration executed the live model helper")

    monkeypatch.setattr(models, "scenario_from_document", forbidden)
    target = tmp_path / "regenerated.json"
    regenerate_golden(target, tmp_path / "work")
    assert target.read_bytes() == GOLDEN_PATH.read_bytes()
    assert models.scenario_from_document is forbidden


@pytest.mark.skipif(not base_sources_available(), reason="sealed base objects unavailable")
def test_sealed_model_helper_and_module_bindings_restore_after_failure() -> None:
    models = importlib.import_module("stanstock.research.models")
    original_helper = models.scenario_from_document
    originals = {name: sys.modules.get(name) for name, _path in BASE_RUNTIME_MODULE_PATHS}

    with pytest.raises(RuntimeError, match="deliberate sealed capture failure"):
        with base_medium_modules():
            sealed_forecasting = sys.modules["stanstock.research.forecasting"]
            assert models.scenario_from_document is sealed_forecasting.scenario_from_document
            raise RuntimeError("deliberate sealed capture failure")

    assert models.scenario_from_document is original_helper
    assert {name: sys.modules.get(name) for name, _path in BASE_RUNTIME_MODULE_PATHS} == originals


@pytest.mark.django_db
@pytest.mark.skipif(
    os.environ.get(WRITE_GOLDEN_ENV) != "1",
    reason=f"set {WRITE_GOLDEN_ENV}=1 to regenerate",
)
def test_regenerate_medium_golden(tmp_path: Path) -> None:  # pragma: no cover
    regenerate_golden(GOLDEN_PATH, tmp_path)


def regenerate_golden(path: Path, work_path: Path) -> dict[str, Any]:
    checksums = base_source_checksums()
    work_path.mkdir(parents=True, exist_ok=True)
    with base_medium_modules() as base:
        capture = _capture(
            work_path,
            config_module=base.forecast_config,
            medium_module=base.medium_forecasts,
            service_module=base.service,
            tags_module=base.template_tags,
            asset_store_class=base.service.AssetStore,
        )
    capture_bytes = _canonical(capture)
    document = {
        "base_sha": BASE_SHA,
        "base_tree": BASE_TREE,
        "generated_by": "tests/test_medium_frozen_differential.py",
        "generated_from": "base_revision_execution",
        "base_source_sha256": {
            source_path: checksums[source_path] for _name, source_path in BASE_MODULE_PATHS
        },
        "base_config_sha256": {
            config_path: checksums[config_path] for config_path in BASE_CONFIG_PATHS
        },
        "fixed_collaborator_sha256": {
            dependency: checksums[dependency]
            for dependency in (*imported_repository_dependencies(), *NON_MODULE_COLLABORATORS)
        },
        "all_base_sha256": checksums,
        "capture_bytes": len(capture_bytes),
        "capture_sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "capture_zlib_base64": base64.b64encode(zlib.compress(capture_bytes, level=9)).decode(),
    }
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return document
