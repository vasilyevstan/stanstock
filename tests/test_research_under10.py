"""`us-under10-shadow-v1`: the pure shadow policy and its payload contract.

Every fixture here is synthetic and constructed in memory: the builder is a
pure function over already-resolved evidence, so these tests never touch the
database, a provider, or an asset store.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from typing import Any
from uuid import UUID, uuid4, uuid5

import polars as pl
import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from stanstock.data.fact_identity import build_observation_hash, build_period_identity
from stanstock.data.models import DataAsset, FundamentalFact
from stanstock.data.provider_policy import SEC_PROVIDER
from stanstock.data.sec_fundamentals import CORRECTION_AVAILABILITY_BASIS
from stanstock.research.under10 import (
    REASON_CURRENT_ASSETS_BELOW_LIABILITIES,
    REASON_EVIDENCE_NOT_CUTOFF_SAFE,
    REASON_FREE_CASH_FLOW_MISSING,
    REASON_INCOMPATIBLE_UNIT,
    REASON_INSTANT_PERIOD_MISMATCH,
    REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING,
    REASON_NEAR_TERM_DEBT_EXCEEDS_CASH,
    REASON_NEGATIVE_FREE_CASH_FLOW,
    REASON_NONFINITE_INPUT,
    REASON_NONPOSITIVE_CURRENT_LIABILITIES,
    REASON_RUNWAY_BELOW_MINIMUM_QUARTERS,
    REASON_STALE_METRIC,
    SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION,
    SOLVENCY_ELEVATED_OBLIGATION_RISK,
    SOLVENCY_INSUFFICIENT_EVIDENCE,
    SOLVENCY_NO_ADVERSE_EVIDENCE,
    UNDER10_ACTIVATED,
    UNDER10_CONCEPTS,
    UNDER10_LIQUIDITY_SESSIONS,
    UNDER10_MAX_METRIC_AGE_DAYS,
    UNDER10_MAX_PRICE_STALENESS_DAYS,
    UNDER10_MIN_RUNWAY_QUARTERS,
    UNDER10_SCHEMA_VERSION,
    UNDER10_SHADOW_POLICY_VERSION,
    build_under10_assessment,
    canonical_json,
    qualify_under10_sec_facts,
    under10_assessment_hash,
    under10_policy_document,
    under10_policy_hash,
)

#: Literal pin. Comparing two current computations proves nothing, so the
#: expected digest is written out; it moves only with an explicit new policy
#: version or a change to the reviewed SEC fundamentals configuration.
EXPECTED_POLICY_HASH = "e97e1b9bd40693efe174ff512b7a6318a353f4c4ca82bf034067682fb017add0"

TARGET_DATE = date(2026, 3, 2)
DATA_CUTOFF = datetime(2026, 3, 2, 21, 0, tzinfo=UTC)
INSTANT_DATE = date(2025, 12, 31)
ANNUAL_PERIOD = (date(2025, 1, 1), date(2025, 12, 31))
FIXTURE_NAMESPACE = UUID("9d1b7d6a-0000-4000-8000-554e44455231")
#: The one shared, permanent `Listing.id` every payload built by `_build`/
#: `_build_payload` in this module claims by default -- deterministic (not
#: `uuid4()`) so a genuine payload's `evaluated_for.listing_id` is stable
#: across test runs; a distinct id is passed explicitly wherever a test
#: constructs a *second*, unrelated listing (see the C1/F2 identity tests).
LISTING_ID = str(uuid5(FIXTURE_NAMESPACE, "listing:primary"))

SOURCE_CONCEPTS = {
    "cash_and_equivalents": "CashAndCashEquivalentsAtCarryingValue",
    "short_term_debt": "ShortTermBorrowings",
    "current_long_term_debt": "LongTermDebtCurrent",
    "current_assets": "AssetsCurrent",
    "current_liabilities": "LiabilitiesCurrent",
    "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditure": "PaymentsToAcquirePropertyPlantAndEquipment",
}

COMPANY_ID = uuid5(FIXTURE_NAMESPACE, "company")


# `sec_config` is a shared pytest fixture defined in `tests/conftest.py`
# (used by this file and `test_web_under10_reader_matrix.py`); no import is
# required or possible for a conftest-provided fixture -- pytest resolves it
# by name alone.


# ---------------------------------------------------------------------------
# Synthetic, in-memory evidence
# ---------------------------------------------------------------------------


def _asset(
    key: str = "sec-companyfacts",
    *,
    retrieved_at: datetime | None = None,
    available_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    provider: str = "sec",
    kind: str = "raw_fundamentals",
) -> DataAsset:
    stamp = retrieved_at or (DATA_CUTOFF - timedelta(days=1))
    return DataAsset(
        id=uuid5(FIXTURE_NAMESPACE, f"asset:{key}"),
        provider=provider,
        kind=kind,
        subject="UNDR",
        relative_path=f"tests/under10/{key}.json",
        sha256=key.encode("utf-8").hex().ljust(64, "0")[:64],
        retrieved_at=stamp,
        available_at=available_at or stamp,
        metadata=metadata or {},
    )


def _fact(
    concept: str,
    value: str,
    *,
    asset: DataAsset,
    period_end: date = INSTANT_DATE,
    period_start: date | None = None,
    unit: str = "USD",
    accession: str | None = None,
    source_revision: int = 1,
    availability_basis: str = "legacy",
    available_at: datetime | None = None,
    provider: str = "sec",
    fact_key: str | None = None,
) -> FundamentalFact:
    period_type = "instant" if period_start is None else "duration"
    identity = build_period_identity(
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
    )
    accession_value = accession or f"0000000000-25-{concept[:6]}"
    return FundamentalFact(
        id=uuid5(FIXTURE_NAMESPACE, fact_key or f"{concept}:{period_end}:{source_revision}"),
        company_id=COMPANY_ID,
        provider=provider,
        concept=concept,
        taxonomy="us-gaap",
        source_concept=f"us-gaap:{SOURCE_CONCEPTS[concept]}",
        value=Decimal(value),
        unit=unit,
        currency="USD" if unit == "USD" else "",
        period_type=period_type,
        period_identity=identity,
        period_start=period_start,
        period_end=period_end,
        fiscal_year=period_end.year,
        fiscal_period="FY",
        accession=accession_value,
        filing_form="10-K",
        filing_date=period_end + timedelta(days=45),
        available_at=available_at or (DATA_CUTOFF - timedelta(days=10)),
        availability_basis=availability_basis,
        source_revision=source_revision,
        observation_hash=build_observation_hash(
            taxonomy="us-gaap",
            source_concept=SOURCE_CONCEPTS[concept],
            value=Decimal(value),
            unit=unit,
            currency="USD",
            period_identity=identity,
            fiscal_year=period_end.year,
            fiscal_period="FY",
            accession=accession_value,
            filing_form="10-K",
            filing_date=period_end + timedelta(days=45),
            acceptance_at=None,
            frame="",
        ),
        source_asset=asset,
    )


def _facts(
    *,
    cash: str | None = "1000.00000000",
    short_term_debt: str | None = "100.00000000",
    current_long_term_debt: str | None = "50.00000000",
    current_assets: str | None = "2000.00000000",
    current_liabilities: str | None = "1000.00000000",
    operating_cash_flow: str | None = "400.00000000",
    capital_expenditure: str | None = "100.00000000",
    instant_date: date = INSTANT_DATE,
    annual_period: tuple[date, date] = ANNUAL_PERIOD,
    asset: DataAsset | None = None,
    unit_overrides: dict[str, str] | None = None,
) -> list[FundamentalFact]:
    source = asset or _asset()
    units = unit_overrides or {}
    rows: list[FundamentalFact] = []
    instants = {
        "cash_and_equivalents": cash,
        "short_term_debt": short_term_debt,
        "current_long_term_debt": current_long_term_debt,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
    }
    for concept, value in instants.items():
        if value is None:
            continue
        rows.append(
            _fact(
                concept,
                value,
                asset=source,
                period_end=instant_date,
                unit=units.get(concept, "USD"),
            )
        )
    durations = {
        "operating_cash_flow": operating_cash_flow,
        "capital_expenditure": capital_expenditure,
    }
    for concept, value in durations.items():
        if value is None:
            continue
        rows.append(
            _fact(
                concept,
                value,
                asset=source,
                period_start=annual_period[0],
                period_end=annual_period[1],
                unit=units.get(concept, "USD"),
            )
        )
    return rows


def _price_asset(
    *,
    metadata: dict[str, Any] | None = None,
) -> DataAsset:
    return _asset(
        "twelve-data-price",
        provider="twelve_data",
        kind="price_history",
        metadata=(
            metadata
            if metadata is not None
            else {
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "currency": "USD",
            }
        ),
    )


def _price_frame(
    *,
    sessions: int = 252,
    close: float = 4.0,
    volume: float = 1_000_000.0,
    last_session: date = TARGET_DATE,
) -> pl.DataFrame:
    dates = [last_session - timedelta(days=index) for index in range(sessions)][::-1]
    return pl.DataFrame(
        {
            "date": dates,
            "close": [close] * sessions,
            "volume": [volume] * sessions,
        }
    )


_UNSET: Any = object()


def _build(
    sec_config,
    *,
    facts: list[FundamentalFact] | None = None,
    price_frame: pl.DataFrame | None = None,
    price_asset: DataAsset | None = _UNSET,
    price_source: dict[str, Any] | None = _UNSET,
    reference_close: Decimal = Decimal("4.250000"),
    target_date: date = TARGET_DATE,
    data_cutoff: datetime = DATA_CUTOFF,
    provider: str = "twelve_data",
    provider_plan: str | None = "basic",
    evidence_cutoff_safe: bool = True,
    company_identity_present: bool = True,
    invalid_session_date_rows: int = 0,
    listing_id: str = LISTING_ID,
) -> dict[str, Any]:
    asset = _price_asset() if price_asset is _UNSET else price_asset
    if price_source is _UNSET:
        price_source = {"asset_id": str(asset.id)} if asset is not None else None
    return build_under10_assessment(
        facts=facts if facts is not None else _facts(),
        sec_config=sec_config,
        price_frame=price_frame if price_frame is not None else _price_frame(),
        price_asset=asset,
        price_source=price_source,
        reference_close=reference_close,
        target_date=target_date,
        data_cutoff=data_cutoff,
        code_revision_value="0" * 40,
        provider=provider,
        provider_plan=provider_plan,
        evidence_cutoff_safe=evidence_cutoff_safe,
        company_identity_present=company_identity_present,
        invalid_session_date_rows=invalid_session_date_rows,
        listing_id=listing_id,
    )


# ---------------------------------------------------------------------------
# Policy identity
# ---------------------------------------------------------------------------


def test_policy_constants_are_frozen(sec_config) -> None:
    assert UNDER10_SHADOW_POLICY_VERSION == "us-under10-shadow-v1"
    assert UNDER10_SCHEMA_VERSION == 1
    assert UNDER10_LIQUIDITY_SESSIONS == 252
    assert UNDER10_MAX_METRIC_AGE_DAYS == 200
    assert UNDER10_MAX_PRICE_STALENESS_DAYS == 7
    assert UNDER10_MIN_RUNWAY_QUARTERS == Decimal("4")
    assert UNDER10_ACTIVATED is False
    assert UNDER10_CONCEPTS == (
        "cash_and_equivalents",
        "short_term_debt",
        "current_long_term_debt",
        "current_assets",
        "current_liabilities",
        "operating_cash_flow",
        "capital_expenditure",
    )


def test_policy_hash_matches_its_literal_pin(sec_config) -> None:
    assert under10_policy_hash(sec_config) == EXPECTED_POLICY_HASH


def test_policy_hash_binds_the_sec_fundamentals_configuration(sec_config) -> None:
    from dataclasses import replace as dataclass_replace

    rebound = dataclass_replace(sec_config, config_hash="f" * 64)

    assert under10_policy_hash(rebound) != under10_policy_hash(sec_config)


def test_policy_hash_binds_both_sec_provider_qualification_fields(
    sec_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.under10 as under10_module

    qualification = under10_policy_document(sec_config)["selection"]["sec_evidence_qualification"]
    assert qualification == {
        "all_conditions_required": True,
        "canonical_concepts": list(UNDER10_CONCEPTS),
        "fact_provider": SEC_PROVIDER,
        "source_asset_provider": SEC_PROVIDER,
    }

    expected = under10_policy_hash(sec_config)
    monkeypatch.setattr(under10_module, "SEC_PROVIDER", "different_sec_provider")

    assert under10_policy_hash(sec_config) != expected


def test_no_liquidity_threshold_constant_exists() -> None:
    import stanstock.research.under10 as module

    assert not hasattr(module, "UNDER10_MIN_MEDIAN_DOLLAR_VOLUME_252")
    assert not any("MIN_MEDIAN" in name for name in dir(module))


# ---------------------------------------------------------------------------
# Payload contract
# ---------------------------------------------------------------------------


def test_payload_shape_and_fixed_inactive_summaries(sec_config) -> None:
    payload = _build(sec_config)

    assert set(payload) == {
        "schema_version",
        "policy_version",
        "policy_hash",
        "assessment_hash",
        "activated",
        "shadow_only",
        "activation_eligible",
        "code_revision",
        "evaluated_for",
        "solvency",
        "liquidity",
        "split_verification",
        "gates",
        "blocking_reasons",
        "new_allocation_percent",
    }
    assert payload["schema_version"] == 1
    assert payload["policy_version"] == "us-under10-shadow-v1"
    assert payload["activated"] is False
    assert payload["shadow_only"] is True
    assert payload["activation_eligible"] is False
    assert payload["gates"] == {
        "solvency_obligation": False,
        "dollar_liquidity_252": False,
        "verified_split_evidence": False,
    }
    assert payload["new_allocation_percent"] == 0
    assert payload["blocking_reasons"] == [payload["split_verification"]["reason"]]
    assert payload["evaluated_for"] == {
        "listing_id": LISTING_ID,
        "target_date": "2026-03-02",
        "data_cutoff": DATA_CUTOFF.isoformat(),
        "price_band": "under_10",
        "reference_close": "4.250000",
        "date_basis": "decision_target",
        "currency": "USD",
    }
    assert set(payload["solvency"]) == {
        "status",
        "reasons",
        "inputs",
        "periods",
        "runway",
        "assessed_fact_ids",
        "assessed_assets",
    }
    assert set(payload["liquidity"]) == {
        "status",
        "metric",
        "value",
        "currency",
        "sessions_used",
        "first_session",
        "last_session",
        "basis",
        "price_asset",
        "reason",
    }


@pytest.mark.parametrize(
    "case",
    [
        {},
        {"facts": []},
        {"facts": _facts(cash="0.00000000", operating_cash_flow="-500.00000000")},
        {"evidence_cutoff_safe": False},
        {"price_frame": _price_frame(sessions=10)},
        {"provider": "synthetic_demo", "provider_plan": None},
    ],
)
def test_activation_summaries_are_false_in_every_branch(sec_config, case: dict[str, Any]) -> None:
    payload = _build(sec_config, **case)

    assert payload["activated"] is False
    assert payload["activation_eligible"] is False
    assert payload["shadow_only"] is True
    assert set(payload["gates"].values()) == {False}
    assert payload["new_allocation_percent"] == 0
    assert payload["split_verification"]["status"] == "unavailable"
    assert payload["split_verification"]["inference_prohibited"] is True


def test_assessment_hash_covers_everything_but_itself(sec_config) -> None:
    payload = _build(sec_config)

    assert payload["assessment_hash"] == under10_assessment_hash(payload)
    reordered = dict(reversed(list(payload.items())))
    assert under10_assessment_hash(reordered) == payload["assessment_hash"]
    mutated = {**payload, "code_revision": "1" * 40}
    assert under10_assessment_hash(mutated) != payload["assessment_hash"]


def test_canonical_json_is_deterministic_and_rejects_nonfinite() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"a": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"a": float("inf")})


def test_payload_is_json_serializable_without_python_objects(sec_config) -> None:
    payload = _build(sec_config)

    round_tripped = json.loads(json.dumps(payload, allow_nan=False))

    assert round_tripped == payload


# ---------------------------------------------------------------------------
# Four-state partition
# ---------------------------------------------------------------------------


def test_complete_evidence_without_adverse_signals(sec_config) -> None:
    payload = _build(sec_config)

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    assert solvency["reasons"] == []
    assert solvency["inputs"] == {
        "cash_and_equivalents": "1000.00000000",
        "near_term_debt": "150.00000000",
        "current_assets": "2000.00000000",
        "current_liabilities": "1000.00000000",
        "current_ratio": "2.0000",
        "free_cash_flow": "300.00000000",
    }
    assert solvency["periods"] == {
        "instant_date": "2025-12-31",
        "duration_start": "2025-01-01",
        "duration_end": "2025-12-31",
        "duration_basis": "annual",
    }
    assert solvency["runway"] == {
        "status": "not_applicable_positive_fcf",
        "quarters": None,
        "reason": None,
    }


def test_adverse_state_through_low_current_assets(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="100.00000000",
            short_term_debt="400.00000000",
            current_long_term_debt="100.00000000",
            current_assets="900.00000000",
            current_liabilities="1000.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    assert REASON_NEAR_TERM_DEBT_EXCEEDS_CASH in solvency["reasons"]
    assert REASON_CURRENT_ASSETS_BELOW_LIABILITIES in solvency["reasons"]


def test_adverse_state_through_short_runway(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="100.00000000",
            short_term_debt="400.00000000",
            current_long_term_debt="100.00000000",
            current_assets="2000.00000000",
            current_liabilities="1000.00000000",
            operating_cash_flow="-500.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    assert REASON_RUNWAY_BELOW_MINIMUM_QUARTERS in solvency["reasons"]
    assert solvency["runway"]["status"] == "computed"
    assert solvency["runway"]["quarters"] == "0.8000"


def test_elevated_state_from_negative_free_cash_flow_alone(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="10000.00000000",
            short_term_debt="10.00000000",
            current_long_term_debt="10.00000000",
            current_assets="5000.00000000",
            current_liabilities="1000.00000000",
            operating_cash_flow="-10.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert solvency["reasons"] == [REASON_NEGATIVE_FREE_CASH_FLOW]
    # Cash covers the burn for far more than four quarters, so the runway
    # predicate did not fire and cannot escalate the state.
    assert REASON_RUNWAY_BELOW_MINIMUM_QUARTERS not in solvency["reasons"]


def test_elevated_state_from_debt_above_cash_alone(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="10.00000000",
            short_term_debt="100.00000000",
            current_long_term_debt="0.00000000",
            current_assets="5000.00000000",
            current_liabilities="1000.00000000",
        ),
    )

    assert payload["solvency"]["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert payload["solvency"]["reasons"] == [REASON_NEAR_TERM_DEBT_EXCEEDS_CASH]


def test_elevated_state_from_current_assets_below_liabilities_alone(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="10000.00000000",
            short_term_debt="10.00000000",
            current_long_term_debt="10.00000000",
            current_assets="900.00000000",
            current_liabilities="1000.00000000",
        ),
    )

    assert payload["solvency"]["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert payload["solvency"]["reasons"] == [REASON_CURRENT_ASSETS_BELOW_LIABILITIES]


def test_missing_evidence_beside_adverse_evidence_stays_insufficient(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="10.00000000",
            short_term_debt="900.00000000",
            current_long_term_debt=None,
            current_assets="100.00000000",
            current_liabilities="1000.00000000",
            operating_cash_flow="-900.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING in solvency["reasons"]
    assert solvency["inputs"]["near_term_debt"] is None
    assert REASON_NEAR_TERM_DEBT_EXCEEDS_CASH not in solvency["reasons"]
    assert REASON_CURRENT_ASSETS_BELOW_LIABILITIES not in solvency["reasons"]


# ---------------------------------------------------------------------------
# Exact Decimal boundaries
# ---------------------------------------------------------------------------


def test_debt_equal_to_cash_is_not_adverse(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="150.00000000",
            short_term_debt="100.00000000",
            current_long_term_debt="50.00000000",
        ),
    )

    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE


def test_current_assets_equal_to_liabilities_is_not_adverse(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(current_assets="1000.00000000", current_liabilities="1000.00000000"),
    )

    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    assert payload["solvency"]["inputs"]["current_ratio"] == "1.0000"


def test_zero_free_cash_flow_is_not_negative(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(operating_cash_flow="100.00000000", capital_expenditure="100.00000000"),
    )

    assert payload["solvency"]["inputs"]["free_cash_flow"] == "0.00000000"
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    assert payload["solvency"]["runway"] == {
        "status": "not_applicable_positive_fcf",
        "quarters": None,
        "reason": None,
    }


def test_exactly_four_quarters_of_runway_is_not_adverse(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="100.00000000",
            short_term_debt="500.00000000",
            current_long_term_debt="0.00000000",
            current_assets="2000.00000000",
            current_liabilities="1000.00000000",
            operating_cash_flow="-100.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["runway"]["quarters"] == "4.0000"
    assert solvency["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert REASON_RUNWAY_BELOW_MINIMUM_QUARTERS not in solvency["reasons"]


def test_runway_just_below_four_quarters_is_adverse_at_eight_decimals(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="99.99999999",
            short_term_debt="500.00000000",
            current_long_term_debt="0.00000000",
            current_assets="2000.00000000",
            current_liabilities="1000.00000000",
            operating_cash_flow="-100.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    # A displayed 4.0000 must never override the exact below-four decision.
    assert solvency["runway"]["quarters"] == "4.0000"
    assert solvency["status"] == SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    assert REASON_RUNWAY_BELOW_MINIMUM_QUARTERS in solvency["reasons"]


def test_runway_just_above_four_quarters_is_not_adverse_at_eight_decimals(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="100.00000001",
            short_term_debt="500.00000000",
            current_long_term_debt="0.00000000",
            current_assets="2000.00000000",
            current_liabilities="1000.00000000",
            operating_cash_flow="-100.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    assert payload["solvency"]["runway"]["quarters"] == "4.0000"
    assert payload["solvency"]["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK


def test_zero_cash_with_negative_free_cash_flow_reports_zero_runway(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash="0.00000000",
            operating_cash_flow="-100.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    runway = payload["solvency"]["runway"]
    assert runway["status"] == "computed"
    assert runway["quarters"] == "0.0000"
    assert runway["reason"] is None


# ---------------------------------------------------------------------------
# Missing versus zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", ["short_term_debt", "current_long_term_debt"])
def test_each_missing_debt_component_is_missing_not_zero(sec_config, component: str) -> None:
    payload = _build(sec_config, facts=_facts(**{component: None}))

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert solvency["reasons"] == [REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING]
    assert solvency["inputs"]["near_term_debt"] is None


def test_explicitly_zero_debt_components_are_usable(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(short_term_debt="0.00000000", current_long_term_debt="0.00000000"),
    )

    assert payload["solvency"]["inputs"]["near_term_debt"] == "0.00000000"
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE


@pytest.mark.parametrize(
    "concept",
    ["cash_and_equivalents", "current_assets", "current_liabilities"],
)
def test_each_missing_instant_concept_is_named(sec_config, concept: str) -> None:
    mapping = {
        "cash_and_equivalents": "cash",
        "current_assets": "current_assets",
        "current_liabilities": "current_liabilities",
    }
    payload = _build(sec_config, facts=_facts(**{mapping[concept]: None}))

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert f"{concept}_missing" in solvency["reasons"]
    assert solvency["inputs"][concept if concept != "cash_and_equivalents" else concept] is None


def test_missing_free_cash_flow_withholds_runway_without_zero_or_infinity(sec_config) -> None:
    payload = _build(sec_config, facts=_facts(capital_expenditure=None))

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_FREE_CASH_FLOW_MISSING in solvency["reasons"]
    assert solvency["inputs"]["free_cash_flow"] is None
    assert solvency["runway"] == {
        "status": "withheld",
        "quarters": None,
        "reason": REASON_FREE_CASH_FLOW_MISSING,
    }


def test_missing_cash_withholds_runway_but_reports_negative_free_cash_flow(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            cash=None,
            operating_cash_flow="-100.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert solvency["runway"]["status"] == "withheld"
    assert solvency["runway"]["quarters"] is None
    assert solvency["runway"]["reason"] == "cash_and_equivalents_missing"
    assert solvency["inputs"]["free_cash_flow"] == "-100.00000000"


def test_runway_is_reportable_while_another_obligation_input_is_missing(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(
            current_assets=None,
            cash="100.00000000",
            operating_cash_flow="-200.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert solvency["runway"]["status"] == "computed"
    assert solvency["runway"]["quarters"] == "2.0000"


# ---------------------------------------------------------------------------
# Numeric validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("liabilities", ["0.00000000", "-5.00000000"])
def test_nonpositive_current_liabilities_withhold_the_ratio(sec_config, liabilities: str) -> None:
    payload = _build(sec_config, facts=_facts(current_liabilities=liabilities))

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_NONPOSITIVE_CURRENT_LIABILITIES in solvency["reasons"]
    assert solvency["inputs"]["current_ratio"] is None
    assert solvency["inputs"]["current_liabilities"] is None


@pytest.mark.parametrize("liabilities", ["0.00000000", "-5.00000000"])
@pytest.mark.parametrize("missing_debt_component", ["short_term_debt", "current_long_term_debt"])
def test_missing_debt_and_nonpositive_liabilities_both_stay_recorded(
    sec_config,
    missing_debt_component: str,
    liabilities: str,
) -> None:
    """A validity defect in one concept must never suppress another's.

    The non-positive current-liabilities check must run before the early
    missing-debt return, not only when every other concept happens to be
    present: both insufficiency reasons survive together, and the withheld
    current-liabilities value is never presented as usable just because a
    different concept was the one that forced the early return.
    """
    payload = _build(
        sec_config,
        facts=_facts(
            **{missing_debt_component: None},
            current_liabilities=liabilities,
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING in solvency["reasons"]
    assert REASON_NONPOSITIVE_CURRENT_LIABILITIES in solvency["reasons"]
    assert solvency["inputs"]["near_term_debt"] is None
    assert solvency["inputs"]["current_liabilities"] is None
    assert solvency["inputs"]["current_ratio"] is None


def test_incompatible_units_are_not_usable_inputs(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(unit_overrides={"cash_and_equivalents": "EUR"}),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert solvency["reasons"] == [REASON_INCOMPATIBLE_UNIT]
    assert solvency["inputs"]["cash_and_equivalents"] is None
    assert solvency["runway"]["status"] == "not_applicable_positive_fcf"


def test_negative_free_cash_flow_is_adverse_evidence_not_missing(sec_config) -> None:
    payload = _build(
        sec_config,
        facts=_facts(operating_cash_flow="-1.00000000", capital_expenditure="0.00000000"),
    )

    solvency = payload["solvency"]
    assert solvency["inputs"]["free_cash_flow"] == "-1.00000000"
    assert solvency["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert REASON_NEGATIVE_FREE_CASH_FLOW in solvency["reasons"]


# ---------------------------------------------------------------------------
# Ambient Decimal precision and nonfinite-operand refusal
#
# `build_sec_fundamental_series` performs unguarded Decimal arithmetic (the
# free-cash-flow subtraction, TTM summation, and YTD/discrete-quarter
# subtraction) with whatever precision is *ambient* when it runs -- Python's
# `decimal` context is a contextvar, not something this policy's caller
# controls. Every case below sets a *different* ambient precision before
# calling the builder and asserts the identical, correct result every time,
# proving the policy's own `localcontext(prec=64)` -- not the caller's
# default -- is what the frozen arithmetic actually uses.
# ---------------------------------------------------------------------------

#: Agree to the last eight decimal places except a single cent-of-a-cent
#: difference. The true difference is exactly ``-0.00000001``; ambient
#: precision 28 (Python's ordinary default) already rounds this away to
#: zero, and low ambient precision rounds it away even more aggressively.
_NEAR_CANCELLING_OCF = "100000000000000000000.00000000"
_NEAR_CANCELLING_CAPEX = "100000000000000000000.00000001"


@pytest.mark.parametrize("ambient_precision", [1, 6, 28, 100])
def test_annual_free_cash_flow_precision_survives_every_ambient_context(
    sec_config,
    ambient_precision: int,
) -> None:
    facts = _facts(
        operating_cash_flow=_NEAR_CANCELLING_OCF,
        capital_expenditure=_NEAR_CANCELLING_CAPEX,
    )
    with localcontext() as ambient:
        ambient.prec = ambient_precision
        payload = _build(sec_config, facts=facts)

    solvency = payload["solvency"]
    assert solvency["periods"]["duration_basis"] == "annual"
    assert solvency["inputs"]["free_cash_flow"] == "-0.00000001"
    assert solvency["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert REASON_NEGATIVE_FREE_CASH_FLOW in solvency["reasons"]


@pytest.mark.parametrize("ambient_precision", [1, 6, 28, 100])
def test_ttm_free_cash_flow_precision_survives_every_ambient_context(
    sec_config,
    ambient_precision: int,
) -> None:
    """The TTM path sums four quarters before the FCF subtraction ever runs.

    Three quarters have identical OCF/capex (cancelling to exactly zero) and
    the fourth carries the same one-cent-of-a-cent difference as the annual
    case, so the *summation* itself -- not only the final subtraction -- must
    survive ambient precision 28 without rounding the difference away.
    """
    quarters = [
        (date(2025, 4, 1), date(2025, 6, 30)),
        (date(2025, 7, 1), date(2025, 9, 30)),
        (date(2025, 10, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 3, 31)),
    ]
    capex_by_quarter = [
        "50000000000000000000.00000000",
        "50000000000000000000.00000000",
        "50000000000000000000.00000000",
        "50000000000000000000.00000001",
    ]
    facts = _facts(operating_cash_flow=None, capital_expenditure=None)
    asset = _asset()
    for index, ((start, end), capex_value) in enumerate(
        zip(quarters, capex_by_quarter, strict=True)
    ):
        facts.append(
            _fact(
                "operating_cash_flow",
                "50000000000000000000.00000000",
                asset=asset,
                period_start=start,
                period_end=end,
                accession=f"0000000000-25-q{index}o",
                fact_key=f"precision-ocf-q{index}",
            )
        )
        facts.append(
            _fact(
                "capital_expenditure",
                capex_value,
                asset=asset,
                period_start=start,
                period_end=end,
                accession=f"0000000000-25-q{index}c",
                fact_key=f"precision-capex-q{index}",
            )
        )

    with localcontext() as ambient:
        ambient.prec = ambient_precision
        payload = _build(sec_config, facts=facts, target_date=date(2026, 4, 30))

    solvency = payload["solvency"]
    assert solvency["periods"]["duration_basis"] == "ttm"
    assert solvency["inputs"]["free_cash_flow"] == "-0.00000001"
    assert solvency["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert REASON_NEGATIVE_FREE_CASH_FLOW in solvency["reasons"]


@pytest.mark.parametrize(
    ("ocf_value", "capex_value"),
    [
        ("Infinity", "Infinity"),
        ("Infinity", "-Infinity"),
        ("NaN", "NaN"),
        ("NaN", "5.00000000"),
        ("Infinity", "NaN"),
    ],
)
def test_nonfinite_flow_operands_are_withheld_without_raising(
    sec_config,
    ocf_value: str,
    capex_value: str,
) -> None:
    """`Infinity - Infinity` (and other nonfinite pairs) must never raise.

    ``Infinity`` combined with ``Infinity``/``-Infinity`` genuinely raises
    `decimal.InvalidOperation` inside the frozen builder's unguarded
    subtraction unless it is preflighted first; every other nonfinite
    combination here is a defense-in-depth case for the same explicit
    withhold. None may raise, and none may report a favorable state.
    """
    facts = _facts(operating_cash_flow=ocf_value, capital_expenditure=capex_value)

    payload = _build(sec_config, facts=facts)

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_NONFINITE_INPUT in solvency["reasons"]
    assert solvency["inputs"]["free_cash_flow"] is None


def test_nonfinite_latest_vintage_is_never_replaced_by_an_older_favorable_one(
    sec_config,
) -> None:
    """The latest (proven) vintage being unusable must not resurrect an older one.

    An older, finite, seemingly favorable operating-cash-flow observation
    exists for the exact same period and accession; a *proven* correction
    replaces it with `Infinity`. Filtering the non-finite fact out of the
    admitted set and letting the frozen vintage selector fall through to the
    older observation would silently substitute favorable-looking evidence
    for evidence that is actually unusable -- so the state must stay
    insufficient, the favorable value must never surface, and both vintages
    must remain in the assessed evidence.
    """
    asset = _asset("shared-ocf-asset", retrieved_at=DATA_CUTOFF - timedelta(days=2))
    facts = _facts(operating_cash_flow=None)
    favorable = _fact(
        "operating_cash_flow",
        "500.00000000",
        asset=asset,
        period_start=ANNUAL_PERIOD[0],
        period_end=ANNUAL_PERIOD[1],
        accession="0000000000-25-ocffix",
        source_revision=1,
        fact_key="nonfinite-fallback-ocf-rev1",
    )
    nonfinite = _fact(
        "operating_cash_flow",
        "Infinity",
        asset=asset,
        period_start=ANNUAL_PERIOD[0],
        period_end=ANNUAL_PERIOD[1],
        accession="0000000000-25-ocffix",
        source_revision=2,
        availability_basis=CORRECTION_AVAILABILITY_BASIS,
        available_at=DATA_CUTOFF - timedelta(days=1),
        fact_key="nonfinite-fallback-ocf-rev2",
    )

    payload = _build(sec_config, facts=[*facts, favorable, nonfinite])

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_NONFINITE_INPUT in solvency["reasons"]
    assert solvency["inputs"]["free_cash_flow"] is None
    assert "500.00000000" not in canonical_json(payload)
    assert str(nonfinite.pk) in solvency["assessed_fact_ids"]
    assert str(favorable.pk) in solvency["assessed_fact_ids"]


# ---------------------------------------------------------------------------
# SEC compatibility, freshness, and availability
# ---------------------------------------------------------------------------


def test_mismatched_instant_dates_are_refused_without_backward_search(sec_config) -> None:
    facts = [
        *_facts(current_assets=None, current_liabilities=None),
        _fact("current_assets", "2000.00000000", asset=_asset(), period_end=date(2025, 9, 30)),
        _fact(
            "current_liabilities",
            "1000.00000000",
            asset=_asset(),
            period_end=date(2025, 9, 30),
        ),
    ]

    payload = _build(sec_config, facts=facts)

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_INSTANT_PERIOD_MISMATCH in solvency["reasons"]
    assert solvency["periods"]["instant_date"] is None
    assert all(
        solvency["inputs"][key] is None
        for key in (
            "cash_and_equivalents",
            "near_term_debt",
            "current_assets",
            "current_liabilities",
        )
    )


def test_a_missing_concept_never_hides_incompatible_surviving_current_assets_and_liabilities(
    sec_config,
) -> None:
    """Reproduction 1: a missing debt component must not let mismatched survivors combine.

    Short-term debt is missing entirely; current assets (2025-09-30) and
    current liabilities (2025-12-31) are both present but from different
    balance-sheet dates. Returning early at the missing-debt check before
    ever validating the surviving pair's own compatibility would let
    `current_ratio` combine two different periods and would never record
    `instant_period_mismatch` at all -- both defects a prior version of this
    builder actually had.
    """
    facts = [
        *_facts(short_term_debt=None, current_assets=None, current_liabilities=None),
        _fact("current_assets", "2000.00000000", asset=_asset(), period_end=date(2025, 9, 30)),
        _fact(
            "current_liabilities",
            "1000.00000000",
            asset=_asset(),
            period_end=date(2025, 12, 31),
        ),
    ]

    first = _build(sec_config, facts=facts)
    second = _build(sec_config, facts=facts)

    solvency = first["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    # Both insufficiency reasons survive together: the missing concept never
    # suppresses the mismatch among the concepts that did survive.
    assert REASON_INSTANT_PERIOD_MISMATCH in solvency["reasons"]
    assert REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING in solvency["reasons"]
    # The incompatible pair can never supply a cross-period current_ratio.
    assert solvency["inputs"]["current_ratio"] is None
    assert solvency["inputs"]["current_assets"] is None
    assert solvency["inputs"]["current_liabilities"] is None
    assert solvency["inputs"]["near_term_debt"] is None
    assert solvency["periods"]["instant_date"] is None
    # Complete resolved-input mapping and assessed lineage survive: every
    # concept still has an entry, and every supplied fact stays referenced.
    assert set(solvency["inputs"]) == {
        "cash_and_equivalents",
        "near_term_debt",
        "current_assets",
        "current_liabilities",
        "current_ratio",
        "free_cash_flow",
    }
    assert len(solvency["assessed_fact_ids"]) == len(facts)
    # Deterministic payload/hash.
    assert canonical_json(first) == canonical_json(second)
    assert first["assessment_hash"] == second["assessment_hash"]


def test_a_missing_concept_never_hides_incompatible_surviving_debt_components(sec_config) -> None:
    """Reproduction 2: a missing instant must not let mismatched debt components combine.

    Current assets is missing entirely; short-term debt (2025-09-30) and the
    current portion of long-term debt (2025-12-31) are both present but from
    different balance-sheet dates. A prior version of this builder returned
    early at the missing-current-assets check before ever validating the
    surviving debt components' own compatibility, incorrectly reporting
    their cross-period sum as usable `near_term_debt`.
    """
    facts = [
        *_facts(current_assets=None, short_term_debt=None, current_long_term_debt=None),
        _fact("short_term_debt", "100.00000000", asset=_asset(), period_end=date(2025, 9, 30)),
        _fact(
            "current_long_term_debt",
            "50.00000000",
            asset=_asset(),
            period_end=date(2025, 12, 31),
        ),
    ]

    first = _build(sec_config, facts=facts)
    second = _build(sec_config, facts=facts)

    solvency = first["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_INSTANT_PERIOD_MISMATCH in solvency["reasons"]
    assert f"{'current_assets'}_missing" in solvency["reasons"]
    # The incompatible debt components can never supply a cross-period sum.
    assert solvency["inputs"]["near_term_debt"] is None
    assert solvency["inputs"]["current_assets"] is None
    assert solvency["periods"]["instant_date"] is None
    assert set(solvency["inputs"]) == {
        "cash_and_equivalents",
        "near_term_debt",
        "current_assets",
        "current_liabilities",
        "current_ratio",
        "free_cash_flow",
    }
    assert len(solvency["assessed_fact_ids"]) == len(facts)
    assert canonical_json(first) == canonical_json(second)
    assert first["assessment_hash"] == second["assessment_hash"]


def test_a_missing_debt_component_with_compatible_survivors_keeps_current_ratio_and_runway(
    sec_config,
) -> None:
    """Positive partial-evidence control: no false mismatch when survivors actually agree.

    Short-term debt is missing, but every other instant shares the same
    balance-sheet date. There is no genuine disagreement among the survivors
    -- only an absence of a different, unrelated concept -- so
    `current_ratio` (an explicitly allowed reported diagnostic) stays
    available, `instant_period_mismatch` must never appear, and the
    independent negative-FCF cash-runway diagnostic remains correctly
    reported from its own compatible evidence.
    """
    payload = _build(
        sec_config,
        facts=_facts(
            short_term_debt=None,
            operating_cash_flow="-100.00000000",
            capital_expenditure="0.00000000",
        ),
    )

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert solvency["reasons"] == [REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING]
    assert REASON_INSTANT_PERIOD_MISMATCH not in solvency["reasons"]
    assert solvency["inputs"]["current_ratio"] == "2.0000"
    assert solvency["inputs"]["cash_and_equivalents"] == "1000.00000000"
    assert solvency["runway"] == {
        "status": "computed",
        "quarters": "40.0000",
        "reason": None,
    }


@pytest.mark.parametrize(
    ("age_days", "expected_status"),
    [
        (UNDER10_MAX_METRIC_AGE_DAYS, SOLVENCY_NO_ADVERSE_EVIDENCE),
        (UNDER10_MAX_METRIC_AGE_DAYS + 1, SOLVENCY_INSUFFICIENT_EVIDENCE),
    ],
)
def test_two_hundred_day_metric_age_boundary(
    sec_config,
    age_days: int,
    expected_status: str,
) -> None:
    instant = TARGET_DATE - timedelta(days=age_days)
    payload = _build(
        sec_config,
        facts=_facts(
            instant_date=instant,
            annual_period=(instant - timedelta(days=364), instant),
        ),
    )

    assert payload["solvency"]["status"] == expected_status
    if expected_status == SOLVENCY_INSUFFICIENT_EVIDENCE:
        assert REASON_STALE_METRIC in payload["solvency"]["reasons"]


def test_future_dated_evidence_is_refused(sec_config) -> None:
    instant = TARGET_DATE + timedelta(days=1)
    payload = _build(
        sec_config,
        facts=_facts(
            instant_date=instant,
            annual_period=(instant - timedelta(days=364), instant),
        ),
    )

    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert "future_period_end" in payload["solvency"]["reasons"]


def test_ttm_evidence_takes_precedence_over_annual(sec_config) -> None:
    quarters = [
        (date(2025, 4, 1), date(2025, 6, 30)),
        (date(2025, 7, 1), date(2025, 9, 30)),
        (date(2025, 10, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 3, 31)),
    ]
    facts = _facts(operating_cash_flow=None, capital_expenditure=None)
    asset = _asset()
    for index, (start, end) in enumerate(quarters):
        facts.append(
            _fact(
                "operating_cash_flow",
                "25.00000000",
                asset=asset,
                period_start=start,
                period_end=end,
                accession=f"0000000000-25-q{index}o",
                fact_key=f"ocf-q{index}",
            )
        )
        facts.append(
            _fact(
                "capital_expenditure",
                "5.00000000",
                asset=asset,
                period_start=start,
                period_end=end,
                accession=f"0000000000-25-q{index}c",
                fact_key=f"capex-q{index}",
            )
        )

    payload = _build(
        sec_config,
        facts=facts,
        target_date=date(2026, 4, 30),
    )

    solvency = payload["solvency"]
    assert solvency["periods"]["duration_basis"] == "ttm"
    assert solvency["periods"]["duration_end"] == "2026-03-31"
    assert solvency["inputs"]["free_cash_flow"] == "80.00000000"


def test_annual_is_used_only_when_no_compatible_ttm_exists(sec_config) -> None:
    payload = _build(sec_config)

    assert payload["solvency"]["periods"]["duration_basis"] == "annual"


def test_mismatched_operating_and_capex_periods_withhold_free_cash_flow(sec_config) -> None:
    facts = _facts(capital_expenditure=None)
    facts.append(
        _fact(
            "capital_expenditure",
            "100.00000000",
            asset=_asset(),
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            accession="0000000000-24-capex",
            fact_key="capex-2024",
        )
    )

    payload = _build(sec_config, facts=facts)

    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert REASON_FREE_CASH_FLOW_MISSING in payload["solvency"]["reasons"]


def test_unproven_same_accession_correction_is_deferred_but_stays_assessed(sec_config) -> None:
    original_asset = _asset("original", retrieved_at=DATA_CUTOFF - timedelta(days=30))
    correction_asset = _asset("correction", retrieved_at=DATA_CUTOFF + timedelta(days=5))
    facts = _facts(cash=None, asset=original_asset)
    original = _fact(
        "cash_and_equivalents",
        "1000.00000000",
        asset=original_asset,
        accession="0000000000-25-cash",
        source_revision=1,
        fact_key="cash-rev1",
    )
    correction = _fact(
        "cash_and_equivalents",
        "10.00000000",
        asset=correction_asset,
        accession="0000000000-25-cash",
        source_revision=2,
        availability_basis=CORRECTION_AVAILABILITY_BASIS,
        available_at=DATA_CUTOFF + timedelta(days=5),
        fact_key="cash-rev2",
    )

    payload = _build(sec_config, facts=[*facts, original, correction])

    solvency = payload["solvency"]
    assert solvency["inputs"]["cash_and_equivalents"] == "1000.00000000"
    assert str(correction.pk) in solvency["assessed_fact_ids"]
    assert {"id": str(correction_asset.id), "sha256": correction_asset.sha256} in (
        solvency["assessed_assets"]
    )


def test_proven_correction_supplies_the_restated_value(sec_config) -> None:
    asset = _asset("proven", retrieved_at=DATA_CUTOFF - timedelta(days=2))
    facts = _facts(cash=None, asset=asset)
    original = _fact(
        "cash_and_equivalents",
        "1000.00000000",
        asset=asset,
        accession="0000000000-25-cash",
        source_revision=1,
        fact_key="cash-rev1",
    )
    correction = _fact(
        "cash_and_equivalents",
        "25.00000000",
        asset=asset,
        accession="0000000000-25-cash",
        source_revision=2,
        availability_basis=CORRECTION_AVAILABILITY_BASIS,
        available_at=DATA_CUTOFF - timedelta(days=1),
        fact_key="cash-rev2",
    )

    payload = _build(sec_config, facts=[*facts, original, correction])

    assert payload["solvency"]["inputs"]["cash_and_equivalents"] == "25.00000000"


def test_unsafe_on_time_evidence_withholds_without_raising(sec_config) -> None:
    payload = _build(sec_config, evidence_cutoff_safe=False)

    solvency = payload["solvency"]
    assert solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert solvency["reasons"] == [REASON_EVIDENCE_NOT_CUTOFF_SAFE]
    assert solvency["runway"] == {
        "status": "withheld",
        "quarters": None,
        "reason": REASON_EVIDENCE_NOT_CUTOFF_SAFE,
    }
    assert solvency["assessed_fact_ids"]
    assert solvency["assessed_assets"]


def test_missing_company_identity_withholds_without_raising(sec_config) -> None:
    payload = _build(sec_config, company_identity_present=False)

    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert "company_identity_unavailable" in payload["solvency"]["reasons"]


# ---------------------------------------------------------------------------
# Assessed evidence
# ---------------------------------------------------------------------------


def test_assessed_evidence_is_complete_deduplicated_and_reference_only(sec_config) -> None:
    asset = _asset()
    facts = _facts(asset=asset)
    payload = _build(sec_config, facts=facts)

    solvency = payload["solvency"]
    assert solvency["assessed_fact_ids"] == sorted(str(fact.pk) for fact in facts)
    assert solvency["assessed_assets"] == [{"id": str(asset.id), "sha256": asset.sha256}]
    serialized = canonical_json(payload)
    assert asset.relative_path not in serialized
    assert "usage_scope" not in serialized
    assert "source_asset" not in serialized


def test_repeated_and_reordered_evidence_yields_an_identical_payload(sec_config) -> None:
    """The same immutable fact appearing twice must never double-count.

    A caller can legitimately hand over a sequence containing the same
    `FundamentalFact` more than once (e.g. two concatenated querysets). The
    assessed identity set, the payload, and its checksum must all stay
    exactly what they would be for the single occurrence -- proving neither
    `assessed_fact_ids` nor `assessment_hash` depends on how many times, or
    in what order, the caller happened to repeat identical evidence.
    """
    asset = _asset()
    facts = _facts(asset=asset)
    baseline = _build(sec_config, facts=facts)

    duplicated = [*facts, facts[0], facts[-1]]
    with_duplicates = _build(sec_config, facts=duplicated)
    assert with_duplicates == baseline
    assert with_duplicates["assessment_hash"] == baseline["assessment_hash"]
    assert with_duplicates["solvency"]["assessed_fact_ids"] == sorted(
        str(fact.pk) for fact in facts
    )

    reordered = list(reversed(facts))
    with_reordered = _build(sec_config, facts=reordered)
    assert with_reordered == baseline

    shuffled_and_duplicated = [*reversed(facts), *facts]
    with_both = _build(sec_config, facts=shuffled_and_duplicated)
    assert with_both == baseline


def test_irrelevant_concepts_are_not_assessed(sec_config) -> None:
    facts = _facts()
    unrelated_asset = _asset("unrelated")
    unrelated = _fact(
        "operating_cash_flow",
        "1.00000000",
        asset=unrelated_asset,
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        fact_key="unrelated",
    )
    unrelated.concept = "revenue"

    payload = _build(sec_config, facts=[*facts, unrelated])

    assert str(unrelated.pk) not in payload["solvency"]["assessed_fact_ids"]
    assert {"id": str(unrelated_asset.id), "sha256": unrelated_asset.sha256} not in (
        payload["solvency"]["assessed_assets"]
    )


@pytest.mark.parametrize(
    ("fact_provider", "asset_provider"),
    [
        pytest.param("other_provider", "other_provider", id="foreign-fact-and-asset"),
        pytest.param(SEC_PROVIDER, "other_provider", id="sec-fact-foreign-asset"),
        pytest.param("other_provider", SEC_PROVIDER, id="foreign-fact-sec-asset"),
    ],
)
def test_builder_uses_only_provider_qualified_sec_facts_for_calculation_and_lineage(
    sec_config,
    fact_provider: str,
    asset_provider: str,
) -> None:
    """Foreign/mismatched canonical rows cannot alter an SEC assessment.

    The first two cases are the reported provenance defect. The third is the
    mutation control proving the fact-provider half of the conjunction is
    independently required even when the row points at an SEC source asset.
    """
    from stanstock.web.views import _under10_solvency_panel

    sec_facts = _facts()
    unqualified_asset = _asset(
        f"unqualified-{fact_provider}-{asset_provider}",
        provider=asset_provider,
    )
    unqualified_fact = _fact(
        "cash_and_equivalents",
        "999999.00000000",
        asset=unqualified_asset,
        provider=fact_provider,
        fact_key=f"unqualified-{fact_provider}-{asset_provider}",
    )

    sec_only = _build(sec_config, facts=sec_facts)
    mixed = _build(sec_config, facts=[*sec_facts, unqualified_fact])

    assert qualify_under10_sec_facts([*sec_facts, unqualified_fact]) == sorted(
        sec_facts,
        key=lambda fact: str(fact.pk),
    )
    assert str(unqualified_fact.pk) not in mixed["solvency"]["assessed_fact_ids"]
    assert {
        "id": str(unqualified_asset.id),
        "sha256": unqualified_asset.sha256,
    } not in mixed["solvency"]["assessed_assets"]
    assert mixed["solvency"]["assessed_fact_ids"] == sec_only["solvency"]["assessed_fact_ids"]
    assert mixed["solvency"]["assessed_assets"] == sec_only["solvency"]["assessed_assets"]

    calculation_keys = ("status", "inputs", "runway")
    sec_calculation = {key: sec_only["solvency"][key] for key in calculation_keys}
    mixed_calculation = {key: mixed["solvency"][key] for key in calculation_keys}
    assert canonical_json(mixed_calculation).encode("utf-8") == canonical_json(
        sec_calculation
    ).encode("utf-8")
    # Since the unqualified row enters no payload surface, even the canonical
    # assessment bytes and checksum remain identical to SEC-only evidence.
    assert canonical_json(mixed).encode("utf-8") == canonical_json(sec_only).encode("utf-8")

    reader = _under10_solvency_panel(mixed["solvency"])
    assert reader["assessed_fact_count"] == len(sec_facts)


def test_rejected_evidence_is_never_labeled_selected(sec_config) -> None:
    payload = _build(sec_config, facts=_facts(cash=None))

    assert "selected" not in canonical_json(payload)
    assert payload["solvency"]["assessed_fact_ids"]


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------


def test_liquidity_is_computed_over_252_sessions_with_unverified_volume_basis(
    sec_config,
) -> None:
    payload = _build(sec_config)

    liquidity = payload["liquidity"]
    assert liquidity["status"] == "computed"
    assert liquidity["metric"] == "median_dollar_volume_252_sessions"
    assert liquidity["value"] == 4_000_000.0
    assert liquidity["currency"] == "USD"
    assert liquidity["sessions_used"] == 252
    assert liquidity["last_session"] == "2026-03-02"
    assert liquidity["basis"] == {
        "interval": "1day",
        "adjustment": "splits",
        "return_definition": "split_adjusted_price_return",
        "volume_basis": "provider_reported_unverified_split_basis",
    }
    assert liquidity["reason"] is None


def test_zero_median_dollar_volume_is_reported_as_zero(sec_config) -> None:
    payload = _build(sec_config, price_frame=_price_frame(volume=0.0))

    assert payload["liquidity"]["status"] == "computed"
    assert payload["liquidity"]["value"] == 0.0
    assert payload["gates"]["dollar_liquidity_252"] is False


def test_insufficient_sessions_withhold_liquidity(sec_config) -> None:
    payload = _build(sec_config, price_frame=_price_frame(sessions=251))

    liquidity = payload["liquidity"]
    assert liquidity["status"] == "withheld"
    assert liquidity["value"] is None
    assert liquidity["reason"] == "insufficient_sessions"
    assert liquidity["sessions_used"] == 251


def test_missing_price_anchor_withholds_liquidity_with_empty_basis(sec_config) -> None:
    payload = _build(sec_config, price_asset=None, price_source=None)

    liquidity = payload["liquidity"]
    assert liquidity["status"] == "withheld"
    assert liquidity["reason"] == "price_provenance_unavailable"
    assert liquidity["price_asset"] is None
    assert liquidity["basis"] == {
        "interval": None,
        "adjustment": None,
        "return_definition": None,
        "volume_basis": "provider_reported_unverified_split_basis",
    }


def test_mismatched_price_anchor_withholds_liquidity(sec_config) -> None:
    payload = _build(sec_config, price_source={"asset_id": str(uuid4())})

    assert payload["liquidity"]["reason"] == "price_provenance_unavailable"


@pytest.mark.parametrize(
    ("metadata", "expected_basis"),
    [
        (
            {},
            {"interval": None, "adjustment": None, "return_definition": None},
        ),
        (
            {
                "interval": "1week",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "currency": "USD",
            },
            {
                "interval": "1week",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
            },
        ),
        (
            {
                "interval": "1day",
                "adjustment": "all",
                "return_definition": "split_adjusted_price_return",
                "currency": "USD",
            },
            {
                "interval": "1day",
                "adjustment": "all",
                "return_definition": "split_adjusted_price_return",
            },
        ),
        (
            {
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "total_return",
                "currency": "USD",
            },
            {"interval": "1day", "adjustment": "splits", "return_definition": "total_return"},
        ),
        (
            {
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "currency": "EUR",
            },
            {
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
            },
        ),
        (
            {
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "currency": "USD",
            },
            {
                "interval": None,
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
            },
        ),
    ],
)
def test_incompatible_basis_metadata_preserves_what_was_observed(
    sec_config,
    metadata: dict[str, Any],
    expected_basis: dict[str, Any],
) -> None:
    """Incompatible-but-present metadata is reported, never nulled to look absent.

    An empty metadata dict (the first case) still reports every field as
    `None`, because nothing was genuinely observed there -- the fix is about
    not *erasing* an observation, not about inventing one. Every other case
    has at least one field that was actually read off the asset and must
    survive into the withheld payload exactly as observed, incompatible or
    not.
    """
    asset = _price_asset(metadata=metadata)

    payload = _build(sec_config, price_asset=asset, price_source={"asset_id": str(asset.id)})

    liquidity = payload["liquidity"]
    assert liquidity["status"] == "withheld"
    assert liquidity["reason"] == "basis_incompatible"
    assert liquidity["value"] is None
    assert liquidity["basis"]["interval"] == expected_basis["interval"]
    assert liquidity["basis"]["adjustment"] == expected_basis["adjustment"]
    assert liquidity["basis"]["return_definition"] == expected_basis["return_definition"]
    assert liquidity["basis"]["volume_basis"] == "provider_reported_unverified_split_basis"


@pytest.mark.parametrize(
    ("stale_days", "expected"),
    [(7, "computed"), (8, "withheld")],
)
def test_price_staleness_boundary(sec_config, stale_days: int, expected: str) -> None:
    payload = _build(
        sec_config,
        price_frame=_price_frame(last_session=TARGET_DATE - timedelta(days=stale_days)),
    )

    assert payload["liquidity"]["status"] == expected
    if expected == "withheld":
        assert payload["liquidity"]["reason"] == "stale_price_evidence"


def test_a_session_after_the_target_date_is_refused(sec_config) -> None:
    payload = _build(
        sec_config,
        price_frame=_price_frame(last_session=TARGET_DATE + timedelta(days=1)),
    )

    assert payload["liquidity"]["status"] == "withheld"
    assert payload["liquidity"]["reason"] == "future_price_session"


def test_liquidity_is_split_equivalent(sec_config) -> None:
    original = _build(sec_config, price_frame=_price_frame(close=40.0, volume=100_000.0))
    split = _build(sec_config, price_frame=_price_frame(close=4.0, volume=1_000_000.0))

    assert original["liquidity"]["value"] == split["liquidity"]["value"]


# ---------------------------------------------------------------------------
# Split verification
# ---------------------------------------------------------------------------


def test_recorded_basic_plan_reports_plan_entitlement_refusal(sec_config) -> None:
    payload = _build(sec_config, provider="twelve_data", provider_plan="Basic")

    assert payload["split_verification"] == {
        "status": "unavailable",
        "reason": "provider_plan_not_entitled",
        "provider": "twelve_data",
        "plan_recorded": True,
        "capability": "corporate_actions_splits",
        "inference_prohibited": True,
    }
    assert payload["blocking_reasons"] == ["provider_plan_not_entitled"]


@pytest.mark.parametrize(
    ("provider", "plan", "plan_recorded"),
    [
        ("twelve_data", None, False),
        ("twelve_data", "", False),
        ("twelve_data", "pro", True),
        ("synthetic_demo", None, False),
    ],
)
def test_other_providers_and_plans_report_the_generic_refusal(
    sec_config,
    provider: str,
    plan: str | None,
    plan_recorded: bool,
) -> None:
    payload = _build(sec_config, provider=provider, provider_plan=plan)

    assert payload["split_verification"]["reason"] == "no_reviewed_corporate_actions_source"
    assert payload["split_verification"]["provider"] == provider
    assert payload["split_verification"]["plan_recorded"] is plan_recorded
    assert payload["blocking_reasons"] == ["no_reviewed_corporate_actions_source"]


def test_raw_plan_text_is_never_persisted(sec_config) -> None:
    payload = _build(sec_config, provider="twelve_data", provider_plan="basic")

    serialized = canonical_json(payload)
    assert '"plan"' not in serialized
    assert "basic" not in serialized.replace("no_reviewed_corporate_actions_source", "")


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_builder_performs_no_database_query(sec_config) -> None:
    with CaptureQueriesContext(connection) as captured:
        _build(sec_config)

    assert list(captured.captured_queries) == []


def test_builder_is_deterministic_for_identical_inputs(sec_config) -> None:
    first = _build(sec_config)
    second = _build(sec_config)

    assert canonical_json(first) == canonical_json(second)
    assert first["assessment_hash"] == second["assessment_hash"]


def test_invalid_session_date_rows_has_no_default_and_must_be_resolved(sec_config) -> None:
    """`invalid_session_date_rows` is a caller-provided fact, not an optimistic default.

    Calling the production `build_under10_assessment` directly (bypassing
    this test file's own `_build` convenience wrapper, which still resolves
    a value on every caller's behalf) without resolving this argument must
    fail loudly with a `TypeError`, not silently behave as though the
    caller's price-frame read had zero invalid session dates.
    """
    asset = _price_asset()
    with pytest.raises(TypeError, match="invalid_session_date_rows"):
        build_under10_assessment(
            facts=_facts(),
            sec_config=sec_config,
            price_frame=_price_frame(),
            price_asset=asset,
            price_source={"asset_id": str(asset.id)},
            reference_close=Decimal("4.250000"),
            target_date=TARGET_DATE,
            data_cutoff=DATA_CUTOFF,
            code_revision_value="0" * 40,
            provider="twelve_data",
            provider_plan="basic",
            evidence_cutoff_safe=True,
            company_identity_present=True,
            listing_id=LISTING_ID,
        )
