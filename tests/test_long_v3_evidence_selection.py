"""`us-sec-long-v3` evidence-selection tests.

Long-v3 changes *evidence selection only*. Every formula weight, bound, cap,
fade path, multiple reversion, peer floor, metric-family rule, freshness
limit, tax proxy, scenario constant, probability withholding, return basis,
and unsupported-SIC policy is identical to long-v2. These tests therefore
pin the frozen v1/v2 config hashes and representative payloads first, then
exercise only the two new capabilities:

1. newest-quarter-anchored homogeneous TTM alias selection;
2. deterministic joint compatible invested-capital pair selection.
"""

from __future__ import annotations

import json
import traceback
from copy import copy, deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from random import shuffle
from typing import Any
from uuid import uuid4

import pytest
import yaml
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test.utils import CaptureQueriesContext

from stanstock.data.asof import AsOfData
from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    Region,
    Security,
)
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.data.sec_fundamentals import (
    CORRECTION_AVAILABILITY_BASIS,
    RESOLUTION_BOUND_OBSERVATION,
    RESOLUTION_LEGACY_ASSET_RETRIEVAL,
    RESOLUTION_UNPROVABLE_LEGACY_ORDERING,
    RESOLUTION_UNPROVABLE_LEGACY_REVERSION,
    TTM_SELECTION_LEGACY,
    TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    build_sec_fundamental_series,
    resolve_availability,
    source_concept_priority,
)
from stanstock.research.evidence_audit import (
    AmbiguousListingSymbolError,
    audit_long_evidence,
)
from stanstock.research.long_forecast_config import (
    REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS,
    LongForecastConfig,
    LongForecastConfigParseError,
    load_long_forecast_config,
    long_forecast_config_hash,
    long_forecast_config_path,
)
from stanstock.research.long_forecasts import (
    CORRECTION_POLICY_PROVEN_OBSERVATION,
    CORRECTION_POLICY_RECORDED_ONLY,
    LONG_FORECAST_CONCEPTS,
    MAX_SAME_DATE_SOURCE_COMBINATIONS,
    audit_invested_capital_pairs,
    build_long_forecasts,
    correction_availability_policy,
    same_date_combination_ceiling,
)

TARGET_DATE = date(2026, 2, 27)
DECISION_TIME = datetime(2026, 3, 1, 12, tzinfo=UTC)

V1_PATH = Path("config/forecasts/us-sec-long-v1.yml")
V2_PATH = Path("config/forecasts/us-sec-long-v2.yml")
V3_PATH = Path("config/forecasts/us-sec-long-v3.yml")

V1_CONFIG_HASH = "ef0e0478aebf53ab605ff47df1d4732ad7cb4e7f3c5d6559b89674ad6a0adeb1"
V2_CONFIG_HASH = "46a81d4bfe87d80ddcf2d62a7f05854eb36381027fb01bc40d637ef3294a5c36"
V3_CONFIG_HASH = "073ac542195b0c67c8ab654aeec61266548ca2758a98ad432f91e48ee5154e57"

REVENUE_PRIMARY = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
REVENUE_SECONDARY = "us-gaap:Revenues"

SOURCE_CONCEPTS = {
    "operating_income": "us-gaap:OperatingIncomeLoss",
    "pretax_income": (
        "us-gaap:"
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
        "NoncontrollingInterest"
    ),
    "income_tax_expense": "us-gaap:IncomeTaxExpenseBenefit",
    "net_income": "us-gaap:NetIncomeLoss",
    "diluted_eps": "us-gaap:EarningsPerShareDiluted",
    "weighted_average_diluted_shares": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
    "operating_cash_flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditure": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "cash_and_equivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "long_term_debt": "us-gaap:LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    "reported_long_term_debt": "us-gaap:LongTermDebt",
    "equity": "us-gaap:StockholdersEquity",
    "revenue": REVENUE_PRIMARY,
}

UNITS = {
    "diluted_eps": "USD/shares",
    "weighted_average_diluted_shares": "shares",
}


# ---------------------------------------------------------------------------
# Frozen v1/v2 contract (pinned before any shared-code behavior is exercised)
# ---------------------------------------------------------------------------


def test_frozen_v1_and_v2_config_hashes_and_capability_absence() -> None:
    v1 = load_long_forecast_config(V1_PATH)
    v2 = load_long_forecast_config(V2_PATH)

    assert long_forecast_config_hash(v1) == V1_CONFIG_HASH
    assert long_forecast_config_hash(v2) == V2_CONFIG_HASH
    for config in (v1, v2):
        assert config.newest_quarter_anchored_homogeneous_ttm_alias_selection is None
        assert config.joint_compatible_invested_capital_pair_selection is None
        assert config.proven_observation_correction_availability is None
        assert config.fundamentals_config_version is None
        assert config.maximum_same_date_source_combinations is None
        for key in (
            "newest_quarter_anchored_homogeneous_ttm_alias_selection",
            "joint_compatible_invested_capital_pair_selection",
            "proven_observation_correction_availability",
            "fundamentals_config_version",
            "maximum_same_date_source_combinations",
        ):
            assert key not in config.raw


def test_v3_is_v2_plus_evidence_selection_only() -> None:
    v2 = load_long_forecast_config(V2_PATH)
    v3 = load_long_forecast_config(V3_PATH)

    assert v3.version == "us-sec-long-v3"
    assert v3.newest_quarter_anchored_homogeneous_ttm_alias_selection is True
    assert v3.joint_compatible_invested_capital_pair_selection is True
    assert v3.proven_observation_correction_availability is True
    assert v3.fundamentals_config_version == "us-sec-fundamentals-v1"
    assert v3.maximum_same_date_source_combinations == 256
    assert v3.adjacent_selected_annual_diluted_share_continuity is True
    assert long_forecast_config_hash(v3) == V3_CONFIG_HASH
    assert long_forecast_config_hash(v3) != long_forecast_config_hash(v2)
    assert long_forecast_config_hash(v3) != V1_CONFIG_HASH

    # Every non-selection term is byte-identical to long-v2.
    frozen_v2 = deepcopy(v2.raw)
    frozen_v3 = deepcopy(v3.raw)
    for key in (
        "version",
        "newest_quarter_anchored_homogeneous_ttm_alias_selection",
        "joint_compatible_invested_capital_pair_selection",
        "proven_observation_correction_availability",
        "fundamentals_config_version",
        "maximum_same_date_source_combinations",
    ):
        frozen_v2.pop(key, None)
        frozen_v3.pop(key, None)
    assert frozen_v3 == frozen_v2

    assert v3.eligibility == v2.eligibility
    assert v3.metric_families == v2.metric_families
    assert v3.growth == v2.growth
    assert v3.peer == v2.peer
    assert v3.horizons == v2.horizons
    assert v3.scenarios == v2.scenarios
    assert v3.return_basis == v2.return_basis
    assert v3.probability_positive_enabled is False


def test_adding_optional_capabilities_cannot_change_a_frozen_hash() -> None:
    """An absent optional key stays out of the effective hash payload."""
    v1 = load_long_forecast_config(V1_PATH)

    assert long_forecast_config_hash(v1) == V1_CONFIG_HASH

    for key in (
        "newest_quarter_anchored_homogeneous_ttm_alias_selection",
        "joint_compatible_invested_capital_pair_selection",
    ):
        mapping = deepcopy(v1.raw)
        mapping[key] = {"enabled": False}
        explicit_false = LongForecastConfig.from_mapping(mapping)
        assert getattr(explicit_false, key) is False
        # An explicit declaration is a different configuration than an absent
        # one, so it must not collide with the frozen hash.
        assert long_forecast_config_hash(explicit_false) != V1_CONFIG_HASH

        for bad_value in (True, "enabled", [True], {}, {"enabled": 1}):
            broken = deepcopy(v1.raw)
            broken[key] = bad_value
            with pytest.raises(ValueError, match=key):
                LongForecastConfig.from_mapping(broken)

    binding = deepcopy(v1.raw)
    binding["fundamentals_config_version"] = "us-sec-fundamentals-v1"
    assert long_forecast_config_hash(LongForecastConfig.from_mapping(binding)) != V1_CONFIG_HASH


@pytest.mark.django_db
def test_v1_and_v2_payloads_stay_free_of_v3_selection_provenance() -> None:
    target = _listing("FROZEN")
    peer = _listing("FROZENP")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    payloads: dict[str, dict[str, Any]] = {}
    for label, path in (("v1", V1_PATH), ("v2", V2_PATH)):
        config = _small_peer_config(load_long_forecast_config(path))
        forecast = build_long_forecasts(
            listings=[target, peer],
            current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
            price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=config,
        )[str(target.pk)]["3y"]
        assert forecast.scenario.base is not None
        assert "evidence_selection" not in forecast.calculation
        assert "invested_capital_selection" not in forecast.calculation["formula_inputs"]
        assert forecast.calculation["method_version"] == f"us-sec-long-{label}"
        payloads[label] = forecast.calculation

    # The frozen versions differ only where v2's declared capability applies.
    assert payloads["v1"]["formula_inputs"]["nopat"] == payloads["v2"]["formula_inputs"]["nopat"]
    assert (
        payloads["v1"]["scenario_paths"]["base"]["cumulative_return"]
        == payloads["v2"]["scenario_paths"]["base"]["cumulative_return"]
    )


@pytest.mark.django_db
def test_legacy_ttm_construction_is_unchanged_for_generic_consumers() -> None:
    listing = _listing("LEGACY")
    _company_evidence(listing, sic="3571")
    sec_config = load_sec_fundamentals_config()
    facts = list(FundamentalFact.objects.filter(company=listing.security.company))

    series = build_sec_fundamental_series(facts, config=sec_config)

    assert series.ttm_selection == TTM_SELECTION_LEGACY
    assert series.ttm_alias_selection == {}
    assert series.ttm["free_cash_flow"].period_start == date(2025, 1, 1)
    assert series.ttm["free_cash_flow"].period_end == date(2025, 12, 31)


# ---------------------------------------------------------------------------
# Capability 1: newest-quarter-anchored homogeneous TTM alias selection
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_newest_quarter_anchor_beats_a_stale_complete_alias() -> None:
    listing = _listing("ANCHOR")
    companyfacts, filing = _sec_assets(listing)
    # Stale but complete alias: four contiguous quarters ending 2025-09-30.
    _revenue_quarters(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_SECONDARY,
        quarters=_quarter_windows(date(2024, 10, 1), 4),
        value=Decimal("100"),
    )
    # Newest alias: four contiguous quarters ending 2025-12-31.
    _revenue_quarters(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_PRIMARY,
        quarters=_quarter_windows(date(2025, 1, 1), 4),
        value=Decimal("110"),
    )

    series = _v3_series(listing)
    selection = series.ttm_alias_selection["revenue"]

    assert series.ttm_selection == TTM_SELECTION_NEWEST_QUARTER_ALIAS
    assert selection["selected_source_concept"] == REVENUE_PRIMARY
    assert selection["newest_quarter_end"] == "2025-12-31"
    assert selection["status"] == "ttm_available"
    assert REVENUE_SECONDARY in selection["stale_complete_alternatives"]
    revenue = series.ttm["revenue"]
    assert revenue.period_start == date(2025, 1, 1)
    assert revenue.period_end == date(2025, 12, 31)
    assert revenue.value == Decimal("540")
    assert revenue.source_concepts == (REVENUE_PRIMARY,)


@pytest.mark.django_db
def test_no_cross_alias_stitching_and_no_annual_current_period_fallback() -> None:
    listing = _listing("STITCH")
    companyfacts, filing = _sec_assets(listing)
    # Complete stale alias covering 2024-10-01..2025-09-30.
    _revenue_quarters(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_SECONDARY,
        quarters=_quarter_windows(date(2024, 10, 1), 4),
        value=Decimal("100"),
    )
    # Newest alias observes only the final quarter; stitching it onto the
    # stale alias's three earlier quarters would be a cross-alias TTM.
    _revenue_quarters(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_PRIMARY,
        quarters=_quarter_windows(date(2025, 10, 1), 1),
        value=Decimal("130"),
    )
    # A full annual period is available and must never substitute for the
    # missing homogeneous quarterly tail.
    _fact(
        listing,
        companyfacts,
        filing,
        concept="revenue",
        value=Decimal("500"),
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        fiscal_period="FY",
        available_at=datetime(2026, 2, 15, tzinfo=UTC),
        accession="STITCH-revenue-FY2025",
    )

    series = _v3_series(listing)
    selection = series.ttm_alias_selection["revenue"]

    assert "revenue" not in series.ttm
    assert selection["status"] == "withheld"
    assert selection["selected_source_concept"] == REVENUE_PRIMARY
    assert selection["newest_quarter_end"] == "2025-12-31"
    assert selection["homogeneous_four_quarter_tail"] is False
    assert "four contiguous compatible quarters" in selection["reason"]
    assert REVENUE_SECONDARY in selection["stale_complete_alternatives"]
    # The legacy path is untouched and still available to frozen consumers.
    assert series.annual["revenue"][-1].period_end == date(2025, 12, 31)


@pytest.mark.django_db
def test_latest_eligible_revision_and_availability_win_within_the_newest_quarter() -> None:
    listing = _listing("REVISION")
    companyfacts, filing = _sec_assets(listing)
    # Both aliases observe the same newest quarter. The lower-priority alias
    # carries the later-available restatement and must anchor selection.
    _revenue_quarters(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_PRIMARY,
        quarters=_quarter_windows(date(2025, 1, 1), 4),
        value=Decimal("100"),
        available_offset_days=0,
    )
    _revenue_quarters(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_SECONDARY,
        quarters=_quarter_windows(date(2025, 1, 1), 4),
        value=Decimal("120"),
        available_offset_days=9,
    )

    selection = _v3_series(listing).ttm_alias_selection["revenue"]

    assert selection["selected_source_concept"] == REVENUE_SECONDARY
    assert selection["status"] == "ttm_available"
    candidates = {item["source_concept"]: item for item in selection["alias_candidates"]}
    assert candidates[REVENUE_PRIMARY]["observes_newest_quarter"] is True
    assert candidates[REVENUE_SECONDARY]["observes_newest_quarter"] is True
    assert (
        candidates[REVENUE_SECONDARY]["newest_quarter_available_at"]
        > candidates[REVENUE_PRIMARY]["newest_quarter_available_at"]
    )


@pytest.mark.django_db
def test_four_quarter_contiguity_span_and_unit_checks_stay_strict() -> None:
    gapped = _listing("GAPPED")
    companyfacts, filing = _sec_assets(gapped)
    windows = _quarter_windows(date(2025, 1, 1), 4)
    _revenue_quarters(
        gapped,
        companyfacts,
        filing,
        source_concept=REVENUE_PRIMARY,
        quarters=[windows[0], windows[2], windows[3]],
        value=Decimal("100"),
    )
    gapped_selection = _v3_series(gapped).ttm_alias_selection["revenue"]
    assert "revenue" not in _v3_series(gapped).ttm
    assert gapped_selection["status"] == "withheld"

    over_span = _listing("OVERSPAN")
    companyfacts, filing = _sec_assets(over_span)
    long_quarters = [
        (date(2025, 1, 1), date(2025, 4, 15)),
        (date(2025, 4, 16), date(2025, 7, 30)),
        (date(2025, 7, 31), date(2025, 11, 13)),
        (date(2025, 11, 14), date(2026, 2, 26)),
    ]
    _revenue_quarters(
        over_span,
        companyfacts,
        filing,
        source_concept=REVENUE_PRIMARY,
        quarters=long_quarters,
        value=Decimal("100"),
    )
    span_series = _v3_series(over_span)
    assert "revenue" not in span_series.ttm
    assert span_series.ttm_alias_selection["revenue"]["status"] == "withheld"

    mixed_unit = _listing("MIXEDUNIT")
    companyfacts, filing = _sec_assets(mixed_unit)
    for index, (start, end) in enumerate(_quarter_windows(date(2025, 1, 1), 4), start=1):
        _fact(
            mixed_unit,
            companyfacts,
            filing,
            concept="revenue",
            value=Decimal("100"),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            available_at=datetime(2026, 2, 10 + index, tzinfo=UTC),
            accession=f"MIXEDUNIT-revenue-Q{index}",
            unit="USD-thousands" if index == 4 else "USD",
        )
    mixed_series = _v3_series(mixed_unit)
    assert "revenue" not in mixed_series.ttm
    assert mixed_series.ttm_alias_selection["revenue"]["status"] == "withheld"


@pytest.mark.django_db
def test_v3_binds_the_reviewed_fundamentals_config_version() -> None:
    target = _listing("BINDING")
    peer = _listing("BINDINGP")
    target_price = _company_evidence(target, sic="3571")
    _company_evidence(peer, sic="3571", scale=1.1)
    config = _small_peer_config(load_long_forecast_config(V3_PATH))
    mismatched = replace(config, fundamentals_config_version="us-sec-fundamentals-v9")

    with pytest.raises(ValueError, match="us-sec-fundamentals-v9"):
        build_long_forecasts(
            listings=[target],
            current_prices={str(target.pk): 50.0},
            price_assets={str(target.pk): target_price},
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=mismatched,
        )


# ---------------------------------------------------------------------------
# Capability 2: deterministic joint compatible invested-capital pair selection
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_joint_selection_finds_a_pair_independent_nearest_dates_miss() -> None:
    target = _listing("JOINT")
    peer = _listing("JOINTP")
    target_price = _company_evidence(
        target,
        sic="3571",
        balance_sheets=(
            # Nearest beginning snapshot uses an incompatible debt basis.
            _balance_sheet(date(2024, 12, 31), debt_concept="long_term_debt"),
            # A compatible beginning snapshot exists three days earlier.
            _balance_sheet(date(2024, 12, 28), debt_concept="reported_long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    listings = [target, peer]
    prices = {str(target.pk): 50.0, str(peer.pk): 55.0}
    assets = {str(target.pk): target_price, str(peer.pk): peer_price}

    v2 = build_long_forecasts(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V2_PATH)),
    )[str(target.pk)]["3y"]
    assert v2.scenario.base is None
    assert v2.scenario.insufficiency_reason == (
        "Beginning/end invested-capital evidence uses incompatible source definitions"
    )

    v3 = build_long_forecasts(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]
    assert v3.scenario.base is not None
    selection = v3.calculation["formula_inputs"]["invested_capital_selection"]
    assert selection["policy"] == "joint_compatible_pair"
    assert selection["selected_beginning_period_end"] == "2024-12-28"
    assert selection["selected_ending_period_end"] == "2025-12-31"
    assert selection["selected_debt_method"] == "reported_long_term_plus_short_term_borrowings"
    assert selection["selected_beginning_fact_ids"]
    assert selection["selected_ending_fact_ids"]
    assert selection["beginning_candidate_period_ends"] == ["2024-12-31", "2024-12-28"]
    assert selection["eligible_pair_count"] == 1
    assert v3.calculation["formula_inputs"]["beginning_invested_capital"]["period_end"] == (
        "2024-12-28"
    )
    assert v3.calculation["evidence_selection"]["invested_capital_selection_policy"] == (
        "joint_compatible_pair"
    )


@pytest.mark.django_db
def test_joint_selection_withholds_when_no_compatible_pair_exists() -> None:
    target = _listing("NOPAIR")
    peer = _listing("NOPAIRP")
    target_price = _company_evidence(
        target,
        sic="3571",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]

    assert forecast.scenario.base is None
    assert forecast.scenario.insufficiency_reason == (
        "No compatible beginning/end invested-capital pair within 7 days of "
        "2024-12-31 and 2025-12-31: every candidate pair uses incompatible "
        "debt-method, debt-component, or source-concept bases"
    )
    assert forecast.calculation["insufficiency_reason"] == (forecast.scenario.insufficiency_reason)
    assert "invested_capital_selection" not in forecast.calculation["formula_inputs"]


@pytest.mark.django_db
def test_joint_selection_tie_is_deterministic() -> None:
    target = _listing("TIE")
    peer = _listing("TIEP")
    target_price = _company_evidence(
        target,
        sic="3571",
        balance_sheets=(
            # Two equidistant, equally recent, equally compatible beginnings.
            _balance_sheet(
                date(2025, 1, 3),
                debt_concept="reported_long_term_debt",
                available_at=datetime(2026, 1, 20, tzinfo=UTC),
            ),
            _balance_sheet(
                date(2024, 12, 28),
                debt_concept="reported_long_term_debt",
                available_at=datetime(2026, 1, 20, tzinfo=UTC),
            ),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    selections = []
    for _ in range(3):
        forecast = build_long_forecasts(
            listings=[target, peer],
            current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
            price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=_small_peer_config(load_long_forecast_config(V3_PATH)),
        )[str(target.pk)]["3y"]
        assert forecast.scenario.base is not None
        selections.append(forecast.calculation["formula_inputs"]["invested_capital_selection"])

    assert selections[0] == selections[1] == selections[2]
    assert selections[0]["eligible_pair_count"] == 2
    # Equal distance, recency, and ending date resolve on the stable earlier
    # beginning period end, never on which pair produces a nicer forecast.
    assert selections[0]["selected_beginning_period_end"] == "2024-12-28"


@pytest.mark.django_db
def test_v3_matches_v2_numbers_when_evidence_selection_is_unambiguous() -> None:
    """Long-v3 changes evidence selection only, never the formula."""
    target = _listing("SAME")
    peer = _listing("SAMEP")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    listings = [target, peer]
    prices = {str(target.pk): 50.0, str(peer.pk): 55.0}
    assets = {str(target.pk): target_price, str(peer.pk): peer_price}

    outputs = {}
    for label, path in (("v2", V2_PATH), ("v3", V3_PATH)):
        outputs[label] = build_long_forecasts(
            listings=listings,
            current_prices=prices,
            price_assets=assets,
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=_small_peer_config(load_long_forecast_config(path)),
        )[str(target.pk)]

    for horizon in ("3y", "5y"):
        v2_forecast = outputs["v2"][horizon]
        v3_forecast = outputs["v3"][horizon]
        assert v2_forecast.scenario.base is not None
        assert v3_forecast.scenario.bear == v2_forecast.scenario.bear
        assert v3_forecast.scenario.base == v2_forecast.scenario.base
        assert v3_forecast.scenario.bull == v2_forecast.scenario.bull
        assert v3_forecast.scenario.confidence == v2_forecast.scenario.confidence
        assert (
            v3_forecast.calculation["scenario_paths"] == v2_forecast.calculation["scenario_paths"]
        )
        assert v3_forecast.calculation["split_basis"] == v2_forecast.calculation["split_basis"]
        assert v3_forecast.calculation["support"] == v2_forecast.calculation["support"]
        assert v3_forecast.calculation["method_version"] == "us-sec-long-v3"
        assert v3_forecast.calculation["config_hash"] != v2_forecast.calculation["config_hash"]

    selection = outputs["v3"]["3y"].calculation["evidence_selection"]
    assert selection["ttm_selection_policy"] == TTM_SELECTION_NEWEST_QUARTER_ALIAS
    assert selection["bound_fundamentals_config_version"] == "us-sec-fundamentals-v1"
    assert (
        selection["ttm_alias_selection"]["operating_cash_flow"]["selected_source_concept"]
        == (SOURCE_CONCEPTS["operating_cash_flow"])
    )
    assert (
        selection["ttm_alias_selection"]["operating_cash_flow"]["homogeneous_four_quarter_tail"]
        is True
    )
    assert "evidence_selection" not in outputs["v2"]["3y"].calculation


@pytest.mark.django_db
def test_joint_selection_prefers_the_more_recent_equidistant_evidence() -> None:
    target = _listing("RECENT")
    peer = _listing("RECENTP")
    target_price = _company_evidence(
        target,
        sic="3571",
        balance_sheets=(
            _balance_sheet(
                date(2024, 12, 28),
                debt_concept="reported_long_term_debt",
                available_at=datetime(2025, 2, 15, tzinfo=UTC),
            ),
            _balance_sheet(
                date(2025, 1, 3),
                debt_concept="reported_long_term_debt",
                available_at=datetime(2026, 1, 20, tzinfo=UTC),
            ),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]

    selection = forecast.calculation["formula_inputs"]["invested_capital_selection"]
    assert selection["eligible_pair_count"] == 2
    assert selection["selected_beginning_period_end"] == "2025-01-03"


# ---------------------------------------------------------------------------
# Offline audit command (RI-4 boundaries, RI-5 listing identity)
# ---------------------------------------------------------------------------


AVAILABLE_THROUGH = DECISION_TIME
AUDIT_ARGS = (
    "--target-date",
    TARGET_DATE.isoformat(),
    "--available-through",
    AVAILABLE_THROUGH.isoformat(),
    "--decision-time",
    DECISION_TIME.isoformat(),
)


@pytest.mark.django_db
def test_audit_command_is_deterministic_read_only_and_provider_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("The offline audit must never contact a provider")

    monkeypatch.setattr(httpx, "Client", _blocked)
    monkeypatch.setattr(httpx, "get", _blocked)

    listing = _listing("AUDIT")
    _company_evidence(
        listing,
        sic="3571",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="long_term_debt"),
            _balance_sheet(date(2024, 12, 28), debt_concept="reported_long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    before = {
        model.__name__: model.objects.count()
        for model in (DataAsset, FundamentalFact, CompanyClassificationObservation, Listing)
    }

    output = StringIO()
    with CaptureQueriesContext(connection) as captured:
        call_command(
            "audit_long_evidence",
            "--listing-ids",
            str(listing.pk),
            "--symbols",
            "MISSING",
            *AUDIT_ARGS,
            "--json",
            stdout=output,
        )
    report = json.loads(output.getvalue())

    for query in captured.captured_queries:
        assert query["sql"].lstrip().upper().startswith("SELECT"), query["sql"]
    after = {
        model.__name__: model.objects.count()
        for model in (DataAsset, FundamentalFact, CompanyClassificationObservation, Listing)
    }
    assert after == before

    assert report["long_forecast_config_version"] == "us-sec-long-v3"
    assert report["fundamentals_config_version"] == "us-sec-fundamentals-v1"
    assert report["ttm_selection_policy"] == TTM_SELECTION_NEWEST_QUARTER_ALIAS
    assert report["target_date"] == TARGET_DATE.isoformat()
    assert report["available_through"] == AVAILABLE_THROUGH.isoformat()
    assert report["decision_time"] == DECISION_TIME.isoformat()
    entries = {
        entry["listing_id"] or entry["requested_symbol"]: entry for entry in report["listings"]
    }
    assert entries["MISSING"]["status"] == "unknown_listing"
    audited = entries[str(listing.pk)]
    assert audited["status"] == "audited"
    assert audited["symbol"] == "AUDIT"
    assert audited["exchange_mic"] == "XNAS"
    concepts = {entry["concept"]: entry for entry in audited["ttm_alias_selection"]}
    assert concepts["operating_cash_flow"]["ttm_available"] is True
    assert concepts["operating_cash_flow"]["homogeneous_four_quarter_tail"] is True
    assert (
        concepts["operating_cash_flow"]["selection"]["selected_source_concept"]
        == (SOURCE_CONCEPTS["operating_cash_flow"])
    )
    controlling = concepts["operating_cash_flow"]["controlling_source_fact"]
    assert controlling is not None
    assert controlling["source_concept"] == SOURCE_CONCEPTS["operating_cash_flow"]
    assert FundamentalFact.objects.filter(pk=controlling["fact_id"]).exists()
    capital = audited["invested_capital"]
    assert capital["status"] == "audited"
    assert capital["compatible_pair_available"] is True
    assert capital["assessment_status"] == "selected_compatible_pair"
    assert capital["independent_nearest_pair_compatible"] is False
    assert capital["joint_selection_recovers_missed_pair"] is True

    repeat = StringIO()
    call_command(
        "audit_long_evidence",
        "--listing-ids",
        str(listing.pk),
        "--symbols",
        "MISSING",
        *AUDIT_ARGS,
        "--json",
        stdout=repeat,
    )
    assert json.loads(repeat.getvalue()) == report

    text = StringIO()
    call_command(
        "audit_long_evidence",
        "--symbols",
        "AUDIT",
        *AUDIT_ARGS,
        stdout=text,
    )
    assert "invested_capital: compatible_pair=True" in text.getvalue()


@pytest.mark.django_db
def test_audit_command_boundaries_are_explicit_and_fail_closed() -> None:
    with pytest.raises(CommandError, match="explicit timezone offset"):
        call_command(
            "audit_long_evidence",
            "--symbols",
            "AUDIT",
            "--target-date",
            TARGET_DATE.isoformat(),
            "--available-through",
            "2026-03-01T12:00:00",
            "--decision-time",
            DECISION_TIME.isoformat(),
            stdout=StringIO(),
        )
    with pytest.raises(CommandError, match="explicit timezone offset"):
        call_command(
            "audit_long_evidence",
            "--symbols",
            "AUDIT",
            "--target-date",
            TARGET_DATE.isoformat(),
            "--available-through",
            AVAILABLE_THROUGH.isoformat(),
            "--decision-time",
            "2026-03-01T12:00:00",
            stdout=StringIO(),
        )
    with pytest.raises(CommandError, match="ISO-8601"):
        call_command(
            "audit_long_evidence",
            "--symbols",
            "AUDIT",
            "--target-date",
            TARGET_DATE.isoformat(),
            "--available-through",
            "not-a-time",
            "--decision-time",
            DECISION_TIME.isoformat(),
            stdout=StringIO(),
        )
    with pytest.raises(CommandError, match="ISO-8601 date"):
        call_command(
            "audit_long_evidence",
            "--symbols",
            "AUDIT",
            "--target-date",
            "2026-02",
            "--available-through",
            AVAILABLE_THROUGH.isoformat(),
            "--decision-time",
            DECISION_TIME.isoformat(),
            stdout=StringIO(),
        )
    with pytest.raises(CommandError, match="at least one of --listing-ids or --symbols"):
        call_command("audit_long_evidence", *AUDIT_ARGS, stdout=StringIO())
    # No boundary may be inferred from another.
    for missing in ("--target-date", "--available-through", "--decision-time"):
        args = list(AUDIT_ARGS)
        index = args.index(missing)
        del args[index : index + 2]
        with pytest.raises(CommandError, match="required"):
            call_command(
                "audit_long_evidence",
                "--symbols",
                "AUDIT",
                *args,
                stdout=StringIO(),
            )


#: Synthetic, credential-shaped canary. It is not a real key and is never
#: sent anywhere; it exists so a leak of the malformed file's contents into a
#: message, traceback, or stream is detectable.
MALFORMED_CONFIG_CANARY = "sk-live-CANARY-0001"

#: Counted to prove a rejected configuration writes nothing.
_AUDIT_PERSISTENT_MODELS = (
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    CompanyClassificationObservation,
    Listing,
)


def _row_counts() -> dict[str, int]:
    return {model.__name__: model.objects.count() for model in _AUDIT_PERSISTENT_MODELS}


@pytest.mark.django_db
def test_malformed_long_config_fails_closed_without_echoing_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A YAML parse failure must never reproduce the offending source line.

    PyYAML quotes the line it choked on. An operator who points
    ``--long-config`` at the wrong file (an environment file, a pasted
    credential fragment) would otherwise have that line echoed into stderr
    and the operator log. The loader normalizes the parse failure and the
    command withholds the parser message entirely, including from the
    chained traceback that ``manage.py --traceback`` would print.
    """
    import httpx

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("A configuration failure must never contact a provider")

    monkeypatch.setattr(httpx, "Client", _blocked)
    monkeypatch.setattr(httpx, "get", _blocked)

    listing = _listing("YAMLBAD")
    _company_evidence(listing, sic="3571")
    before = _row_counts()

    config_path = tmp_path / "malformed-long-config.yml"
    config_path.write_text(
        'schema_version: 1\napi_key: "' + MALFORMED_CONFIG_CANARY + "\nmore: [\n",
        encoding="utf-8",
    )

    # The risk is real, not hypothetical: the raw parser message quotes the
    # credential-shaped line verbatim.
    with pytest.raises(yaml.YAMLError) as parser_error:
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert MALFORMED_CONFIG_CANARY in str(parser_error.value)

    # The loader itself fails closed, so every caller inherits the redaction.
    with pytest.raises(LongForecastConfigParseError) as loader_error:
        load_long_forecast_config(config_path)
    assert MALFORMED_CONFIG_CANARY not in str(loader_error.value)
    assert loader_error.value.__cause__ is None
    assert loader_error.value.__suppress_context__ is True

    stdout = StringIO()
    stderr = StringIO()
    with CaptureQueriesContext(connection) as captured:
        with pytest.raises(CommandError) as raised:
            call_command(
                "audit_long_evidence",
                "--listing-ids",
                str(listing.pk),
                *AUDIT_ARGS,
                "--long-config",
                str(config_path),
                "--json",
                stdout=stdout,
                stderr=stderr,
            )

    message = str(raised.value)
    assert "is not valid YAML" in message
    assert config_path.name in message
    # No chained parser error survives, so no traceback frame can surface it.
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    rendered = "".join(traceback.format_exception(raised.value))
    # Nothing PyYAML said survives anywhere in the raised exception chain.
    for fragment in ("ScannerError", "yaml.scanner", "yaml.safe_load"):
        assert fragment not in rendered
    for parser_line in str(parser_error.value).splitlines():
        stripped = parser_line.strip()
        if len(stripped) >= 10:
            assert stripped not in rendered
    for text in (message, rendered, stdout.getvalue(), stderr.getvalue()):
        assert MALFORMED_CONFIG_CANARY not in text
        assert "api_key" not in text

    # Rejected before any evidence read: no writes, no provider call.
    for query in captured.captured_queries:
        assert query["sql"].lstrip().upper().startswith("SELECT"), query["sql"]
    assert _row_counts() == before


@pytest.mark.django_db
def test_well_formed_long_config_errors_stay_explicit_and_value_free(tmp_path: Path) -> None:
    """Semantic validation keeps its useful wording and still names no value.

    Redacting YAML parse failures must not blunt the ordinary errors: a
    well-formed file that violates the contract still says which key is
    wrong, a non-mapping document still says so, and a missing file still
    reports the path.
    """
    listing = _listing("YAMLOK")
    _company_evidence(listing, sic="3571")
    before = _row_counts()

    def _audit(path: Path) -> CommandError:
        with pytest.raises(CommandError) as raised:
            call_command(
                "audit_long_evidence",
                "--listing-ids",
                str(listing.pk),
                *AUDIT_ARGS,
                "--long-config",
                str(path),
                "--json",
                stdout=StringIO(),
            )
        return raised.value

    semantic = tmp_path / "wrong-schema-version.yml"
    raw = yaml.safe_load(V3_PATH.read_text(encoding="utf-8"))
    raw["schema_version"] = 2
    raw["api_key"] = MALFORMED_CONFIG_CANARY
    semantic.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    semantic_message = str(_audit(semantic))
    assert "schema_version must be 1" in semantic_message
    assert MALFORMED_CONFIG_CANARY not in semantic_message

    sequence = tmp_path / "not-a-mapping.yml"
    sequence.write_text("- " + MALFORMED_CONFIG_CANARY + "\n", encoding="utf-8")
    sequence_message = str(_audit(sequence))
    assert "must be a mapping" in sequence_message
    assert MALFORMED_CONFIG_CANARY not in sequence_message

    absent = tmp_path / "absent-long-config.yml"
    absent_message = str(_audit(absent))
    assert "Could not load" in absent_message
    assert absent.name in absent_message

    assert _row_counts() == before


@pytest.mark.django_db
def test_audit_rejects_incoherent_or_borrowed_boundaries() -> None:
    listing = _listing("BOUNDS")
    _company_evidence(listing, sic="3571")
    config = load_long_forecast_config(V3_PATH)

    with pytest.raises(ValueError, match="cannot be after the decision time"):
        audit_long_evidence(
            listing_ids=(str(listing.pk),),
            target_date=TARGET_DATE,
            available_through=DECISION_TIME + timedelta(days=1),
            decision_time=DECISION_TIME,
            config=config,
        )
    with pytest.raises(ValueError, match="cannot be after the data cutoff date"):
        audit_long_evidence(
            listing_ids=(str(listing.pk),),
            target_date=DECISION_TIME.date() + timedelta(days=1),
            available_through=DECISION_TIME,
            decision_time=DECISION_TIME,
            config=config,
        )
    with pytest.raises(ValueError, match="not the requested audit decision time"):
        audit_long_evidence(
            listing_ids=(str(listing.pk),),
            target_date=TARGET_DATE,
            available_through=DECISION_TIME,
            decision_time=DECISION_TIME,
            config=config,
            asof=AsOfData(DECISION_TIME + timedelta(days=30)),
        )
    with pytest.raises(ValueError, match="at least one listing ID or ticker"):
        audit_long_evidence(
            target_date=TARGET_DATE,
            available_through=DECISION_TIME,
            decision_time=DECISION_TIME,
            config=config,
        )


@pytest.mark.django_db
def test_audit_separates_retrieval_time_from_the_historical_data_cutoff() -> None:
    """Later retrieval of historically available evidence stays admissible."""
    listing = _listing("LATERETR")
    # The companyfacts/filing assets are retrieved long after the historical
    # cutoff, but every fact was already available to the market by then.
    _company_evidence(
        listing,
        sic="3571",
        sec_asset_retrieved_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    config = load_long_forecast_config(V3_PATH)

    # A decision time before the retrieval cannot see the source asset at all.
    unseen = audit_long_evidence(
        listing_ids=(str(listing.pk),),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=config,
    )
    assert unseen["listings"][0]["status"] == "no_visible_facts"

    # A later reader with the same historical cutoff sees the same facts a
    # contemporaneous run would have had, and no later ones.
    reconstructed = audit_long_evidence(
        listing_ids=(str(listing.pk),),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=datetime(2026, 9, 1, tzinfo=UTC),
        config=config,
    )
    entry = reconstructed["listings"][0]
    assert entry["status"] == "audited"
    concepts = {item["concept"]: item for item in entry["ttm_alias_selection"]}
    assert concepts["operating_cash_flow"]["ttm_available"] is True
    assert concepts["operating_cash_flow"]["ttm_period_end"] == "2025-12-31"


@pytest.mark.django_db
def test_audit_excludes_post_cutoff_restatements_and_post_target_quarters() -> None:
    listing = _listing("CUTOFF")
    companyfacts, filing = _sec_assets(listing)
    for index, (start, end) in enumerate(_quarter_windows(date(2025, 1, 1), 4), start=1):
        _fact(
            listing,
            companyfacts,
            filing,
            concept="operating_cash_flow",
            value=Decimal("30"),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            available_at=datetime(2026, 2, 10 + index, tzinfo=UTC),
            accession=f"CUTOFF-ocf-Q{index}",
        )
    # A restatement of the newest quarter that only became available *after*
    # the historical cutoff must never enter the reconstruction.
    _fact(
        listing,
        companyfacts,
        filing,
        concept="operating_cash_flow",
        value=Decimal("999"),
        start=date(2025, 10, 1),
        end=date(2025, 12, 31),
        fiscal_period="Q4",
        available_at=datetime(2026, 4, 1, tzinfo=UTC),
        accession="CUTOFF-restatement",
        source_revision=2,
    )
    # A quarter that ends after the forecast target date is out of window even
    # though a later reader can see it.
    _fact(
        listing,
        companyfacts,
        filing,
        concept="operating_cash_flow",
        value=Decimal("40"),
        start=date(2026, 1, 1),
        end=date(2026, 3, 31),
        fiscal_period="Q1",
        available_at=datetime(2026, 4, 20, tzinfo=UTC),
        accession="CUTOFF-post-target",
    )
    config = load_long_forecast_config(V3_PATH)

    report = audit_long_evidence(
        listing_ids=(str(listing.pk),),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=datetime(2026, 9, 1, tzinfo=UTC),
        config=config,
    )
    entry = report["listings"][0]
    visible = set(entry["visible_fact_ids"])
    restated = FundamentalFact.objects.get(accession="CUTOFF-restatement")
    post_target = FundamentalFact.objects.get(accession="CUTOFF-post-target")

    assert str(restated.pk) not in visible
    assert str(post_target.pk) not in visible
    assert len(visible) == 4
    concepts = {item["concept"]: item for item in entry["ttm_alias_selection"]}
    selection = concepts["operating_cash_flow"]["selection"]
    assert selection["newest_quarter_end"] == "2025-12-31"
    assert selection["controlling_source_fact"]["accession"] == "CUTOFF-ocf-Q4"
    assert selection["controlling_source_fact"]["source_revision"] == 1


@pytest.mark.django_db
def test_audit_and_forecast_select_identical_facts_under_identical_boundaries() -> None:
    target = _listing("PARITY")
    peer = _listing("PARITYP")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    config = _small_peer_config(load_long_forecast_config(V3_PATH))
    reader = AsOfData(DECISION_TIME)

    forecast_fact_ids = sorted(
        str(fact.pk)
        for fact in reader.fundamental_facts_for_companies(
            company_ids=[target.security.company_id],
            concepts=list(LONG_FORECAST_CONCEPTS),
            available_through=DECISION_TIME,
        )
        .filter(provider=config.fundamentals_provider)
        .filter(
            period_end__gte=TARGET_DATE - timedelta(days=4 * 366),
            period_end__lte=TARGET_DATE,
        )
    )
    audit_fact_ids = audit_long_evidence(
        listing_ids=(str(target.pk),),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=config,
        asof=reader,
    )["listings"][0]["visible_fact_ids"]

    assert audit_fact_ids == forecast_fact_ids

    # And the two paths agree on the selected evidence, not just the inputs.
    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=reader,
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(target.pk)]["3y"]
    audited = audit_long_evidence(
        listing_ids=(str(target.pk),),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=config,
        asof=reader,
    )["listings"][0]
    forecast_selection = forecast.calculation["evidence_selection"]["ttm_alias_selection"]
    audited_concepts = {item["concept"]: item for item in audited["ttm_alias_selection"]}
    for concept, selection in forecast_selection.items():
        if concept in audited_concepts:
            assert audited_concepts[concept]["selection"] == selection
    assert (
        audited["invested_capital"]["selected_beginning_fact_ids"]
        == forecast.calculation["formula_inputs"]["invested_capital_selection"][
            "selected_beginning_fact_ids"
        ]
    )


@pytest.mark.django_db
def test_audit_reads_only_evidence_available_at_the_decision_time() -> None:
    listing = _listing("ASOF")
    _company_evidence(listing, sic="3571")
    config = load_long_forecast_config(V3_PATH)

    early = audit_long_evidence(
        listing_ids=(str(listing.pk),),
        target_date=date(2026, 2, 11),
        available_through=datetime(2026, 2, 11, tzinfo=UTC),
        decision_time=datetime(2026, 2, 11, tzinfo=UTC),
        config=config,
    )
    late = audit_long_evidence(
        listing_ids=(str(listing.pk),),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=config,
    )

    early_concepts = {
        entry["concept"]: entry for entry in early["listings"][0]["ttm_alias_selection"]
    }
    late_concepts = {
        entry["concept"]: entry for entry in late["listings"][0]["ttm_alias_selection"]
    }
    assert early_concepts["operating_cash_flow"]["ttm_available"] is False
    assert late_concepts["operating_cash_flow"]["ttm_available"] is True


# ---------------------------------------------------------------------------
# RI-5: a ticker is not a listing identity
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.parametrize("reverse_creation", [False, True])
def test_same_ticker_on_two_exchanges_is_ambiguous(reverse_creation: bool) -> None:
    exchanges = ["XNYS", "XNAS"] if reverse_creation else ["XNAS", "XNYS"]
    listings = [_listing("DUAL", exchange_mic=mic) for mic in exchanges]
    for listing in listings:
        _company_evidence(listing, sic="3571")
    config = load_long_forecast_config(V3_PATH)

    with pytest.raises(AmbiguousListingSymbolError, match="matches 2 listings"):
        audit_long_evidence(
            symbols=("DUAL",),
            target_date=TARGET_DATE,
            available_through=DECISION_TIME,
            decision_time=DECISION_TIME,
            config=config,
        )
    with pytest.raises(CommandError, match="audit by immutable listing ID"):
        call_command(
            "audit_long_evidence",
            "--symbols",
            "DUAL",
            *AUDIT_ARGS,
            stdout=StringIO(),
        )

    # Each immutable listing ID still resolves to exactly its own company.
    for listing in listings:
        report = audit_long_evidence(
            listing_ids=(str(listing.pk),),
            target_date=TARGET_DATE,
            available_through=DECISION_TIME,
            decision_time=DECISION_TIME,
            config=config,
        )
        entry = report["listings"][0]
        assert entry["listing_id"] == str(listing.pk)
        assert entry["exchange_mic"] == listing.exchange_mic
        assert entry["status"] == "audited"
        assert str(listing.security.company_id) not in entry["visible_fact_ids"]
        company_fact_ids = sorted(
            str(fact.pk)
            for fact in FundamentalFact.objects.filter(company=listing.security.company)
        )
        assert set(entry["visible_fact_ids"]) <= set(company_fact_ids)


@pytest.mark.django_db
@pytest.mark.parametrize("reverse_creation", [False, True])
def test_historical_ticker_reuse_on_one_exchange_is_ambiguous(reverse_creation: bool) -> None:
    windows: list[tuple[date, date | None]] = [
        (date(2015, 1, 1), date(2022, 5, 31)),
        (date(2022, 6, 1), None),
    ]
    if reverse_creation:
        windows.reverse()
    listings = [
        _listing("REUSED", valid_from=valid_from, valid_to=valid_to)
        for valid_from, valid_to in windows
    ]
    config = load_long_forecast_config(V3_PATH)

    with pytest.raises(AmbiguousListingSymbolError, match="matches 2 listings"):
        audit_long_evidence(
            symbols=("REUSED",),
            target_date=TARGET_DATE,
            available_through=DECISION_TIME,
            decision_time=DECISION_TIME,
            config=config,
        )
    report = audit_long_evidence(
        listing_ids=tuple(str(listing.pk) for listing in listings),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=config,
    )
    assert [entry["listing_id"] for entry in report["listings"]] == [
        str(listing.pk) for listing in listings
    ]
    assert [entry["valid_from"] for entry in report["listings"]] == [
        valid_from.isoformat() for valid_from, _valid_to in windows
    ]
    assert {entry["status"] for entry in report["listings"]} == {"no_visible_facts"}


@pytest.mark.django_db
def test_unknown_and_malformed_listing_requests_are_explicit() -> None:
    config = load_long_forecast_config(V3_PATH)
    report = audit_long_evidence(
        listing_ids=(str(uuid4()), "not-a-uuid"),
        symbols=("NOSUCH",),
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=config,
    )
    statuses = [entry["status"] for entry in report["listings"]]
    assert statuses == ["unknown_listing", "invalid_listing_id", "unknown_listing"]
    assert all(entry["reason"] for entry in report["listings"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _small_peer_config(config: LongForecastConfig) -> LongForecastConfig:
    return replace(config, peer=replace(config.peer, minimum_peers={4: 1, 3: 1, 2: 1}))


def _v3_series(listing: Listing) -> Any:
    return build_sec_fundamental_series(
        list(FundamentalFact.objects.filter(company=listing.security.company)),
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )


def _quarter_windows(start: date, count: int) -> list[tuple[date, date]]:
    boundaries = [
        (date(2024, 10, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 3, 31)),
        (date(2025, 4, 1), date(2025, 6, 30)),
        (date(2025, 7, 1), date(2025, 9, 30)),
        (date(2025, 10, 1), date(2025, 12, 31)),
    ]
    index = [window[0] for window in boundaries].index(start)
    return boundaries[index : index + count]


def _revenue_quarters(
    listing: Listing,
    companyfacts: DataAsset,
    filing: DataAsset,
    *,
    source_concept: str,
    quarters: list[tuple[date, date]],
    value: Decimal,
    available_offset_days: int = 0,
) -> None:
    alias = source_concept.split(":", maxsplit=1)[1][:12]
    for index, (start, end) in enumerate(quarters, start=1):
        _fact(
            listing,
            companyfacts,
            filing,
            concept="revenue",
            value=value + Decimal(index * 10),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            available_at=datetime(2026, 2, 1, tzinfo=UTC)
            + timedelta(days=index + available_offset_days),
            accession=f"{listing.ticker}-{alias}-{index}",
            source_concept=source_concept,
        )


def _balance_sheet(
    period_end: date,
    *,
    debt_concept: str,
    available_at: datetime | None = None,
) -> tuple[date, str, datetime]:
    return (
        period_end,
        debt_concept,
        available_at or datetime(period_end.year + 1, 2, 15, tzinfo=UTC),
    )


def _listing(
    ticker: str,
    *,
    exchange_mic: str = "XNAS",
    valid_from: date | None = None,
    valid_to: date | None = None,
) -> Listing:
    company = Company.objects.create(name=f"{ticker} {exchange_mic} Company", country="US")
    security = Security.objects.create(company=company, name=f"{ticker} Common")
    return Listing.objects.create(
        security=security,
        ticker=ticker,
        provider_symbol=ticker,
        exchange_mic=exchange_mic,
        currency="USD",
        region=Region.US,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def _company_evidence(
    listing: Listing,
    *,
    sic: str | None,
    scale: float = 1.0,
    sec_asset_retrieved_at: datetime | None = None,
    equity_aliases: dict[date, tuple[str, ...]] | None = None,
    balance_sheets: tuple[tuple[date, str, datetime], ...] = (
        (date(2024, 12, 31), "reported_long_term_debt", datetime(2025, 2, 15, tzinfo=UTC)),
        (date(2025, 12, 31), "reported_long_term_debt", datetime(2026, 2, 15, tzinfo=UTC)),
    ),
) -> DataAsset:
    """Build a complete, eligible FCF/share company with a 2025 TTM window."""
    companyfacts, filing = _sec_assets(listing, retrieved_at=sec_asset_retrieved_at)
    if sic is not None:
        classification_asset = _asset(
            provider="sec",
            kind="sec_submissions",
            subject=f"{listing.ticker}-{listing.exchange_mic}-classification",
            retrieved_at=DECISION_TIME - timedelta(days=1),
        )
        CompanyClassificationObservation.objects.create(
            company=listing.security.company,
            provider="sec",
            scheme="sec_sic",
            code=sic,
            description="Test industry",
            observed_at=DECISION_TIME - timedelta(days=1),
            available_at=DECISION_TIME - timedelta(days=1),
            source_asset=classification_asset,
        )

    annual_periods = (
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 12, 31)),
    )
    shares = 20.0
    for index, (start, end) in enumerate(annual_periods, start=1):
        available = datetime(end.year + 1, 2, 15, tzinfo=UTC)
        _fact(
            listing,
            companyfacts,
            filing,
            concept="weighted_average_diluted_shares",
            value=Decimal(str(shares * scale)),
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{listing.ticker}-shares-{index}",
        )
        free_cash_flow = Decimal(str((80 + index * 15) * scale))
        capex = Decimal(str((20 + index * 2) * scale))
        _fact(
            listing,
            companyfacts,
            filing,
            concept="operating_cash_flow",
            value=free_cash_flow + capex,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{listing.ticker}-ocf-{index}",
        )
        _fact(
            listing,
            companyfacts,
            filing,
            concept="capital_expenditure",
            value=capex,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{listing.ticker}-capex-{index}",
        )
        net_income = Decimal(str((70 + index * 15) * scale))
        _fact(
            listing,
            companyfacts,
            filing,
            concept="net_income",
            value=net_income,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{listing.ticker}-income-{index}",
        )
        _fact(
            listing,
            companyfacts,
            filing,
            concept="diluted_eps",
            value=net_income / Decimal(str(shares * scale)),
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{listing.ticker}-eps-{index}",
        )

    for index, (start, end) in enumerate(_quarter_windows(date(2025, 1, 1), 4), start=1):
        available_at = datetime(2026, 2, 10 + index, tzinfo=UTC)
        values = {
            "weighted_average_diluted_shares": Decimal(str(shares * scale)),
            "operating_income": Decimal(str(25 * scale)),
            "pretax_income": Decimal(str(22.5 * scale)),
            "income_tax_expense": Decimal(str(4.5 * scale)),
            "operating_cash_flow": Decimal(str((28 + index) * scale)),
            "capital_expenditure": Decimal(str((5 + index / 2) * scale)),
        }
        for concept, value in values.items():
            _fact(
                listing,
                companyfacts,
                filing,
                concept=concept,
                value=value,
                start=start,
                end=end,
                fiscal_period=f"Q{index}",
                available_at=available_at,
                accession=f"{listing.ticker}-{concept}-Q{index}",
            )

    for period_end, debt_concept, balance_available_at in balance_sheets:
        for concept, value in (
            (debt_concept, 100.0 + (period_end.year - 2024) * 10),
            ("equity", 400.0 + (period_end.year - 2024) * 50),
            ("cash_and_equivalents", 50.0 + (period_end.year - 2024) * 10),
        ):
            aliases: tuple[str | None, ...] = (None,)
            if concept == "equity" and equity_aliases is not None:
                aliases = equity_aliases.get(period_end, (None,))
            for alias in aliases:
                _fact(
                    listing,
                    companyfacts,
                    filing,
                    concept=concept,
                    value=Decimal(str(value * scale)),
                    start=None,
                    end=period_end,
                    fiscal_period="FY",
                    available_at=balance_available_at,
                    accession=(
                        f"{listing.ticker}-{concept}-{period_end.isoformat()}"
                        if alias is None
                        else f"{listing.ticker}-{concept}-{alias[-12:]}-{period_end.isoformat()}"
                    ),
                    source_concept=alias,
                )

    return _asset(
        provider="twelve_data",
        kind="price_history",
        subject=listing.ticker,
        retrieved_at=DECISION_TIME,
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )


def _sec_assets(
    listing: Listing,
    *,
    retrieved_at: datetime | None = None,
) -> tuple[DataAsset, DataAsset]:
    asset_retrieved_at = retrieved_at or datetime(2026, 2, 1, tzinfo=UTC)
    companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{listing.ticker}-{listing.exchange_mic}-companyfacts",
        retrieved_at=asset_retrieved_at,
    )
    filing = _asset(
        provider="sec",
        kind="sec_submissions",
        subject=f"{listing.ticker}-{listing.exchange_mic}-filing",
        retrieved_at=asset_retrieved_at,
    )
    return companyfacts, filing


def _fact(
    listing: Listing,
    companyfacts: DataAsset,
    filing: DataAsset,
    *,
    concept: str,
    value: Decimal,
    start: date | None,
    end: date,
    fiscal_period: str,
    available_at: datetime,
    accession: str,
    source_revision: int = 1,
    source_concept: str | None = None,
    unit: str | None = None,
    acceptance_at: datetime | None = None,
    availability_basis: str = "acceptance_datetime",
) -> FundamentalFact:
    fact = FundamentalFact.objects.create(
        company=listing.security.company,
        provider="sec",
        concept=concept,
        taxonomy="us-gaap",
        source_concept=source_concept or SOURCE_CONCEPTS[concept],
        value=value,
        unit=unit or UNITS.get(concept, "USD"),
        currency="" if concept == "weighted_average_diluted_shares" else "USD",
        period_type=(
            FundamentalFact.PeriodType.DURATION
            if start is not None
            else FundamentalFact.PeriodType.INSTANT
        ),
        period_start=start,
        period_end=end,
        fiscal_year=end.year,
        fiscal_period=fiscal_period,
        accession=accession,
        filing_form="10-K" if fiscal_period in {"FY", "Q4"} else "10-Q",
        filing_date=(acceptance_at or available_at).date(),
        filed_at=acceptance_at or available_at,
        acceptance_at=acceptance_at or available_at,
        available_at=available_at,
        availability_basis=availability_basis,
        source_revision=source_revision,
        source_asset=companyfacts,
    )
    FundamentalFactEvidence.objects.create(
        fact=fact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=filing,
    )
    return fact


def _asset(
    *,
    provider: str,
    kind: str,
    subject: str,
    retrieved_at: datetime,
    metadata: dict[str, object] | None = None,
) -> DataAsset:
    return DataAsset.objects.create(
        provider=provider,
        kind=kind,
        subject=subject[:120],
        relative_path=f"tests/long-v3/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        metadata=metadata or {},
    )


def test_v3_config_path_helper_resolves_pinned_versions() -> None:
    assert long_forecast_config_path("us-sec-long-v3").name == "us-sec-long-v3.yml"
    assert long_forecast_config_path("us-sec-long-v3").exists()


# ---------------------------------------------------------------------------
# RI-1: a derived quarter's rank belongs to one real controlling source fact
# ---------------------------------------------------------------------------


EQUITY_PRIMARY = "us-gaap:StockholdersEquity"
EQUITY_ALTERNATE = "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
NET_INCOME_PRIMARY = "us-gaap:NetIncomeLoss"
NET_INCOME_ALTERNATE = "us-gaap:ProfitLoss"


def _ytd_revenue_chain(
    listing: Listing,
    companyfacts: DataAsset,
    filing: DataAsset,
    *,
    source_concept: str,
    availability: dict[date, datetime],
    revisions: dict[date, int],
) -> None:
    """Year-to-date revenue facts whose consecutive differences are quarters."""
    for period_end in sorted(availability):
        _fact(
            listing,
            companyfacts,
            filing,
            concept="revenue",
            value=Decimal("100") * Decimal(period_end.month),
            start=date(2025, 1, 1),
            end=period_end,
            fiscal_period="YTD",
            available_at=availability[period_end],
            accession=f"{listing.ticker}-ytd-{period_end.isoformat()}",
            source_concept=source_concept,
            source_revision=revisions[period_end],
        )


def _adversarial_split_lineage(listing: Listing, *, alternate_quarters: int) -> None:
    """Availability and maximum revision deliberately live on different facts.

    The YTD alias's newest quarter is derived from two filings: the nine-month
    filing carries the later availability, while the full-year filing carries
    the higher revision. Independently maximizing those fields would describe
    a vintage that was never filed -- and would beat the competing alias.
    """
    companyfacts, filing = _sec_assets(listing)
    _ytd_revenue_chain(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_PRIMARY,
        availability={
            date(2025, 3, 31): datetime(2026, 2, 5, tzinfo=UTC),
            date(2025, 6, 30): datetime(2026, 2, 6, tzinfo=UTC),
            date(2025, 9, 30): datetime(2026, 2, 18, tzinfo=UTC),
            date(2025, 12, 31): datetime(2026, 2, 14, tzinfo=UTC),
        },
        revisions={
            date(2025, 3, 31): 1,
            date(2025, 6, 30): 1,
            date(2025, 9, 30): 1,
            date(2025, 12, 31): 7,
        },
    )
    windows = _quarter_windows(date(2025, 1, 1), 4)
    alternate_availability = {
        date(2025, 3, 31): datetime(2026, 2, 15, tzinfo=UTC),
        date(2025, 6, 30): datetime(2026, 2, 16, tzinfo=UTC),
        date(2025, 9, 30): datetime(2026, 2, 17, tzinfo=UTC),
        # Deliberately identical to the YTD alias's controlling availability,
        # so the comparison has to fall through to a real revision.
        date(2025, 12, 31): datetime(2026, 2, 18, tzinfo=UTC),
    }
    for index, (start, end) in enumerate(windows[4 - alternate_quarters :], start=1):
        _fact(
            listing,
            companyfacts,
            filing,
            concept="revenue",
            value=Decimal("90") + Decimal(index),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            available_at=alternate_availability[end],
            accession=f"{listing.ticker}-alt-{end.isoformat()}",
            source_concept=REVENUE_SECONDARY,
            source_revision=4,
        )


@pytest.mark.django_db
def test_derived_quarter_rank_uses_one_real_controlling_source_fact() -> None:
    listing = _listing("LINEAGE")
    _adversarial_split_lineage(listing, alternate_quarters=4)

    selection = _v3_series(listing).ttm_alias_selection["revenue"]
    candidates = {item["source_concept"]: item for item in selection["alias_candidates"]}
    ytd = candidates[REVENUE_PRIMARY]
    alternate = candidates[REVENUE_SECONDARY]

    # The YTD alias's newest quarter is a two-filing derivation, and its rank
    # comes from the single filing that gates knowability -- the nine-month
    # one -- not from a revision borrowed off the full-year filing.
    assert ytd["newest_quarter_derivation"] == "ytd_difference"
    assert len(ytd["newest_quarter_source_fact_ids"]) == 2
    controlling = ytd["controlling_source_fact"]
    assert controlling["accession"] == "LINEAGE-ytd-2025-09-30"
    assert controlling["available_at"] == "2026-02-18T00:00:00+00:00"
    assert controlling["source_revision"] == 1
    assert controlling["fact_id"] in ytd["newest_quarter_source_fact_ids"]
    assert FundamentalFact.objects.filter(
        pk=controlling["fact_id"],
        accession=controlling["accession"],
        source_revision=controlling["source_revision"],
        available_at=controlling["available_at"],
    ).exists()

    # Ranking the real filing (rev 1) instead of the synthesized max (rev 7)
    # hands the anchor to the competing alias.
    assert alternate["controlling_source_fact"]["source_revision"] == 4
    assert selection["selected_source_concept"] == REVENUE_SECONDARY
    assert selection["controlling_source_fact"] == alternate["controlling_source_fact"]
    assert selection["status"] == "ttm_available"
    lineage = selection["selected_quarter_lineage"]
    assert [item["period_end"] for item in lineage] == selection["selected_quarter_period_ends"]
    assert {item["derivation"] for item in lineage} == {"reported"}
    for item in lineage:
        assert item["controlling_source_fact_id"] in item["source_fact_ids"]
        assert FundamentalFact.objects.get(
            pk=item["controlling_source_fact_id"]
        ).source_concept == (REVENUE_SECONDARY)


@pytest.mark.django_db
def test_controlling_lineage_withholds_when_the_winning_alias_lacks_a_tail() -> None:
    listing = _listing("LINEAGEW")
    _adversarial_split_lineage(listing, alternate_quarters=1)

    series = _v3_series(listing)
    selection = series.ttm_alias_selection["revenue"]

    assert selection["selected_source_concept"] == REVENUE_SECONDARY
    assert selection["status"] == "withheld"
    assert selection["homogeneous_four_quarter_tail"] is False
    assert selection["selected_quarter_lineage"] == []
    assert "four contiguous compatible quarters" in selection["reason"]
    # The complete YTD alias is recorded as an alternative, never substituted.
    assert selection["stale_complete_alternatives"] == [REVENUE_PRIMARY]
    assert "revenue" not in series.ttm


@pytest.mark.django_db
def test_controlling_lineage_is_deterministic_under_reversed_input_order() -> None:
    listing = _listing("LINEAGED")
    _adversarial_split_lineage(listing, alternate_quarters=4)
    sec_config = load_sec_fundamentals_config()
    facts = list(FundamentalFact.objects.filter(company=listing.security.company))

    forward = build_sec_fundamental_series(
        facts,
        config=sec_config,
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )
    reverse = build_sec_fundamental_series(
        list(reversed(facts)),
        config=sec_config,
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )

    assert forward.ttm_alias_selection == reverse.ttm_alias_selection
    assert forward.ttm["revenue"].value == reverse.ttm["revenue"].value
    assert forward.ttm["revenue"].source_fact_ids == reverse.ttm["revenue"].source_fact_ids


# ---------------------------------------------------------------------------
# RI-2: a compatible pair may exist only through a non-winning same-date alias
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_joint_search_recovers_a_pair_only_a_same_date_alias_can_supply() -> None:
    target = _listing("ALIASIC")
    peer = _listing("ALIASICP")
    target_price = _company_evidence(
        target,
        sic="3571",
        equity_aliases={
            # The beginning date reports equity only under the alternate alias.
            date(2024, 12, 31): (EQUITY_ALTERNATE,),
            # The ending date reports both; the collapsed series keeps only the
            # higher-priority alias, hiding the compatible alternative.
            date(2025, 12, 31): (EQUITY_PRIMARY, EQUITY_ALTERNATE),
        },
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    listings = [target, peer]
    prices = {str(target.pk): 50.0, str(peer.pk): 55.0}
    assets = {str(target.pk): target_price, str(peer.pk): peer_price}

    series = build_sec_fundamental_series(
        list(FundamentalFact.objects.filter(company=target.security.company)),
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
        alias_instant_candidates=True,
    )
    # The collapsed legacy surface really does discard the alternative.
    assert [fact.source_concept for fact in series.instants["equity"]] == [
        EQUITY_ALTERNATE,
        EQUITY_PRIMARY,
    ]
    assert sorted(series.alias_instants["equity"]) == [EQUITY_PRIMARY, EQUITY_ALTERNATE]
    assert [
        fact.period_end.isoformat() for fact in series.alias_instants["equity"][EQUITY_ALTERNATE]
    ] == ["2024-12-31", "2025-12-31"]

    v2 = build_long_forecasts(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V2_PATH)),
    )[str(target.pk)]["3y"]
    assert v2.scenario.base is None
    assert v2.scenario.insufficiency_reason == (
        "Beginning/end invested-capital evidence uses incompatible source definitions"
    )

    v3 = build_long_forecasts(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]

    assert v3.scenario.base is not None
    selection = v3.calculation["formula_inputs"]["invested_capital_selection"]
    assert selection["selected_beginning_period_end"] == "2024-12-31"
    assert selection["selected_ending_period_end"] == "2025-12-31"
    assert {item["concept"]: item["source_concept"] for item in selection["selected_source_basis"]}[
        "equity"
    ] == EQUITY_ALTERNATE
    # The higher-priority ending alias is still enumerated; it simply has no
    # compatible partner at the beginning date.
    assessment = v3.calculation["evidence_selection"]["invested_capital_assessment"]
    ending_equity_aliases = {
        item["source_concept"]
        for candidate in assessment["ending_candidates"]
        for item in candidate["source_basis"]
        if item["concept"] == "equity"
    }
    assert ending_equity_aliases == {EQUITY_PRIMARY, EQUITY_ALTERNATE}
    assert assessment["compatible_pair_count"] == 1


@pytest.mark.django_db
def test_same_date_alias_enumeration_still_withholds_incompatible_alternatives() -> None:
    target = _listing("ALIASNO")
    peer = _listing("ALIASNOP")
    target_price = _company_evidence(
        target,
        sic="3571",
        equity_aliases={
            date(2024, 12, 31): (EQUITY_PRIMARY,),
            date(2025, 12, 31): (EQUITY_ALTERNATE,),
        },
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]

    assert forecast.scenario.base is None
    assessment = forecast.calculation["evidence_selection"]["invested_capital_assessment"]
    assert assessment["status"] == "no_compatible_pair"
    assert assessment["compatible_pair_count"] == 0
    assert assessment["beginning_candidate_count"] >= 1
    assert assessment["ending_candidate_count"] >= 1


@pytest.mark.django_db
def test_same_date_alias_selection_is_deterministic() -> None:
    target = _listing("ALIASDET")
    peer = _listing("ALIASDETP")
    target_price = _company_evidence(
        target,
        sic="3571",
        equity_aliases={
            date(2024, 12, 31): (EQUITY_PRIMARY, EQUITY_ALTERNATE),
            date(2025, 12, 31): (EQUITY_PRIMARY, EQUITY_ALTERNATE),
        },
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    selections = []
    for _ in range(3):
        forecast = build_long_forecasts(
            listings=[target, peer],
            current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
            price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=_small_peer_config(load_long_forecast_config(V3_PATH)),
        )[str(target.pk)]["3y"]
        assert forecast.scenario.base is not None
        selections.append(forecast.calculation["formula_inputs"]["invested_capital_selection"])

    assert selections[0] == selections[1] == selections[2]
    # Two identical-date, identically-available combinations exist on each
    # side; the declared alias priority decides, never a generated fact UUID.
    assert selections[0]["eligible_pair_count"] == 2
    assert {
        item["concept"]: item["source_concept"] for item in selections[0]["selected_source_basis"]
    }["equity"] == EQUITY_PRIMARY


@pytest.mark.django_db
def test_alias_candidate_surface_is_required_before_a_joint_search() -> None:
    """A joint search over the collapsed surface would silently miss pairs."""
    listing = _listing("NOSURFACE")
    _company_evidence(listing, sic="3571")
    series = build_sec_fundamental_series(
        list(FundamentalFact.objects.filter(company=listing.security.company)),
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )

    assert series.alias_instant_candidates is False
    assert series.alias_instants == {}
    with pytest.raises(ValueError, match="without the alias candidate surface"):
        audit_invested_capital_pairs(
            series=series,
            beginning_target=date(2024, 12, 31),
            ending_target=date(2025, 12, 31),
            tolerance_days=7,
            priority=source_concept_priority(load_sec_fundamentals_config()),
            maximum_combinations=REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS,
        )


# ---------------------------------------------------------------------------
# RI-3: a rejected pair search is assessed evidence, not silence
# ---------------------------------------------------------------------------


def _v3_forecast_with_balance_sheets(
    ticker: str,
    *,
    balance_sheets: tuple[tuple[date, str, datetime], ...],
    minimum_peers: dict[int, int] | None = None,
) -> Any:
    target = _listing(ticker)
    peer = _listing(f"{ticker}P")
    target_price = _company_evidence(target, sic="3571", balance_sheets=balance_sheets)
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    config = load_long_forecast_config(V3_PATH)
    config = replace(
        config,
        peer=replace(config.peer, minimum_peers=minimum_peers or {4: 1, 3: 1, 2: 1}),
    )
    return build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(target.pk)]["3y"]


@pytest.mark.django_db
def test_missing_beginning_candidates_are_assessed_not_dropped() -> None:
    forecast = _v3_forecast_with_balance_sheets(
        "NOBEGIN",
        balance_sheets=(
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    assessment = forecast.calculation["evidence_selection"]["invested_capital_assessment"]

    assert forecast.scenario.base is None
    assert assessment["status"] == "missing_beginning_candidates"
    assert assessment["assessment_status"] == "assessed_incompatible_or_unavailable"
    assert assessment["beginning_target_date"] == "2024-12-31"
    assert assessment["ending_target_date"] == "2025-12-31"
    assert assessment["beginning_candidates"] == []
    assert assessment["ending_candidates"]
    assert assessment["compatible_pair_count"] == 0
    assert assessment["rejection_reason"] == forecast.calculation["insufficiency_reason"]
    assert "verified" not in json.dumps(assessment)


@pytest.mark.django_db
def test_missing_ending_candidates_are_assessed_not_dropped() -> None:
    forecast = _v3_forecast_with_balance_sheets(
        "NOEND",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    assessment = forecast.calculation["evidence_selection"]["invested_capital_assessment"]

    assert forecast.scenario.base is None
    assert assessment["status"] == "missing_ending_candidates"
    assert assessment["assessment_status"] == "assessed_incompatible_or_unavailable"
    assert assessment["beginning_candidates"]
    assert assessment["ending_candidates"] == []
    assert assessment["rejection_reason"] == forecast.calculation["insufficiency_reason"]


@pytest.mark.django_db
def test_zero_compatible_pairs_records_every_candidate_and_its_basis() -> None:
    forecast = _v3_forecast_with_balance_sheets(
        "NOPAIR2",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    assessment = forecast.calculation["evidence_selection"]["invested_capital_assessment"]

    assert forecast.scenario.base is None
    assert assessment["status"] == "no_compatible_pair"
    assert assessment["assessment_status"] == "assessed_incompatible_or_unavailable"
    assert assessment["compatible_pair_count"] == 0
    assert assessment["selected_beginning_fact_ids"] == []
    assert assessment["selected_ending_fact_ids"] == []
    beginning = assessment["beginning_candidates"][0]
    ending = assessment["ending_candidates"][0]
    assert beginning["debt_method"] == "sum_non_overlapping_debt_components"
    assert ending["debt_method"] == "reported_long_term_plus_short_term_borrowings"
    assert beginning["fact_ids"] and ending["fact_ids"]
    assert "verified" not in json.dumps(assessment)


@pytest.mark.django_db
def test_a_selected_pair_survives_a_later_peer_insufficiency() -> None:
    forecast = _v3_forecast_with_balance_sheets(
        "PEERGAP",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="reported_long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
        minimum_peers={4: 9, 3: 9, 2: 9},
    )
    assessment = forecast.calculation["evidence_selection"]["invested_capital_assessment"]

    assert forecast.scenario.base is None
    assert "peer set met floors" in forecast.calculation["insufficiency_reason"]
    # The successful pair assessment is not lost by the later withholding.
    assert assessment["status"] == "selected_compatible_pair"
    assert assessment["assessment_status"] == "selected_compatible_pair"
    assert assessment["rejection_reason"] == ""
    assert assessment["selected_beginning_period_end"] == "2024-12-31"
    assert assessment["selected_ending_period_end"] == "2025-12-31"
    assert assessment["selected_beginning_fact_ids"]


@pytest.mark.django_db
def test_a_successful_run_reports_the_same_assessment_it_selected_from() -> None:
    forecast = _v3_forecast_with_balance_sheets(
        "PAIROK",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="reported_long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    assessment = forecast.calculation["evidence_selection"]["invested_capital_assessment"]
    selection = forecast.calculation["formula_inputs"]["invested_capital_selection"]

    assert forecast.scenario.base is not None
    assert assessment["status"] == "selected_compatible_pair"
    assert assessment["compatible_pair_count"] == selection["eligible_pair_count"]
    for key in (
        "selected_beginning_period_end",
        "selected_ending_period_end",
        "selected_debt_method",
        "selected_debt_components",
        "selected_source_basis",
        "selected_beginning_fact_ids",
        "selected_ending_fact_ids",
    ):
        assert assessment[key] == selection[key]


@pytest.mark.django_db
def test_evidence_selection_is_absent_before_any_pair_assessment() -> None:
    """An unassessed pair search is explicit ``None``, never a success shape."""
    target = _listing("NOSIC")
    price_asset = _company_evidence(target, sic=None)

    forecast = build_long_forecasts(
        listings=[target],
        current_prices={str(target.pk): 50.0},
        price_assets={str(target.pk): price_asset},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]

    assert forecast.scenario.base is None
    assert forecast.calculation["evidence_selection"]["invested_capital_assessment"] is None


@pytest.mark.django_db
def test_v3_provenance_is_complete_but_stays_within_a_bounded_payload() -> None:
    """Selection provenance references evidence rather than repeating it."""
    target = _listing("BUDGET")
    peer = _listing("BUDGETP")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    listings = [target, peer]
    prices = {str(target.pk): 50.0, str(peer.pk): 55.0}
    assets = {str(target.pk): target_price, str(peer.pk): peer_price}

    sizes = {}
    payloads = {}
    for label, path in (("v2", V2_PATH), ("v3", V3_PATH)):
        forecast = build_long_forecasts(
            listings=listings,
            current_prices=prices,
            price_assets=assets,
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=_small_peer_config(load_long_forecast_config(path)),
        )[str(target.pk)]["3y"]
        assert forecast.scenario.base is not None
        payloads[label] = forecast.calculation
        sizes[label] = len(json.dumps(forecast.calculation))

    # Provenance growth is bounded; a per-quarter copy of every filing would
    # blow straight through this.
    assert sizes["v3"] < sizes["v2"] * 1.5

    input_fact_ids = {fact["id"] for fact in payloads["v3"]["input_facts"]}
    selection = payloads["v3"]["evidence_selection"]["ttm_alias_selection"]
    for concept_selection in selection.values():
        for item in concept_selection["selected_quarter_lineage"]:
            # Every referenced filing is still fully described in the payload.
            assert item["controlling_source_fact_id"] in input_fact_ids
            assert set(item["source_fact_ids"]) <= input_fact_ids


# ---------------------------------------------------------------------------
# RI re-review 1: within-alias direct-vs-derived quarter rank
# ---------------------------------------------------------------------------


def _direct_revenue_fact(
    listing: Listing,
    companyfacts: DataAsset,
    filing: DataAsset,
    *,
    start: date,
    end: date,
    value: Decimal,
    available_at: datetime,
    accession: str,
    source_concept: str,
    source_revision: int = 1,
) -> FundamentalFact:
    return _fact(
        listing,
        companyfacts,
        filing,
        concept="revenue",
        value=value,
        start=start,
        end=end,
        fiscal_period="Q",
        available_at=available_at,
        accession=accession,
        source_concept=source_concept,
        source_revision=source_revision,
    )


#: Availability every observation at the newest quarter deliberately shares,
#: so the comparison can never be settled by availability alone.
TIED_AVAILABILITY = datetime(2026, 2, 20, tzinfo=UTC)


def _direct_versus_derived_quarter_fixture(listing: Listing) -> None:
    """One complete alias whose newest quarter is contested two ways.

    Within `REVENUE_PRIMARY` the newest quarter exists twice at exactly the
    same availability: a directly reported Q4 filed as revision 1 with a
    late-sorting accession, and a YTD-derived Q4 whose controlling filing is
    revision 5 with an early-sorting accession. Legacy `_quarter_series`
    resolves that by availability only and therefore keeps the revision 1
    direct value.

    `REVENUE_SECONDARY` observes the same newest quarter at the same
    availability under revision 3 but supplies only two quarters, so it can
    never produce a homogeneous tail. If the alias anchor were still decided
    from the revision 1 direct observation, revision 3 would win the anchor
    and the concept would be withheld.
    """
    companyfacts, filing = _sec_assets(listing)
    _ytd_revenue_chain(
        listing,
        companyfacts,
        filing,
        source_concept=REVENUE_PRIMARY,
        availability={
            date(2025, 3, 31): datetime(2026, 2, 5, tzinfo=UTC),
            date(2025, 6, 30): datetime(2026, 2, 6, tzinfo=UTC),
            date(2025, 9, 30): datetime(2026, 2, 7, tzinfo=UTC),
            date(2025, 12, 31): TIED_AVAILABILITY,
        },
        revisions={
            date(2025, 3, 31): 1,
            date(2025, 6, 30): 1,
            date(2025, 9, 30): 1,
            # The controlling filing of the derived Q4: latest availability
            # *and* the highest revision, both on one real filing.
            date(2025, 12, 31): 5,
        },
    )
    # Same alias, same quarter end, same availability, revision 1, and an
    # accession that sorts *after* the YTD filing's -- so an accession-first
    # or availability-only rule would pick this stale direct value.
    _direct_revenue_fact(
        listing,
        companyfacts,
        filing,
        start=date(2025, 10, 1),
        end=date(2025, 12, 31),
        value=Decimal("11"),
        available_at=TIED_AVAILABILITY,
        accession=f"{listing.ticker}-zz-direct-q4",
        source_concept=REVENUE_PRIMARY,
        source_revision=1,
    )
    for index, (start, end) in enumerate(_quarter_windows(date(2025, 10, 1), 1), start=1):
        _direct_revenue_fact(
            listing,
            companyfacts,
            filing,
            start=start,
            end=end,
            value=Decimal("77") + Decimal(index),
            available_at=TIED_AVAILABILITY,
            accession=f"{listing.ticker}-alt-q4",
            source_concept=REVENUE_SECONDARY,
            source_revision=3,
        )
    _direct_revenue_fact(
        listing,
        companyfacts,
        filing,
        start=date(2025, 7, 1),
        end=date(2025, 9, 30),
        value=Decimal("70"),
        available_at=datetime(2026, 2, 8, tzinfo=UTC),
        accession=f"{listing.ticker}-alt-q3",
        source_concept=REVENUE_SECONDARY,
        source_revision=3,
    )


@pytest.mark.django_db
def test_within_alias_direct_and_derived_quarters_rank_by_controlling_fact() -> None:
    listing = _listing("DIRDER")
    _direct_versus_derived_quarter_fixture(listing)

    legacy = build_sec_fundamental_series(
        list(FundamentalFact.objects.filter(company=listing.security.company)),
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_LEGACY,
    )
    series = _v3_series(listing)
    selection = series.ttm_alias_selection["revenue"]
    candidates = {item["source_concept"]: item for item in selection["alias_candidates"]}

    # Frozen legacy construction is untouched: it still collapses the newest
    # quarter to the directly reported revision 1 observation.
    legacy_q4 = [
        value for value in legacy.quarters["revenue"] if value.period_end == date(2025, 12, 31)
    ]
    assert [value.derivation for value in legacy_q4] == ["reported"]

    # v3 retains both observations and ranks them on one real controlling
    # filing, so the revision 5 derived observation wins inside the alias...
    assert candidates[REVENUE_PRIMARY]["newest_quarter_derivation"] == "ytd_difference"
    assert candidates[REVENUE_PRIMARY]["controlling_source_fact"]["source_revision"] == 5
    assert candidates[REVENUE_PRIMARY]["controlling_source_fact"]["available_at"] == (
        TIED_AVAILABILITY.isoformat()
    )
    assert candidates[REVENUE_SECONDARY]["controlling_source_fact"]["source_revision"] == 3
    assert candidates[REVENUE_SECONDARY]["has_homogeneous_four_quarter_tail"] is False

    # ... and that carries the anchor to the only complete alias.
    assert selection["selected_source_concept"] == REVENUE_PRIMARY
    assert selection["status"] == "ttm_available"
    assert selection["homogeneous_four_quarter_tail"] is True
    lineage = {item["period_end"]: item for item in selection["selected_quarter_lineage"]}
    assert lineage["2025-12-31"]["derivation"] == "ytd_difference"
    controlling_id = lineage["2025-12-31"]["controlling_source_fact_id"]
    assert FundamentalFact.objects.get(pk=controlling_id).source_revision == 5


@pytest.mark.django_db
@pytest.mark.parametrize("reverse_input", [False, True])
def test_direct_versus_derived_selection_ignores_input_order(reverse_input: bool) -> None:
    listing = _listing("DIRDERO" if reverse_input else "DIRDERF")
    _direct_versus_derived_quarter_fixture(listing)
    facts = list(FundamentalFact.objects.filter(company=listing.security.company))
    if reverse_input:
        facts = list(reversed(facts))

    series = build_sec_fundamental_series(
        facts,
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )
    selection = series.ttm_alias_selection["revenue"]

    assert selection["selected_source_concept"] == REVENUE_PRIMARY
    assert selection["status"] == "ttm_available"
    assert series.ttm["revenue"].derivation == "sum_four_contiguous_quarters"


@pytest.mark.django_db
def test_distinct_quarter_identities_sharing_an_end_are_ranked_not_overwritten() -> None:
    """Two quarters can end on the same day and still be different periods.

    Legacy quarter construction keys on ``(concept, period end)`` and lets the
    last one processed win. v3 keeps both, ranks them under the controlling
    filing, and -- when even the accession ties -- falls through to the
    observation's own identity rather than to a generated row UUID.
    """
    listing = _listing("SAMEEND")
    companyfacts, filing = _sec_assets(listing)
    for index, (start, end) in enumerate(_quarter_windows(date(2025, 1, 1), 3), start=1):
        _direct_revenue_fact(
            listing,
            companyfacts,
            filing,
            start=start,
            end=end,
            value=Decimal("50") + Decimal(index),
            available_at=datetime(2026, 2, 10 + index, tzinfo=UTC),
            accession=f"{listing.ticker}-q{index}",
            source_concept=REVENUE_PRIMARY,
        )
    # Same end, same alias, same availability, same accession: the two
    # observations differ only by period identity, value, and revision. The
    # contiguous one is the *lower*-sorting period start, so a "last write
    # wins" collapse keeps the wrong one.
    for start, value, revision in (
        (date(2025, 10, 1), Decimal("61"), 4),
        (date(2025, 10, 2), Decimal("62"), 1),
    ):
        _direct_revenue_fact(
            listing,
            companyfacts,
            filing,
            start=start,
            end=date(2025, 12, 31),
            value=value,
            available_at=TIED_AVAILABILITY,
            accession=f"{listing.ticker}-q4",
            source_concept=REVENUE_PRIMARY,
            source_revision=revision,
        )
    identities = set(
        FundamentalFact.objects.filter(
            company=listing.security.company,
            period_end=date(2025, 12, 31),
        ).values_list("period_identity", flat=True)
    )
    assert identities == {
        "duration:2025-10-01:2025-12-31",
        "duration:2025-10-02:2025-12-31",
    }

    facts = list(FundamentalFact.objects.filter(company=listing.security.company))
    forward = build_sec_fundamental_series(
        facts,
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )
    reverse = build_sec_fundamental_series(
        list(reversed(facts)),
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )

    legacy = build_sec_fundamental_series(
        facts,
        config=load_sec_fundamentals_config(),
        ttm_selection=TTM_SELECTION_LEGACY,
    )

    # Frozen legacy construction keeps whichever same-end observation it
    # processed last, which here is the non-contiguous 2 October period, so
    # no legacy TTM window exists at all. That behavior is untouched.
    legacy_q4 = [
        value for value in legacy.quarters["revenue"] if value.period_end == date(2025, 12, 31)
    ]
    assert [value.period_start for value in legacy_q4] == [date(2025, 10, 2)]
    assert "revenue" not in legacy.ttm

    # v3 ranks the two identities instead of overwriting one, so the
    # revision 4 observation wins and the window is contiguous.
    selection = forward.ttm_alias_selection["revenue"]
    assert selection["status"] == "ttm_available"
    lineage = {item["period_end"]: item for item in selection["selected_quarter_lineage"]}
    assert lineage["2025-12-31"]["period_start"] == "2025-10-01"
    assert forward.ttm["revenue"].value == Decimal("61") + Decimal("51") + Decimal("52") + Decimal(
        "53"
    )
    assert forward.ttm_alias_selection == reverse.ttm_alias_selection
    assert forward.ttm["revenue"].value == reverse.ttm["revenue"].value

    # Reassigning row UUIDs cannot move the choice either.
    for _trial in range(8):
        shuffled = build_sec_fundamental_series(
            _with_reassigned_ids(facts),
            config=load_sec_fundamentals_config(),
            ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
        )
        assert shuffled.ttm["revenue"].value == forward.ttm["revenue"].value
        assert shuffled.ttm["revenue"].accessions == forward.ttm["revenue"].accessions


def _with_reassigned_ids(facts: list[FundamentalFact]) -> list[FundamentalFact]:
    """In-memory copies of ``facts`` carrying fresh, shuffled row UUIDs.

    Nothing is written: persisted SEC facts are immutable. This produces an
    alternative *view* of the same evidence whose only difference is the
    generated primary keys, which is exactly what a `us-sec-long-v3`
    selection must be blind to.
    """
    reassigned = [copy(fact) for fact in facts]
    for fact in reassigned:
        fact.id = uuid4()
    shuffle(reassigned)
    return reassigned


# ---------------------------------------------------------------------------
# RI re-review 2: the assessed-evidence manifest must close
# ---------------------------------------------------------------------------


def _referenced_fact_ids(payload: Any) -> set[str]:
    """Independently collect every fact id an assessment payload cites.

    Deliberately written from scratch here rather than importing the
    production walker, so the test proves closure instead of restating the
    implementation.
    """
    found: set[str] = set()

    def walk(node: Any, key: str | None) -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                walk(value, name)
        elif isinstance(node, list):
            if key is not None and key.endswith("fact_ids"):
                found.update(item for item in node if isinstance(item, str))
            else:
                for item in node:
                    walk(item, None)
        elif isinstance(node, str) and key is not None and key.endswith("fact_id"):
            found.add(node)

    walk(payload, None)
    return found


def _rejected_candidate_fact_ids(selection: dict[str, Any]) -> set[str]:
    """Candidate facts the invested-capital search read but did not select.

    Derived independently from the assessment payload -- its candidate lists
    and any refused combination axes, minus whatever it reports as the
    selected pair. It deliberately never consults ``input_facts``,
    ``selected_input_fact_ids``, or the assessed lists it is used to check,
    so it can prove a rejected candidate was not classified as a verified
    formula input.
    """
    assessment = selection["invested_capital_assessment"] or {}
    candidates: set[str] = {
        fact_id
        for side in ("beginning_candidates", "ending_candidates")
        for candidate in assessment.get(side, [])
        for fact_id in candidate["fact_ids"]
    }
    overflow = assessment.get("same_date_combination_overflow") or {}
    for axis in overflow.get("axes", []):
        candidates.update(axis["fact_ids"])
    selected = set(assessment.get("selected_beginning_fact_ids", [])) | set(
        assessment.get("selected_ending_fact_ids", [])
    )
    return candidates - selected


def _assert_manifest_closes(forecast: Any) -> dict[str, Any]:
    """Every referenced fact is classified, described, and asset-provable.

    The selected/assessed split is taken from the payload's own explicit
    lists and cross-checked against an independently derived set of rejected
    candidates, rather than assuming every ``input_facts`` member is a
    verified selection.
    """
    calculation = forecast.calculation
    selection = calculation["evidence_selection"]
    manifest = set(selection["manifest_evidence_fact_ids"])
    assessed = set(selection["assessed_evidence_fact_ids"])
    selected = set(selection["selected_input_fact_ids"])
    verified = {fact["id"] for fact in calculation["input_facts"]}
    described = {entry["id"] for entry in selection["assessed_evidence"]}
    asset_ids = {str(asset.pk) for asset in forecast.source_assets}

    assert selection["assessed_evidence_role"] == "assessed_candidate_evidence"
    # 1. Everything the payload names is inside the manifest closure.
    assert _referenced_fact_ids(selection) <= manifest
    # 2. Selected inputs and assessed candidates partition that closure, and
    #    nothing is ever both.
    assert manifest == selected | assessed
    assert not (selected & assessed)
    assert verified == selected
    assert described == assessed
    # 3. Every described fact's source and filing assets are provable from
    #    the immutable forecast manifest.
    for entry in [*calculation["input_facts"], *selection["assessed_evidence"]]:
        assert entry["source_asset_id"] in asset_ids
        assert entry["filing_evidence_asset_id"] in asset_ids
    # 4. TTM dependency closure, including windows this forecast never used.
    for dependency in selection["ttm_dependencies"].values():
        assert set(dependency["source_fact_ids"]) <= manifest
    # 5. A candidate the search did not select is assessed evidence and is
    #    never a verified formula input, whatever the outcome was.
    rejected = _rejected_candidate_fact_ids(selection)
    assert rejected <= assessed
    assert not (rejected & verified)
    # 6. Rejected/assessed evidence is never labelled verified.
    assert "verified" not in json.dumps(selection["assessed_evidence"])
    assert "verified" not in json.dumps(selection["invested_capital_assessment"] or {})
    return selection


def _alternate_balance_sheet_evidence(
    listing: Listing,
    *,
    period_end: date,
    available_at: datetime,
    source_concept: str = EQUITY_ALTERNATE,
    value: Decimal = Decimal("420"),
) -> tuple[DataAsset, DataAsset]:
    """A same-date equity alternative filed through its own source assets.

    Keeping the alternative on distinct companyfacts and filing assets is
    what makes manifest closure observable: if the assessment referenced it
    without the manifest covering it, these two asset IDs would be missing
    from the forecast's immutable ``source_assets``.
    """
    companyfacts, filing = _sec_assets(listing)
    _fact(
        listing,
        companyfacts,
        filing,
        concept="equity",
        value=value,
        start=None,
        end=period_end,
        fiscal_period="FY",
        available_at=available_at,
        accession=f"{listing.ticker}-alt-equity-{period_end.isoformat()}",
        source_concept=source_concept,
    )
    return companyfacts, filing


def _v3_manifest_case(
    ticker: str,
    *,
    balance_sheets: tuple[tuple[date, str, datetime], ...],
    minimum_peers: dict[int, int] | None = None,
) -> tuple[Any, set[str]]:
    target = _listing(ticker)
    peer = _listing(f"{ticker}P")
    target_price = _company_evidence(target, sic="3571", balance_sheets=balance_sheets)
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    alternate_assets: set[str] = set()
    for period_end, _debt_concept, available_at in balance_sheets:
        companyfacts, filing = _alternate_balance_sheet_evidence(
            target,
            period_end=period_end,
            available_at=available_at,
        )
        alternate_assets.update({str(companyfacts.pk), str(filing.pk)})
    config = load_long_forecast_config(V3_PATH)
    config = replace(
        config,
        peer=replace(config.peer, minimum_peers=minimum_peers or {4: 1, 3: 1, 2: 1}),
    )
    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(target.pk)]["3y"]
    return forecast, alternate_assets


@pytest.mark.django_db
def test_manifest_closes_over_a_successful_assessment() -> None:
    forecast, alternate_assets = _v3_manifest_case(
        "MANOK",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="reported_long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    selection = _assert_manifest_closes(forecast)

    assert forecast.scenario.base is not None
    assert selection["invested_capital_assessment"]["status"] == "selected_compatible_pair"
    # The rejected same-date alternative is carried, not dropped.
    assert alternate_assets <= {str(asset.pk) for asset in forecast.source_assets}
    assert selection["assessed_evidence"]
    assert selection["ttm_dependencies"]


@pytest.mark.django_db
def test_manifest_closes_when_no_compatible_pair_exists() -> None:
    forecast, alternate_assets = _v3_manifest_case(
        "MANNOPAIR",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    selection = _assert_manifest_closes(forecast)

    assert forecast.scenario.base is None
    assessment = selection["invested_capital_assessment"]
    assert assessment["status"] == "no_compatible_pair"
    assert assessment["assessment_status"] == "assessed_incompatible_or_unavailable"
    assert alternate_assets <= {str(asset.pk) for asset in forecast.source_assets}


@pytest.mark.django_db
def test_manifest_closes_when_one_side_has_no_candidate() -> None:
    forecast, alternate_assets = _v3_manifest_case(
        "MANNOBEG",
        balance_sheets=(
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    selection = _assert_manifest_closes(forecast)

    assert forecast.scenario.base is None
    assessment = selection["invested_capital_assessment"]
    assert assessment["status"] == "missing_beginning_candidates"
    assert assessment["beginning_candidates"] == []
    # The examined ending-side alternative is still fully accounted for.
    assert alternate_assets <= {str(asset.pk) for asset in forecast.source_assets}


@pytest.mark.django_db
def test_manifest_closes_when_a_selected_pair_is_later_withheld_for_peers() -> None:
    forecast, alternate_assets = _v3_manifest_case(
        "MANPEER",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="reported_long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
        minimum_peers={4: 9, 3: 9, 2: 9},
    )
    selection = _assert_manifest_closes(forecast)

    assert forecast.scenario.base is None
    assert "peer set met floors" in forecast.calculation["insufficiency_reason"]
    assert selection["invested_capital_assessment"]["status"] == "selected_compatible_pair"
    assert alternate_assets <= {str(asset.pk) for asset in forecast.source_assets}


# ---------------------------------------------------------------------------
# RI re-review 4: failure-path candidates are assessed, never selected inputs
# ---------------------------------------------------------------------------


def _assert_rejected_candidates_are_assessed_only(
    forecast: Any,
    selection: dict[str, Any],
) -> set[str]:
    """Rejected candidates are described as assessed and asset-provable.

    Returns the independently derived rejected set so a caller can assert
    what its own scenario expects about it.
    """
    rejected = _rejected_candidate_fact_ids(selection)
    input_fact_ids = {fact["id"] for fact in forecast.calculation["input_facts"]}
    described = {entry["id"]: entry for entry in selection["assessed_evidence"]}
    asset_ids = {str(asset.pk) for asset in forecast.source_assets}

    assert rejected, "the scenario must actually have read candidate evidence"
    # 1. Never a selected, verified formula input.
    assert not (rejected & input_fact_ids)
    assert not (rejected & set(selection["selected_input_fact_ids"]))
    # 2. Always assessed evidence, described once.
    assert rejected <= set(selection["assessed_evidence_fact_ids"])
    assert rejected <= set(described)
    # 3. Always provable from the immutable manifest, source and filing.
    for fact_id in rejected:
        assert described[fact_id]["source_asset_id"] in asset_ids
        assert described[fact_id]["filing_evidence_asset_id"] in asset_ids
    return rejected


@pytest.mark.django_db
def test_no_compatible_pair_candidates_are_assessed_never_verified_inputs() -> None:
    """A rejected balance-sheet candidate cannot become a verified input."""
    forecast, alternate_assets = _v3_manifest_case(
        "CLSNOPAIR",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    selection = _assert_manifest_closes(forecast)
    assessment = selection["invested_capital_assessment"]

    assert forecast.scenario.base is None
    assert assessment["status"] == "no_compatible_pair"
    assert assessment["selected_beginning_fact_ids"] == []
    assert assessment["selected_ending_fact_ids"] == []
    rejected = _assert_rejected_candidates_are_assessed_only(forecast, selection)
    # Both sides were examined, and neither side became an input.
    assert {
        fact_id
        for candidate in assessment["beginning_candidates"]
        for fact_id in candidate["fact_ids"]
    } <= rejected
    assert {
        fact_id
        for candidate in assessment["ending_candidates"]
        for fact_id in candidate["fact_ids"]
    } <= rejected
    assert alternate_assets <= {str(asset.pk) for asset in forecast.source_assets}


@pytest.mark.django_db
def test_missing_side_candidates_are_assessed_never_verified_inputs() -> None:
    """The side that did exist is assessed; nothing at all is selected."""
    forecast, alternate_assets = _v3_manifest_case(
        "CLSNOBEG",
        balance_sheets=(
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    selection = _assert_manifest_closes(forecast)
    assessment = selection["invested_capital_assessment"]

    assert forecast.scenario.base is None
    assert assessment["status"] == "missing_beginning_candidates"
    assert assessment["beginning_candidates"] == []
    assert assessment["ending_candidates"]
    rejected = _assert_rejected_candidates_are_assessed_only(forecast, selection)
    assert {
        fact_id
        for candidate in assessment["ending_candidates"]
        for fact_id in candidate["fact_ids"]
    } == rejected
    assert alternate_assets <= {str(asset.pk) for asset in forecast.source_assets}


@pytest.mark.django_db
def test_a_selected_pair_stays_a_verified_input_when_peers_withhold() -> None:
    """Control: selection survives, and only the alternatives are assessed."""
    forecast, alternate_assets = _v3_manifest_case(
        "CLSPEER",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="reported_long_term_debt"),
            _balance_sheet(date(2025, 12, 31), debt_concept="reported_long_term_debt"),
        ),
        minimum_peers={4: 9, 3: 9, 2: 9},
    )
    selection = _assert_manifest_closes(forecast)
    assessment = selection["invested_capital_assessment"]
    selected_pair = set(assessment["selected_beginning_fact_ids"]) | set(
        assessment["selected_ending_fact_ids"]
    )
    input_fact_ids = {fact["id"] for fact in forecast.calculation["input_facts"]}

    assert forecast.scenario.base is None
    assert "peer set met floors" in forecast.calculation["insufficiency_reason"]
    assert assessment["status"] == "selected_compatible_pair"
    # The selected pair's facts really were used, and stay selected inputs.
    assert selected_pair
    assert selected_pair <= input_fact_ids
    assert selected_pair <= set(selection["selected_input_fact_ids"])
    assert not (selected_pair & set(selection["assessed_evidence_fact_ids"]))
    # The same-date alternatives it did not use remain assessed evidence.
    unused = _assert_rejected_candidates_are_assessed_only(forecast, selection)
    assert not (unused & selected_pair)
    assert alternate_assets <= {str(asset.pk) for asset in forecast.source_assets}


@pytest.mark.django_db
def test_unselected_ttm_alias_lineage_is_still_manifest_covered() -> None:
    """A losing alias's controlling filings are assessed evidence too."""
    target = _listing("MANALIAS")
    peer = _listing("MANALIASP")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    primary_facts, primary_filing = _sec_assets(target)
    for index, (start, end) in enumerate(_quarter_windows(date(2025, 1, 1), 4), start=1):
        _fact(
            target,
            primary_facts,
            primary_filing,
            concept="net_income",
            value=Decimal("17") + Decimal(index),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            available_at=datetime(2026, 2, 10 + index, tzinfo=UTC),
            accession=f"{target.ticker}-ni-primary-q{index}",
            source_concept=NET_INCOME_PRIMARY,
        )
    # A second, incomplete net-income alias filed through its own source
    # assets. It loses the anchor but was still read, ranked, and rejected.
    companyfacts, filing = _sec_assets(target)
    for index, (start, end) in enumerate(_quarter_windows(date(2025, 7, 1), 2), start=3):
        _fact(
            target,
            companyfacts,
            filing,
            concept="net_income",
            value=Decimal("30") + Decimal(index),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            available_at=datetime(2026, 2, 6 + index, tzinfo=UTC),
            accession=f"{target.ticker}-ni-alt-q{index}",
            source_concept=NET_INCOME_ALTERNATE,
        )
    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]

    selection = _assert_manifest_closes(forecast)
    net_income = selection["ttm_alias_selection"]["net_income"]
    assert net_income["selected_source_concept"] == NET_INCOME_PRIMARY
    assert NET_INCOME_ALTERNATE in {
        candidate["source_concept"] for candidate in net_income["alias_candidates"]
    }
    asset_ids = {str(asset.pk) for asset in forecast.source_assets}
    assert {str(companyfacts.pk), str(filing.pk)} <= asset_ids


@pytest.mark.django_db
def test_frozen_versions_gain_no_assessed_evidence_manifest() -> None:
    target = _listing("MANV2")
    peer = _listing("MANV2P")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    for path in (V1_PATH, V2_PATH):
        forecast = build_long_forecasts(
            listings=[target, peer],
            current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
            price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=_small_peer_config(load_long_forecast_config(path)),
        )[str(target.pk)]["3y"]

        assert "evidence_selection" not in forecast.calculation
        assert "assessed_evidence" not in json.dumps(forecast.calculation)


# ---------------------------------------------------------------------------
# RI re-review 3: instant identity, UUID blindness, combination overflow
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_reassigning_row_uuids_alone_cannot_change_the_selected_pair() -> None:
    """Same-date alias candidates must be resolved by evidence, not row order."""
    listing = _listing("UUIDBLIND")
    _company_evidence(
        listing,
        sic="3571",
        equity_aliases={
            date(2024, 12, 31): (EQUITY_PRIMARY, EQUITY_ALTERNATE),
            date(2025, 12, 31): (EQUITY_PRIMARY, EQUITY_ALTERNATE),
        },
    )
    facts = list(FundamentalFact.objects.filter(company=listing.security.company))
    sec_config = load_sec_fundamentals_config()
    priority = source_concept_priority(sec_config)

    def _report(rows: list[FundamentalFact]) -> dict[str, Any]:
        series = build_sec_fundamental_series(
            rows,
            config=sec_config,
            ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
            alias_instant_candidates=True,
        )
        report = audit_invested_capital_pairs(
            series=series,
            beginning_target=date(2024, 12, 31),
            ending_target=date(2025, 12, 31),
            tolerance_days=45,
            priority=priority,
            maximum_combinations=REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS,
        )
        return {
            key: report[key]
            for key in (
                "selected_beginning_period_end",
                "selected_ending_period_end",
                "selected_debt_method",
                "selected_debt_components",
                "selected_source_basis",
                "compatible_pair_count",
            )
        }

    baseline = _report(facts)
    assert baseline["compatible_pair_count"] > 1
    assert baseline["selected_source_basis"][0] == {
        "concept": "equity",
        "source_concept": EQUITY_PRIMARY,
    }

    for _trial in range(12):
        assert _report(_with_reassigned_ids(facts)) == baseline


def _noncanonical_instant_identity(listing: Listing) -> FundamentalFact:
    """Persist one balance-sheet row carrying a conflicting period identity.

    Nothing is mutated: the row is written this way, exactly as a malformed
    or mislabelled ingest would leave it.
    """
    fact = FundamentalFact.objects.filter(
        company=listing.security.company,
        concept="equity",
        period_end=date(2025, 12, 31),
    ).first()
    assert fact is not None
    companyfacts, filing = _sec_assets(listing)
    conflicting = FundamentalFact.objects.create(
        company=listing.security.company,
        provider="sec",
        concept="equity",
        taxonomy="us-gaap",
        source_concept=EQUITY_ALTERNATE,
        value=Decimal("451"),
        unit="USD",
        currency="USD",
        period_type=FundamentalFact.PeriodType.INSTANT,
        period_start=None,
        period_end=date(2025, 12, 31),
        # A duration-shaped identity on an instant observation: the alias join
        # would treat this as a different balance-sheet period.
        period_identity="duration:2025-01-01:2025-12-31",
        fiscal_year=2025,
        fiscal_period="FY",
        accession=f"{listing.ticker}-equity-noncanonical",
        filing_form="10-K",
        filing_date=date(2026, 2, 15),
        filed_at=datetime(2026, 2, 15, tzinfo=UTC),
        acceptance_at=datetime(2026, 2, 15, tzinfo=UTC),
        available_at=datetime(2026, 2, 15, tzinfo=UTC),
        availability_basis="acceptance_datetime",
        source_revision=1,
        observation_hash="",
        source_asset=companyfacts,
    )
    FundamentalFactEvidence.objects.create(
        fact=conflicting,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=filing,
    )
    return conflicting


@pytest.mark.django_db
def test_conflicting_instant_identities_withhold_that_listing_explicitly() -> None:
    broken = _listing("BADIDENT")
    healthy = _listing("GOODIDENT")
    healthy_peer = _listing("GOODIDENTP")
    broken_price = _company_evidence(broken, sic="3571")
    healthy_price = _company_evidence(healthy, sic="3571", scale=1.1)
    peer_price = _company_evidence(healthy_peer, sic="3571", scale=1.2)
    conflicting = _noncanonical_instant_identity(broken)

    forecasts = build_long_forecasts(
        listings=[broken, healthy, healthy_peer],
        current_prices={
            str(broken.pk): 50.0,
            str(healthy.pk): 55.0,
            str(healthy_peer.pk): 60.0,
        },
        price_assets={
            str(broken.pk): broken_price,
            str(healthy.pk): healthy_price,
            str(healthy_peer.pk): peer_price,
        },
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )
    withheld = forecasts[str(broken.pk)]["3y"]
    assessment = withheld.calculation["evidence_selection"]["invested_capital_assessment"]

    assert withheld.scenario.base is None
    assert assessment["status"] == "noncanonical_instant_period_identity"
    assert assessment["assessment_status"] == "assessed_incompatible_or_unavailable"
    anomalies = assessment["noncanonical_instant_facts"]
    assert [item["fact_id"] for item in anomalies] == [str(conflicting.pk)]
    assert anomalies[0]["period_identity"] == "duration:2025-01-01:2025-12-31"
    assert anomalies[0]["expected_period_identity"] == "instant::2025-12-31"
    assert "canonical instant" in withheld.calculation["insufficiency_reason"]
    assert assessment["selected_beginning_period_end"] is None
    assert assessment["selected_ending_period_end"] is None
    # The conflicting row is assessed evidence, never silently selected.
    assert (
        str(conflicting.pk)
        in (withheld.calculation["evidence_selection"]["assessed_evidence_fact_ids"])
    )
    _assert_manifest_closes(withheld)

    # The rest of the run is unaffected.
    assert forecasts[str(healthy.pk)]["3y"].scenario.base is not None


@pytest.mark.django_db
def test_conflicting_instant_identities_still_return_an_audit_entry() -> None:
    broken = _listing("BADIDENTA")
    healthy = _listing("GOODIDENTA")
    _company_evidence(broken, sic="3571")
    _company_evidence(healthy, sic="3571", scale=1.1)
    _noncanonical_instant_identity(broken)

    report = audit_long_evidence(
        listing_ids=[str(broken.pk), str(healthy.pk)],
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=load_long_forecast_config(V3_PATH),
    )
    entries = {entry["listing_id"]: entry for entry in report["listings"]}

    assert len(entries) == 2
    assert entries[str(broken.pk)]["status"] == "assessed_withheld"
    assert entries[str(broken.pk)]["noncanonical_instant_facts"]
    assert entries[str(healthy.pk)]["status"] == "audited"


#: 7 equity aliases x 7 cash aliases x 7 reported-long-term-debt aliases,
#: i.e. 21 same-date alternatives on each accounting date.
OVERFLOW_ALIAS_COUNT = 7
OVERFLOW_AXES = 3
OVERFLOW_ALTERNATIVES = OVERFLOW_ALIAS_COUNT * OVERFLOW_AXES
OVERFLOW_COMBINATIONS = OVERFLOW_ALIAS_COUNT**OVERFLOW_AXES


def _wide_alias_sec_config() -> Any:
    """The reviewed fundamentals config widened to 7 balance-sheet aliases.

    Purely synthetic: no reviewed configuration file is edited. It exists so
    the combination ceiling can be exercised against a real code path while
    the config version and hash the forecast binds stay exactly as reviewed.
    """
    base = load_sec_fundamentals_config()
    widened = []
    for rule in base.concept_rules:
        if rule.canonical_concept in {"equity", "cash_and_equivalents", "reported_long_term_debt"}:
            extra = tuple(
                f"Synthetic{rule.canonical_concept.title().replace('_', '')}Alias{index}"
                for index in range(OVERFLOW_ALIAS_COUNT - len(rule.source_concepts))
            )
            widened.append(replace(rule, source_concepts=(*rule.source_concepts, *extra)))
        else:
            widened.append(rule)
    return replace(base, concept_rules=tuple(widened))


def _wide_alias_balance_sheet(
    listing: Listing,
    sec_config: Any,
    *,
    period_ends: tuple[date, ...] = (date(2024, 12, 31), date(2025, 12, 31)),
) -> dict[date, list[FundamentalFact]]:
    """File every declared balance-sheet alias through its own assets.

    Each alternative gets a distinct companyfacts asset and a distinct filing
    asset, so a manifest that failed to carry one responsible fact would be
    detectably missing that fact's two assets. The caller supplies
    ``balance_sheets=()`` to `_company_evidence` so every alternative on a
    date, including the default alias, is filed here on equal footing.
    """
    rules = {rule.canonical_concept: rule for rule in sec_config.concept_rules}
    filed: dict[date, list[FundamentalFact]] = {}
    for period_end in period_ends:
        available_at = datetime(period_end.year + 1, 2, 15, tzinfo=UTC)
        for concept, base_value in (
            ("equity", 400.0),
            ("cash_and_equivalents", 50.0),
            ("reported_long_term_debt", 100.0),
        ):
            for index, alias in enumerate(rules[concept].source_concepts):
                companyfacts, filing = _sec_assets(listing)
                filed.setdefault(period_end, []).append(
                    _fact(
                        listing,
                        companyfacts,
                        filing,
                        concept=concept,
                        value=Decimal(str(base_value + index)),
                        start=None,
                        end=period_end,
                        fiscal_period="FY",
                        available_at=available_at,
                        accession=(f"{listing.ticker}-{concept}-{index}-{period_end.isoformat()}"),
                        source_concept=f"us-gaap:{alias}",
                    )
                )
    return filed


@pytest.mark.django_db
def test_combination_overflow_withholds_one_listing_without_aborting_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sec_config = _wide_alias_sec_config()
    monkeypatch.setattr(
        "stanstock.research.long_forecasts.load_sec_fundamentals_config",
        lambda: sec_config,
    )
    overflow = _listing("OVERFLOW")
    valid = _listing("VALIDRUN")
    overflow_price = _company_evidence(overflow, sic="3571", balance_sheets=())
    valid_price = _company_evidence(valid, sic="3571", scale=1.1)
    filed = _wide_alias_balance_sheet(overflow, sec_config)

    forecasts = build_long_forecasts(
        listings=[overflow, valid],
        current_prices={str(overflow.pk): 50.0, str(valid.pk): 55.0},
        price_assets={str(overflow.pk): overflow_price, str(valid.pk): valid_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )
    withheld = forecasts[str(overflow.pk)]["3y"]
    selection = _assert_manifest_closes(withheld)
    assessment = selection["invested_capital_assessment"]

    assert MAX_SAME_DATE_SOURCE_COMBINATIONS == 256
    assert OVERFLOW_COMBINATIONS == 343
    assert withheld.scenario.base is None
    assert assessment["status"] == "same_date_combination_ceiling_exceeded"
    assert assessment["assessment_status"] == "assessed_incompatible_or_unavailable"
    overflow_detail = assessment["same_date_combination_overflow"]
    assert overflow_detail["combination_count"] == OVERFLOW_COMBINATIONS
    assert overflow_detail["ceiling"] == MAX_SAME_DATE_SOURCE_COMBINATIONS
    assert overflow_detail["period_end"] == "2024-12-31"
    assert str(OVERFLOW_COMBINATIONS) in withheld.calculation["insufficiency_reason"]
    # Nothing was truncated, enumerated, or selected.
    assert assessment["beginning_candidates"] == []
    assert assessment["ending_candidates"] == []
    assert assessment["selected_beginning_period_end"] is None
    assert assessment["selected_ending_period_end"] is None
    assert assessment["selected_beginning_fact_ids"] == []
    assert assessment["selected_ending_fact_ids"] == []
    assert len(json.dumps(overflow_detail)) < 10_000

    # The axes are the factors of the refused bound, never its product.
    axes = {axis["concept"]: axis for axis in overflow_detail["axes"]}
    assert overflow_detail["axis_count"] == OVERFLOW_AXES
    assert set(axes) == {"equity", "cash_and_equivalents", "reported_long_term_debt"}
    product = 1
    for axis in axes.values():
        assert axis["option_count"] == OVERFLOW_ALIAS_COUNT
        assert axis["combination_multiplier"] == OVERFLOW_ALIAS_COUNT
        assert len(set(axis["source_concepts"])) == OVERFLOW_ALIAS_COUNT
        assert len(set(axis["fact_ids"])) == OVERFLOW_ALIAS_COUNT
        product *= axis["combination_multiplier"]
    assert product == OVERFLOW_COMBINATIONS

    # Every fact responsible for the 343 combinations is named, assessed,
    # asset-provable, and never a selected formula input.
    responsible = {str(fact.pk) for fact in filed[date(2024, 12, 31)]}
    assert len(responsible) == OVERFLOW_ALTERNATIVES
    assert set(overflow_detail["responsible_fact_ids"]) == responsible
    assert responsible <= set(selection["assessed_evidence_fact_ids"])
    assert responsible <= set(selection["manifest_evidence_fact_ids"])
    assert not (responsible & set(selection["selected_input_fact_ids"]))
    assert not (responsible & {fact["id"] for fact in withheld.calculation["input_facts"]})
    described = {entry["id"]: entry for entry in selection["assessed_evidence"]}
    assert responsible <= set(described)
    source_assets = {str(asset.pk) for asset in withheld.source_assets}
    filing_assets = {described[fact_id]["filing_evidence_asset_id"] for fact_id in responsible}
    companyfacts_assets = {described[fact_id]["source_asset_id"] for fact_id in responsible}
    # Each alternative is provable through its own distinct filing evidence.
    assert len(filing_assets) == OVERFLOW_ALTERNATIVES
    assert len(companyfacts_assets) == OVERFLOW_ALTERNATIVES
    assert (filing_assets | companyfacts_assets) <= source_assets

    # The rest of the analysis still produces a forecast.
    assert forecasts[str(valid.pk)]["3y"].scenario.base is not None


@pytest.mark.django_db
def test_combination_overflow_on_one_side_keeps_the_other_sides_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused ending side does not erase the assessed beginning side."""
    sec_config = _wide_alias_sec_config()
    monkeypatch.setattr(
        "stanstock.research.long_forecasts.load_sec_fundamentals_config",
        lambda: sec_config,
    )
    overflow = _listing("OVERFLOWEND")
    peer = _listing("OVERFLOWENDP")
    overflow_price = _company_evidence(
        overflow,
        sic="3571",
        balance_sheets=(
            _balance_sheet(date(2024, 12, 31), debt_concept="reported_long_term_debt"),
        ),
    )
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    filed = _wide_alias_balance_sheet(overflow, sec_config, period_ends=(date(2025, 12, 31),))

    withheld = build_long_forecasts(
        listings=[overflow, peer],
        current_prices={str(overflow.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(overflow.pk): overflow_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(overflow.pk)]["3y"]
    selection = _assert_manifest_closes(withheld)
    assessment = selection["invested_capital_assessment"]

    assert withheld.scenario.base is None
    assert assessment["status"] == "same_date_combination_ceiling_exceeded"
    assert assessment["same_date_combination_overflow"]["period_end"] == "2025-12-31"
    # The beginning side was assessed before the refusal and is retained.
    assert assessment["beginning_candidates"]
    assert assessment["beginning_candidate_period_ends"] == ["2024-12-31"]
    assert assessment["ending_candidates"] == []
    assert assessment["selected_beginning_fact_ids"] == []
    responsible = {str(fact.pk) for fact in filed[date(2025, 12, 31)]}
    beginning_side = {
        fact_id
        for candidate in assessment["beginning_candidates"]
        for fact_id in candidate["fact_ids"]
    }
    assessed = set(selection["assessed_evidence_fact_ids"])
    assert responsible <= assessed
    assert beginning_side <= assessed
    assert not ((responsible | beginning_side) & set(selection["selected_input_fact_ids"]))


@pytest.mark.django_db
def test_combination_overflow_still_returns_both_audit_entries() -> None:
    sec_config = _wide_alias_sec_config()
    overflow = _listing("OVERFLOWA")
    valid = _listing("VALIDRUNA")
    _company_evidence(overflow, sic="3571", balance_sheets=())
    _company_evidence(valid, sic="3571", scale=1.1)
    _wide_alias_balance_sheet(overflow, sec_config)

    report = audit_long_evidence(
        listing_ids=[str(overflow.pk), str(valid.pk)],
        target_date=TARGET_DATE,
        available_through=DECISION_TIME,
        decision_time=DECISION_TIME,
        config=load_long_forecast_config(V3_PATH),
        sec_config=sec_config,
    )
    entries = {entry["listing_id"]: entry for entry in report["listings"]}

    assert len(entries) == 2
    overflow_entry = entries[str(overflow.pk)]["invested_capital"]
    assert overflow_entry["status"] == "audited"
    assert overflow_entry["assessment_status"] == "assessed_incompatible_or_unavailable"
    assert overflow_entry["same_date_combination_overflow"]["combination_count"] == (
        OVERFLOW_COMBINATIONS
    )
    assert overflow_entry["compatible_pair_available"] is False
    assert entries[str(valid.pk)]["invested_capital"]["compatible_pair_available"] is True


# ---------------------------------------------------------------------------
# RI-1: a backdated same-accession correction is resolved conservatively
# ---------------------------------------------------------------------------


HISTORICAL_CUTOFF = DECISION_TIME
CORRECTION_RETRIEVAL = datetime(2026, 4, 1, 12, tzinfo=UTC)
LATER_DECISION_TIME = datetime(2026, 5, 1, 12, tzinfo=UTC)


def _backdated_correction(
    listing: Listing,
    *,
    concept: str,
    quarter: tuple[date, date],
    fiscal_period: str,
    value: Decimal,
    backdated_available_at: datetime,
    retrieved_at: datetime,
) -> tuple[FundamentalFact, DataAsset, DataAsset]:
    """A legacy-shaped correction: revision 2, but dated at acceptance.

    This is what rows persisted before the correction availability basis
    existed look like. The restating retrieval happened at ``retrieved_at``,
    yet ``available_at`` claims the original filing's acceptance, so any
    cutoff after acceptance would read it.
    """
    companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{listing.ticker}-correction-companyfacts",
        retrieved_at=retrieved_at,
    )
    filing = _asset(
        provider="sec",
        kind="sec_submissions",
        subject=f"{listing.ticker}-correction-filing",
        retrieved_at=retrieved_at,
    )
    fact = _fact(
        listing,
        companyfacts,
        filing,
        concept=concept,
        value=value,
        start=quarter[0],
        end=quarter[1],
        fiscal_period=fiscal_period,
        available_at=backdated_available_at,
        accession=f"{listing.ticker}-{concept}-{fiscal_period}",
        source_revision=2,
    )
    return fact, companyfacts, filing


def _backdated_correction_fixture(ticker: str) -> tuple[Listing, Listing, Any, Any, Any]:
    target = _listing(ticker)
    peer = _listing(f"{ticker}P")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)
    correction, companyfacts, filing = _backdated_correction(
        target,
        concept="operating_cash_flow",
        quarter=(date(2025, 10, 1), date(2025, 12, 31)),
        fiscal_period="Q4",
        value=Decimal("35"),
        # Exactly the original quarter observation's availability.
        backdated_available_at=datetime(2026, 2, 14, tzinfo=UTC),
        retrieved_at=CORRECTION_RETRIEVAL,
    )
    return target, peer, (target_price, peer_price), correction, (companyfacts, filing)


def _forecast_at(
    *,
    target: Listing,
    peer: Listing,
    prices: Any,
    config_path: Path,
    data_cutoff: datetime,
    decision_time: datetime = LATER_DECISION_TIME,
) -> Any:
    target_price, peer_price = prices
    return build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(decision_time),
        data_cutoff=data_cutoff,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(config_path)),
    )[str(target.pk)]["3y"]


@pytest.mark.django_db
def test_v3_defers_a_backdated_correction_and_keeps_the_proven_original() -> None:
    """An unproven correction cannot enter a historical reconstruction.

    The correction is visible (its asset was retrieved before the decision
    time) and its recorded ``available_at`` passes the historical cutoff --
    but only because a legacy ingestion backdated it to the acceptance of the
    accession it restates. Nothing before `CORRECTION_RETRIEVAL` proves the
    corrected value existed, so long-v3 defers it, keeps the original
    revision that *is* proven at that cutoff, and records the deferral as
    assessed evidence with its own reason.
    """
    target, peer, prices, correction, (late_companyfacts, late_filing) = (
        _backdated_correction_fixture("BACKDATE")
    )

    # The exposure this closes: recorded availability admits it at the
    # historical cutoff, while the retrieval that proved it does not.
    assert correction.source_revision == 2
    assert correction.available_at <= HISTORICAL_CUTOFF
    assert correction.source_asset.retrieved_at > HISTORICAL_CUTOFF
    assert correction.source_asset.retrieved_at <= LATER_DECISION_TIME
    assert correction.availability_basis == "acceptance_datetime"

    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="operating_cash_flow",
        period_start=date(2025, 10, 1),
        period_end=date(2025, 12, 31),
        source_revision=1,
    )

    v3 = _forecast_at(
        target=target,
        peer=peer,
        prices=prices,
        config_path=V3_PATH,
        data_cutoff=HISTORICAL_CUTOFF,
    )
    selection = _assert_manifest_closes(v3)

    assert v3.scenario.base is not None
    deferred = selection["deferred_unproven_corrections"]
    assert [entry["fact_id"] for entry in deferred] == [str(correction.pk)]
    entry = deferred[0]
    assert entry["source_revision"] == 2
    assert entry["accession"] == correction.accession
    assert entry["recorded_available_at"] == correction.available_at.isoformat()
    assert entry["proven_available_at"] == CORRECTION_RETRIEVAL.isoformat()
    assert entry["source_asset_retrieved_at"] == CORRECTION_RETRIEVAL.isoformat()
    assert entry["available_through"] == HISTORICAL_CUTOFF.isoformat()
    assert "after the data cutoff" in entry["reason"]

    # Assessed, never selected -- and the proven original took its place.
    assert str(correction.pk) in selection["assessed_evidence_fact_ids"]
    assert str(correction.pk) not in selection["selected_input_fact_ids"]
    assert str(correction.pk) not in {fact["id"] for fact in v3.calculation["input_facts"]}
    assert str(original.pk) in selection["selected_input_fact_ids"]

    # The deferred row is still fully provable from the immutable manifest.
    asset_ids = {str(asset.pk) for asset in v3.source_assets}
    assert {str(late_companyfacts.pk), str(late_filing.pk)} <= asset_ids

    # No TTM window may cite the deferred correction.
    for dependency in selection["ttm_dependencies"].values():
        assert str(correction.pk) not in dependency["source_fact_ids"]

    # Read-time resolution only: neither immutable row was rewritten.
    correction.refresh_from_db()
    original.refresh_from_db()
    assert correction.available_at == datetime(2026, 2, 14, tzinfo=UTC)
    assert correction.availability_basis == "acceptance_datetime"
    assert original.source_revision == 1


@pytest.mark.django_db
def test_v3_admits_the_same_correction_once_its_retrieval_is_proven() -> None:
    """Past the correction's own retrieval boundary it is ordinary evidence.

    The deferral is a point-in-time statement, not a permanent rejection of
    corrections, so the identical row is selected at a cutoff that its
    retrieval precedes.
    """
    target, peer, prices, correction, _assets = _backdated_correction_fixture("BACKDATEOK")
    proven_cutoff = datetime(2026, 4, 15, 12, tzinfo=UTC)
    assert correction.source_asset.retrieved_at < proven_cutoff

    v3 = _forecast_at(
        target=target,
        peer=peer,
        prices=prices,
        config_path=V3_PATH,
        data_cutoff=proven_cutoff,
    )
    selection = v3.calculation["evidence_selection"]

    assert selection["deferred_unproven_corrections"] == []
    assert str(correction.pk) in selection["selected_input_fact_ids"]


@pytest.mark.django_db
def test_frozen_v2_reading_of_the_same_correction_is_unchanged() -> None:
    """long-v2 keeps its released behavior; the new gate is v3-only.

    This is the frozen-contract half of the finding: the deferral must not
    silently re-date, re-select, or withhold anything for an already-released
    version, whose payload has no evidence-selection section at all.
    """
    target, peer, prices, correction, _assets = _backdated_correction_fixture("BACKDATEV2")

    v2 = _forecast_at(
        target=target,
        peer=peer,
        prices=prices,
        config_path=V2_PATH,
        data_cutoff=HISTORICAL_CUTOFF,
    )

    assert v2.calculation["method_version"] == "us-sec-long-v2"
    assert "evidence_selection" not in v2.calculation
    # v2 reads recorded availability only, exactly as released.
    assert str(correction.pk) in {fact["id"] for fact in v2.calculation["input_facts"]}


# ---------------------------------------------------------------------------
# RI-2: every alias-tail assessment carries the facts that established it
# ---------------------------------------------------------------------------


PRETAX_ALTERNATE = (
    "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterest"
    "AndIncomeLossFromEquityMethodInvestments"
)


def _alias_quarters(
    listing: Listing,
    *,
    label: str,
    concept: str,
    source_concept: str,
    quarters: list[tuple[date, date]],
    value: Decimal,
) -> tuple[DataAsset, DataAsset, set[str]]:
    """File one alias's quarters through its own companyfacts/filing pair.

    Independent assets are what make manifest closure observable: if the
    assessment describes this alias but never carries its evidence, the
    assets simply will not be in the forecast's immutable manifest.
    """
    companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{listing.ticker}-{label}-companyfacts",
        retrieved_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    filing = _asset(
        provider="sec",
        kind="sec_submissions",
        subject=f"{listing.ticker}-{label}-filing",
        retrieved_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    created: set[str] = set()
    for index, (start, end) in enumerate(quarters, start=1):
        fact = _fact(
            listing,
            companyfacts,
            filing,
            concept=concept,
            value=value + Decimal(index),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            available_at=datetime(2026, 2, 1, tzinfo=UTC) + timedelta(days=index),
            accession=f"{listing.ticker}-{label}-{index}",
            source_concept=source_concept,
        )
        created.add(str(fact.pk))
    return companyfacts, filing, created


@pytest.mark.django_db
def test_every_alias_tail_assessment_names_its_own_evidence() -> None:
    """A losing alias may not be described without its own facts and assets.

    The successful forecast states three separate alias-tail outcomes: a
    winner anchored on the newest quarter with a complete tail, a stale but
    complete alternative, and an alias whose tail is incomplete. Each is
    filed through its own companyfacts/filing pair, so all three lineages
    must be carried as assessed evidence -- never as selected inputs -- and
    every one of their source *and* filing assets must be provable from the
    immutable forecast manifest.
    """
    target = _listing("ALIASLIN")
    peer = _listing("ALIASLINP")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    winner_assets = _alias_quarters(
        target,
        label="winner",
        concept="net_income",
        source_concept=NET_INCOME_PRIMARY,
        quarters=_quarter_windows(date(2025, 1, 1), 4),
        value=Decimal("30"),
    )
    stale_assets = _alias_quarters(
        target,
        label="stale",
        concept="net_income",
        source_concept=NET_INCOME_ALTERNATE,
        quarters=_quarter_windows(date(2024, 10, 1), 4),
        value=Decimal("20"),
    )
    # A losing tail that is incomplete rather than merely stale: the frozen
    # primary alias already supplies this concept's four quarters.
    incomplete_assets = _alias_quarters(
        target,
        label="incomplete",
        concept="pretax_income",
        source_concept=PRETAX_ALTERNATE,
        quarters=_quarter_windows(date(2025, 1, 1), 2),
        value=Decimal("10"),
    )

    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]
    selection = _assert_manifest_closes(forecast)

    assert forecast.scenario.base is not None

    net_income = selection["ttm_alias_selection"]["net_income"]
    net_candidates = {entry["source_concept"]: entry for entry in net_income["alias_candidates"]}
    pretax = selection["ttm_alias_selection"]["pretax_income"]
    pretax_candidates = {entry["source_concept"]: entry for entry in pretax["alias_candidates"]}

    # 1. The three assessed outcomes the payload actually states.
    assert net_income["selected_source_concept"] == NET_INCOME_PRIMARY
    assert net_income["stale_complete_alternatives"] == [NET_INCOME_ALTERNATE]
    assert net_candidates[NET_INCOME_PRIMARY]["has_homogeneous_four_quarter_tail"] is True
    assert net_candidates[NET_INCOME_ALTERNATE]["has_homogeneous_four_quarter_tail"] is True
    assert pretax["selected_source_concept"] == SOURCE_CONCEPTS["pretax_income"]
    assert pretax_candidates[PRETAX_ALTERNATE]["has_homogeneous_four_quarter_tail"] is False
    assert pretax_candidates[PRETAX_ALTERNATE]["quarter_count"] == 2

    # 2. Each alias names the facts its own tail assessment read. The two
    #    alternates are filed entirely by this test, so their lineage is
    #    exact; the primary alias also carries the fixture's annual facts.
    assert set(net_candidates[NET_INCOME_PRIMARY]["assessed_source_fact_ids"]) >= winner_assets[2]
    assert set(net_candidates[NET_INCOME_ALTERNATE]["assessed_source_fact_ids"]) == stale_assets[2]
    assert (
        set(pretax_candidates[PRETAX_ALTERNATE]["assessed_source_fact_ids"]) == incomplete_assets[2]
    )
    for candidate in (
        net_candidates[NET_INCOME_PRIMARY],
        net_candidates[NET_INCOME_ALTERNATE],
        pretax_candidates[PRETAX_ALTERNATE],
    ):
        assert candidate["assessed_quarter_period_ends"]
    assert net_candidates[NET_INCOME_ALTERNATE]["assessed_tail_period_ends"] == [
        "2024-12-31",
        "2025-03-31",
        "2025-06-30",
        "2025-09-30",
    ]
    assert pretax_candidates[PRETAX_ALTERNATE]["assessed_tail_period_ends"] == []

    # 3. Losing lineages are assessed evidence, never selected formula inputs.
    losing = stale_assets[2] | incomplete_assets[2]
    assessed = set(selection["assessed_evidence_fact_ids"])
    selected = set(selection["selected_input_fact_ids"])
    described = {entry["id"] for entry in selection["assessed_evidence"]}
    assert losing <= assessed
    assert not (losing & selected)
    assert losing <= described

    # 4. Both the companyfacts and the filing asset of every assessed alias
    #    are provable from the immutable manifest.
    asset_ids = {str(asset.pk) for asset in forecast.source_assets}
    for label, (companyfacts, filing, _facts) in (
        ("winner", winner_assets),
        ("stale", stale_assets),
        ("incomplete", incomplete_assets),
    ):
        assert str(companyfacts.pk) in asset_ids, label
        assert str(filing.pk) in asset_ids, label


@pytest.mark.django_db
def test_an_alias_with_no_usable_quarter_still_names_its_rejected_evidence() -> None:
    """Facts rejected *before* any candidate exists are still assessed.

    Lineage collected only from constructed quarter candidates leaves an
    alias that never produces one described with an empty list. Two shapes
    reach that state and both are filed here on their own assets:

    - an annual-only alternate alias, whose single 365-day duration is not a
      quarter and cannot chain with anything; and
    - a year-to-date pair whose implied quarter spans 184 days, so the
      derivation is refused outright.

    All three facts were read and rejected, so all three must be assessed,
    manifest-provable through both their companyfacts and filing assets, and
    never selected.
    """
    target = _listing("NOCAND")
    peer = _listing("NOCANDP")
    target_price = _company_evidence(target, sic="3571")
    peer_price = _company_evidence(peer, sic="3571", scale=1.1)

    companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{target.ticker}-nocand-companyfacts",
        retrieved_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    filing = _asset(
        provider="sec",
        kind="sec_submissions",
        subject=f"{target.ticker}-nocand-filing",
        retrieved_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    rejected: dict[str, str] = {}
    for label, start, end in (
        # Annual-only: a 365-day duration is never a quarter observation.
        ("annual_only", date(2025, 1, 1), date(2025, 12, 31)),
        # A YTD pair sharing one period start. 182 and 366 days are both
        # outside the quarter bounds, and the quarter their difference
        # implies spans 2024-07-01..2024-12-31 (184 days), so the derivation
        # is rejected and no candidate is ever constructed.
        ("ytd_half", date(2024, 1, 1), date(2024, 6, 30)),
        ("ytd_full", date(2024, 1, 1), date(2024, 12, 31)),
    ):
        fact = _fact(
            target,
            companyfacts,
            filing,
            concept="pretax_income",
            value=Decimal("40"),
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=datetime(2026, 2, 5, tzinfo=UTC),
            accession=f"{target.ticker}-nocand-{label}",
            source_concept=PRETAX_ALTERNATE,
        )
        rejected[label] = str(fact.pk)

    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=_small_peer_config(load_long_forecast_config(V3_PATH)),
    )[str(target.pk)]["3y"]
    selection = _assert_manifest_closes(forecast)

    assert forecast.scenario.base is not None

    pretax = selection["ttm_alias_selection"]["pretax_income"]
    rejected_ids = set(rejected.values())

    # The alias produced nothing, so it is not among the ranked candidates.
    assert PRETAX_ALTERNATE not in {entry["source_concept"] for entry in pretax["alias_candidates"]}
    # ...but every fact it examined and rejected is still named.
    assert rejected_ids <= set(pretax["unusable_alias_source_fact_ids"])

    assessed = set(selection["assessed_evidence_fact_ids"])
    selected = set(selection["selected_input_fact_ids"])
    described = {entry["id"]: entry for entry in selection["assessed_evidence"]}
    asset_ids = {str(asset.pk) for asset in forecast.source_assets}

    assert rejected_ids <= assessed
    assert not (rejected_ids & selected)
    assert rejected_ids <= set(described)
    # Both the companyfacts and the filing asset are provable.
    assert str(companyfacts.pk) in asset_ids
    assert str(filing.pk) in asset_ids
    for fact_id in rejected_ids:
        assert described[fact_id]["source_asset_id"] == str(companyfacts.pk)
        assert described[fact_id]["filing_evidence_asset_id"] == str(filing.pk)


# ---------------------------------------------------------------------------
# RI-1: a content reversion is selected by its own observation, end to end
# ---------------------------------------------------------------------------


AUGUST = datetime(2026, 8, 15, 12, tzinfo=UTC)
SEPTEMBER = datetime(2026, 9, 20, 12, tzinfo=UTC)
OCTOBER = datetime(2026, 10, 18, 12, tzinfo=UTC)
REVERSION_DECISION_TIME = datetime(2026, 12, 1, 12, tzinfo=UTC)
AUDIT_ARGS_TARGET = TARGET_DATE

#: The quarter `_company_evidence` files last, and the one this chain
#: restates. Its original acceptance is the fixture's own Q4 availability.
Q4_2025 = (date(2025, 10, 1), date(2025, 12, 31))
Q4_ACCEPTANCE = datetime(2026, 2, 14, tzinfo=UTC)


def _reversion_chain(ticker: str) -> tuple[Listing, Listing, Any, dict[str, FundamentalFact]]:
    """Build the persisted shape a 100 -> 101 -> 100 ingestion produces.

    Revision 3 is a *content reversion*: its bytes deduplicated onto the
    original August asset, so it shares that asset and repeats revision 1's
    observation identity, while its availability is bound to the October
    observation event that actually carried it.
    """
    target = _listing(ticker)
    peer = _listing(f"{ticker}P")
    target_price = _company_evidence(target, sic="3571", sec_asset_retrieved_at=AUGUST)
    peer_price = _company_evidence(peer, sic="3571", scale=1.1, sec_asset_retrieved_at=AUGUST)

    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="operating_cash_flow",
        period_start=Q4_2025[0],
        period_end=Q4_2025[1],
    )
    august_companyfacts = original.source_asset
    august_filing = original.evidence_links.get(
        role=FundamentalFactEvidence.Role.FILING
    ).source_asset

    september_companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{ticker}-september-companyfacts",
        retrieved_at=SEPTEMBER,
    )
    corrected = _fact(
        target,
        september_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("35"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=SEPTEMBER,
        accession=f"{ticker}-operating_cash_flow-Q4",
        source_revision=2,
        acceptance_at=Q4_ACCEPTANCE,
        availability_basis=CORRECTION_AVAILABILITY_BASIS,
    )
    # Exact raw-content reuse: the same August asset, and therefore the same
    # observation identity as revision 1. The literal matches the fixture's
    # own Q4 operating cash flow so the persisted observation hash -- which
    # is computed from the pre-save value -- really does repeat.
    reverted = _fact(
        target,
        august_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("32.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=OCTOBER,
        accession=f"{ticker}-operating_cash_flow-Q4",
        source_revision=3,
        acceptance_at=Q4_ACCEPTANCE,
        availability_basis=CORRECTION_AVAILABILITY_BASIS,
    )
    assert original.value == Decimal("32.0")
    assert reverted.observation_hash == original.observation_hash
    assert reverted.source_asset_id == original.source_asset_id
    return (
        target,
        peer,
        (target_price, peer_price),
        {
            "original": original,
            "corrected": corrected,
            "reverted": reverted,
        },
    )


def _audit_selection(report: dict[str, Any], *, concept: str) -> dict[str, Any] | None:
    entry = report["listings"][0]
    for concept_entry in entry["ttm_alias_selection"]:
        if concept_entry["concept"] == concept:
            selection: dict[str, Any] | None = concept_entry["selection"]
            return selection
    raise AssertionError(f"{concept} missing from the audit report")


def _audit_referenced_fact_ids(report: dict[str, Any], *, concept: str) -> set[str]:
    selection = _audit_selection(report, concept=concept)
    assert selection is not None, f"{concept} has no alias selection in this report"
    return _referenced_fact_ids(selection)


def _audit_selected_fact_ids(report: dict[str, Any], *, concept: str) -> set[str]:
    """Only the lineage the audited TTM window actually selected.

    Deliberately narrower than `_audit_referenced_fact_ids`: a superseded
    revision is still legitimately *referenced* as assessed evidence, so the
    reference set cannot show which revision won.
    """
    selection = _audit_selection(report, concept=concept)
    assert selection is not None, f"{concept} has no alias selection in this report"
    return _referenced_fact_ids(
        {
            "selected_quarter_lineage": selection.get("selected_quarter_lineage", []),
            "controlling_source_fact": selection.get("controlling_source_fact"),
        }
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        # Between the September and October observations the correction stands.
        (datetime(2026, 10, 1, 12, tzinfo=UTC), "corrected"),
        # Only after the October observation does the reversion apply.
        (datetime(2026, 10, 19, 12, tzinfo=UTC), "reverted"),
    ],
)
def test_reversion_selection_follows_the_observation_in_forecast_and_audit(
    cutoff: datetime,
    expected: str,
) -> None:
    """The forecast and the audit agree, and both follow the observation.

    Revision 3 shares revision 1's asset and content identity, so anything
    reading `DataAsset.retrieved_at` would date it to August and select it a
    month early. Both readers must instead honour the October observation
    bound into its availability.
    """
    target, peer, prices, chain = _reversion_chain(f"REV{cutoff.day:02d}")
    expected_fact = chain[expected]
    other = chain["reverted" if expected == "corrected" else "corrected"]

    forecast = _forecast_at(
        target=target,
        peer=peer,
        prices=prices,
        config_path=V3_PATH,
        data_cutoff=cutoff,
        decision_time=REVERSION_DECISION_TIME,
    )
    selection = _assert_manifest_closes(forecast)

    assert forecast.scenario.base is not None
    assert str(expected_fact.pk) in selection["selected_input_fact_ids"]
    assert str(other.pk) not in selection["selected_input_fact_ids"]
    # Nothing here is a deferral: each revision is bound to a real event.
    assert selection["deferred_unproven_corrections"] == []

    report = audit_long_evidence(
        listing_ids=[str(target.pk)],
        target_date=AUDIT_ARGS_TARGET,
        available_through=cutoff,
        decision_time=REVERSION_DECISION_TIME,
        config=load_long_forecast_config(V3_PATH),
    )
    audited = report["listings"][0]

    assert audited["status"] == "audited"
    assert audited["deferred_unproven_corrections"] == []
    selected = _audit_selected_fact_ids(report, concept="operating_cash_flow")
    assert str(expected_fact.pk) in selected
    assert str(other.pk) not in selected
    # A superseded revision that is visible stays assessed, never selected.
    if str(other.pk) in audited["visible_fact_ids"]:
        assert str(other.pk) in _audit_referenced_fact_ids(
            report,
            concept="operating_cash_flow",
        )

    # Immutable rows are untouched by either reader.
    for fact in chain.values():
        before = (fact.value, fact.available_at, fact.availability_basis)
        fact.refresh_from_db()
        assert (fact.value, fact.available_at, fact.availability_basis) == before


# ---------------------------------------------------------------------------
# RI-3: the correction policy is config-gated in the audit exactly as in the
# forecast, so a frozen version is never audited against a policy it lacks
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.parametrize("config_path", [V1_PATH, V2_PATH, V3_PATH])
def test_audit_and_forecast_share_one_config_gated_correction_policy(
    config_path: Path,
) -> None:
    """Frozen versions must be audited against their own frozen selection.

    A legacy backdated correction is visible at the cutoff by recorded
    availability alone. long-v1 and long-v2 read exactly that, so both their
    forecast and their audit must still select it and must report no
    deferral. Only long-v3 resolves it against the retrieval that proved it.
    """
    label = config_path.stem.rsplit("-", maxsplit=1)[-1]
    target, peer, prices, correction, _assets = _backdated_correction_fixture(f"POLICY{label}")
    config = _small_peer_config(load_long_forecast_config(config_path))
    frozen = config.version in {"us-sec-long-v1", "us-sec-long-v2"}

    forecast = _forecast_at(
        target=target,
        peer=peer,
        prices=prices,
        config_path=config_path,
        data_cutoff=HISTORICAL_CUTOFF,
    )
    report = audit_long_evidence(
        listing_ids=[str(target.pk)],
        target_date=TARGET_DATE,
        available_through=HISTORICAL_CUTOFF,
        decision_time=LATER_DECISION_TIME,
        config=config,
    )
    audited = report["listings"][0]
    input_fact_ids = {fact["id"] for fact in forecast.calculation["input_facts"]}

    assert report["correction_availability_policy"] == audited["correction_availability_policy"]

    if frozen:
        assert report["correction_availability_policy"] == CORRECTION_POLICY_RECORDED_ONLY
        assert audited["deferred_unproven_corrections"] == []
        assert "evidence_selection" not in forecast.calculation
        # The frozen selection reads recorded availability, so it keeps the
        # backdated correction -- in the forecast and in the audit alike.
        assert str(correction.pk) in input_fact_ids
        assert str(correction.pk) in audited["visible_fact_ids"]
        # Nothing was withheld from the audited series either.
        assert audited["assessed_fact_count"] == len(audited["visible_fact_ids"])
        # Alias selection is a long-v3 capability, so the frozen audit
        # reports no alias lineage rather than inventing one.
        assert _audit_selection(report, concept="operating_cash_flow") is None
        return

    assert report["correction_availability_policy"] == CORRECTION_POLICY_PROVEN_OBSERVATION
    assert [entry["fact_id"] for entry in audited["deferred_unproven_corrections"]] == [
        str(correction.pk)
    ]
    assert str(correction.pk) not in input_fact_ids
    assert str(correction.pk) not in _audit_referenced_fact_ids(
        report,
        concept="operating_cash_flow",
    )


@pytest.mark.django_db
def test_v3_audit_admits_the_correction_once_its_retrieval_is_proven() -> None:
    """The audit's deferral is point-in-time, exactly like the forecast's."""
    target, _peer, _prices, correction, _assets = _backdated_correction_fixture("POLICYOK")
    proven_cutoff = datetime(2026, 4, 15, 12, tzinfo=UTC)

    report = audit_long_evidence(
        listing_ids=[str(target.pk)],
        target_date=TARGET_DATE,
        available_through=proven_cutoff,
        decision_time=LATER_DECISION_TIME,
        config=load_long_forecast_config(V3_PATH),
    )
    audited = report["listings"][0]

    assert audited["deferred_unproven_corrections"] == []
    assert str(correction.pk) in _audit_referenced_fact_ids(
        report,
        concept="operating_cash_flow",
    )


@pytest.mark.django_db
def test_a_legacy_reversion_chain_is_unprovable_and_never_admitted() -> None:
    """Legacy rows predate the observation event, so a reversion has no clock.

    Before `CORRECTION_AVAILABILITY_BASIS` existed, a 100 -> 101 -> 100 chain
    persisted revision 3 against the *reused* original asset and the original
    acceptance. Nothing in those rows records when revision 3 was actually
    seen, so no cutoff can justify it. It is deferred at every boundary --
    conservatively, and without rewriting any row -- while revision 2, whose
    own retrieval is provable, is admitted once its cutoff passes.
    """
    target = _listing("LEGREV")
    peer = _listing("LEGREVP")
    target_price = _company_evidence(target, sic="3571", sec_asset_retrieved_at=AUGUST)
    peer_price = _company_evidence(peer, sic="3571", scale=1.1, sec_asset_retrieved_at=AUGUST)

    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="operating_cash_flow",
        period_start=Q4_2025[0],
        period_end=Q4_2025[1],
    )
    august_companyfacts = original.source_asset
    august_filing = original.evidence_links.get(
        role=FundamentalFactEvidence.Role.FILING
    ).source_asset
    september_companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{target.ticker}-legacy-september",
        retrieved_at=SEPTEMBER,
    )

    # Legacy shape: acceptance-dated availability, no correction basis.
    corrected = _fact(
        target,
        september_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("35.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=Q4_ACCEPTANCE,
        accession=f"{target.ticker}-operating_cash_flow-Q4",
        source_revision=2,
    )
    reverted = _fact(
        target,
        august_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("32.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=Q4_ACCEPTANCE,
        accession=f"{target.ticker}-operating_cash_flow-Q4",
        source_revision=3,
    )
    assert reverted.observation_hash == original.observation_hash

    resolved = resolve_availability([original, corrected, reverted])
    assert resolved[str(original.pk)].proven_at == Q4_ACCEPTANCE
    # Revision 2 is bounded by the retrieval of the asset it came from.
    assert resolved[str(corrected.pk)].proven_at == SEPTEMBER
    assert resolved[str(corrected.pk)].basis == RESOLUTION_LEGACY_ASSET_RETRIEVAL
    # Revision 3 shares revision 1's asset, so it has no clock at all.
    assert resolved[str(reverted.pk)].proven_at is None
    assert resolved[str(reverted.pk)].basis == RESOLUTION_UNPROVABLE_LEGACY_REVERSION

    # Deferred at every boundary, including one long after every retrieval.
    for cutoff in (
        datetime(2026, 10, 1, 12, tzinfo=UTC),
        datetime(2027, 6, 1, 12, tzinfo=UTC),
    ):
        forecast = _forecast_at(
            target=target,
            peer=peer,
            prices=(target_price, peer_price),
            config_path=V3_PATH,
            data_cutoff=cutoff,
            decision_time=datetime(2027, 7, 1, 12, tzinfo=UTC),
        )
        selection = _assert_manifest_closes(forecast)
        deferred = {entry["fact_id"]: entry for entry in selection["deferred_unproven_corrections"]}

        assert str(reverted.pk) in deferred
        assert deferred[str(reverted.pk)]["proven_available_at"] is None
        assert (
            deferred[str(reverted.pk)]["resolution_basis"] == RESOLUTION_UNPROVABLE_LEGACY_REVERSION
        )
        assert "no persisted observation proves" in deferred[str(reverted.pk)]["reason"]
        # Assessed evidence, never a selected input.
        assert str(reverted.pk) in selection["assessed_evidence_fact_ids"]
        assert str(reverted.pk) not in selection["selected_input_fact_ids"]
        # Revision 2 is provable and stands in its place.
        assert str(corrected.pk) not in deferred
        assert str(corrected.pk) in selection["selected_input_fact_ids"]

    # No row was mutated by any resolution.
    for fact, expected in ((corrected, Decimal("35.0")), (reverted, Decimal("32.0"))):
        fact.refresh_from_db()
        assert fact.value == expected
        assert fact.available_at == Q4_ACCEPTANCE
        assert fact.availability_basis == "acceptance_datetime"


@pytest.mark.django_db
def test_a_legacy_revision_from_an_earlier_asset_is_unresolved_not_lower_bounded() -> None:
    """A chain-ordering lower bound is context, never an admission boundary.

    Revision 3 carries *distinct* content -- so it is not a reversion -- but
    the asset it came from was retrieved in August, before revision 2 became
    knowable in September. That retrieval therefore cannot be an observation
    of revision 3. Ordering only proves revision 3 is no earlier than
    September; it never proves revision 3 existed *by* September, so revision
    3 stays deferred at every cutoff, including ones long after its
    predecessor's boundary.
    """
    target = _listing("LEGORD")
    peer = _listing("LEGORDP")
    target_price = _company_evidence(target, sic="3571", sec_asset_retrieved_at=AUGUST)
    peer_price = _company_evidence(peer, sic="3571", scale=1.1, sec_asset_retrieved_at=AUGUST)

    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="operating_cash_flow",
        period_start=Q4_2025[0],
        period_end=Q4_2025[1],
    )
    august_companyfacts = original.source_asset
    august_filing = original.evidence_links.get(
        role=FundamentalFactEvidence.Role.FILING
    ).source_asset
    september_companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{target.ticker}-legacy-ordering-september",
        retrieved_at=SEPTEMBER,
    )

    corrected = _fact(
        target,
        september_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("35.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=Q4_ACCEPTANCE,
        accession=f"{target.ticker}-operating_cash_flow-Q4",
        source_revision=2,
    )
    # Distinct content, so no hash repeats -- but an *earlier* asset.
    out_of_order = _fact(
        target,
        august_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("33.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=Q4_ACCEPTANCE,
        accession=f"{target.ticker}-operating_cash_flow-Q4",
        source_revision=3,
    )
    assert out_of_order.observation_hash != corrected.observation_hash
    assert out_of_order.observation_hash != original.observation_hash
    assert out_of_order.source_asset.retrieved_at < corrected.source_asset.retrieved_at

    resolved = resolve_availability([original, corrected, out_of_order])
    assert resolved[str(corrected.pk)].proven_at == SEPTEMBER
    assert resolved[str(corrected.pk)].basis == RESOLUTION_LEGACY_ASSET_RETRIEVAL
    entry = resolved[str(out_of_order.pk)]
    assert entry.proven_at is None
    assert entry.basis == RESOLUTION_UNPROVABLE_LEGACY_ORDERING
    # The ordering bound is retained, but only as context.
    assert entry.earliest_possible_at == SEPTEMBER

    for cutoff in (
        # Exactly the predecessor's boundary...
        SEPTEMBER,
        # ...and long after it. Neither proves revision 3.
        datetime(2027, 6, 1, 12, tzinfo=UTC),
    ):
        forecast = _forecast_at(
            target=target,
            peer=peer,
            prices=(target_price, peer_price),
            config_path=V3_PATH,
            data_cutoff=cutoff,
            decision_time=datetime(2027, 7, 1, 12, tzinfo=UTC),
        )
        selection = _assert_manifest_closes(forecast)
        deferred = {entry["fact_id"]: entry for entry in selection["deferred_unproven_corrections"]}

        assert str(out_of_order.pk) in deferred
        assert deferred[str(out_of_order.pk)]["proven_available_at"] is None
        assert (
            deferred[str(out_of_order.pk)]["resolution_basis"]
            == RESOLUTION_UNPROVABLE_LEGACY_ORDERING
        )
        # The lower bound is reported as context and admits nothing.
        assert (
            deferred[str(out_of_order.pk)]["earliest_possible_available_at"]
            == SEPTEMBER.isoformat()
        )
        assert "before the revision it supersedes" in deferred[str(out_of_order.pk)]["reason"]
        # Assessed and manifest-covered, never selected.
        assert str(out_of_order.pk) in selection["assessed_evidence_fact_ids"]
        assert str(out_of_order.pk) in selection["manifest_evidence_fact_ids"]
        assert str(out_of_order.pk) not in selection["selected_input_fact_ids"]
        assert str(out_of_order.pk) in {entry["id"] for entry in selection["assessed_evidence"]}
        # The provable predecessor stands in its place.
        assert str(corrected.pk) in selection["selected_input_fact_ids"]

    # Frozen versions are untouched by any of this.
    for config_path in (V1_PATH, V2_PATH):
        frozen = _forecast_at(
            target=target,
            peer=peer,
            prices=(target_price, peer_price),
            config_path=config_path,
            data_cutoff=datetime(2027, 6, 1, 12, tzinfo=UTC),
            decision_time=datetime(2027, 7, 1, 12, tzinfo=UTC),
        )
        assert "evidence_selection" not in frozen.calculation
        # Recorded availability alone, so the highest revision still wins.
        assert str(out_of_order.pk) in {fact["id"] for fact in frozen.calculation["input_facts"]}

    for fact, expected in ((corrected, Decimal("35.0")), (out_of_order, Decimal("33.0"))):
        fact.refresh_from_db()
        assert fact.value == expected
        assert fact.available_at == Q4_ACCEPTANCE
        assert fact.availability_basis == "acceptance_datetime"


REBOUND_OBSERVATION = datetime(2027, 1, 20, 12, tzinfo=UTC)


@pytest.mark.django_db
def test_a_rebound_vintage_restores_selection_after_fresh_proof() -> None:
    """Fresh proof of an unprovable legacy reversion changes what v3 selects.

    Before the fresh observation, revision 3 has no provable timing, so v3
    defers it and selects the superseded revision 2. Ingestion then appends
    revision 4 -- identical value and observation hash, but bound to a real
    October/January observation -- and from that boundary onward v3 selects
    the reverted value again. Revision 3 stays deferred and assessed
    throughout; nothing about it is rewritten.
    """
    target = _listing("REBIND")
    peer = _listing("REBINDP")
    target_price = _company_evidence(target, sic="3571", sec_asset_retrieved_at=AUGUST)
    peer_price = _company_evidence(peer, sic="3571", scale=1.1, sec_asset_retrieved_at=AUGUST)

    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="operating_cash_flow",
        period_start=Q4_2025[0],
        period_end=Q4_2025[1],
    )
    august_companyfacts = original.source_asset
    august_filing = original.evidence_links.get(
        role=FundamentalFactEvidence.Role.FILING
    ).source_asset
    september_companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{target.ticker}-rebind-september",
        retrieved_at=SEPTEMBER,
    )
    fresh_companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{target.ticker}-rebind-fresh",
        retrieved_at=REBOUND_OBSERVATION,
    )
    accession = f"{target.ticker}-operating_cash_flow-Q4"

    # Legacy chain: revision 2 provable, revision 3 an unprovable reversion.
    corrected = _fact(
        target,
        september_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("35.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=Q4_ACCEPTANCE,
        accession=accession,
        source_revision=2,
    )
    reverted = _fact(
        target,
        august_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("32.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=Q4_ACCEPTANCE,
        accession=accession,
        source_revision=3,
    )
    assert reverted.observation_hash == original.observation_hash

    before_cutoff = datetime(2026, 12, 1, 12, tzinfo=UTC)
    decision_time = datetime(2027, 3, 1, 12, tzinfo=UTC)

    def _selection(cutoff: datetime) -> dict[str, Any]:
        forecast = _forecast_at(
            target=target,
            peer=peer,
            prices=(target_price, peer_price),
            config_path=V3_PATH,
            data_cutoff=cutoff,
            decision_time=decision_time,
        )
        return _assert_manifest_closes(forecast)

    # Before fresh proof: revision 3 deferred, superseded revision 2 selected.
    stale = _selection(before_cutoff)
    assert str(reverted.pk) in {
        entry["fact_id"] for entry in stale["deferred_unproven_corrections"]
    }
    assert str(corrected.pk) in stale["selected_input_fact_ids"]

    # Ingestion appends the observation-bound vintage: same value and hash.
    rebound = _fact(
        target,
        fresh_companyfacts,
        august_filing,
        concept="operating_cash_flow",
        value=Decimal("32.0"),
        start=Q4_2025[0],
        end=Q4_2025[1],
        fiscal_period="Q4",
        available_at=REBOUND_OBSERVATION,
        accession=accession,
        source_revision=4,
        acceptance_at=Q4_ACCEPTANCE,
        availability_basis=CORRECTION_AVAILABILITY_BASIS,
    )
    assert rebound.observation_hash == reverted.observation_hash

    resolved = resolve_availability([original, corrected, reverted, rebound])
    assert resolved[str(reverted.pk)].proven_at is None
    assert resolved[str(rebound.pk)].proven_at == REBOUND_OBSERVATION
    assert resolved[str(rebound.pk)].basis == RESOLUTION_BOUND_OBSERVATION

    # The earlier cutoff is completely unchanged by the new vintage.
    unchanged = _selection(before_cutoff)
    assert unchanged["selected_input_fact_ids"] == stale["selected_input_fact_ids"]
    assert str(rebound.pk) not in unchanged["manifest_evidence_fact_ids"]

    # From the fresh boundary onward the reverted value is selected again.
    after_cutoff = REBOUND_OBSERVATION + timedelta(days=1)
    proven = _selection(after_cutoff)
    assert str(rebound.pk) in proven["selected_input_fact_ids"]
    assert str(corrected.pk) not in proven["selected_input_fact_ids"]
    # Revision 3 remains deferred and assessed, never selected.
    assert str(reverted.pk) in {
        entry["fact_id"] for entry in proven["deferred_unproven_corrections"]
    }
    assert str(reverted.pk) in proven["assessed_evidence_fact_ids"]
    assert str(reverted.pk) not in proven["selected_input_fact_ids"]

    # The audit agrees with the forecast at both boundaries.
    for cutoff, expected in ((before_cutoff, corrected), (after_cutoff, rebound)):
        report = audit_long_evidence(
            listing_ids=[str(target.pk)],
            target_date=TARGET_DATE,
            available_through=cutoff,
            decision_time=decision_time,
            config=load_long_forecast_config(V3_PATH),
        )
        selected = _audit_selected_fact_ids(report, concept="operating_cash_flow")
        assert str(expected.pk) in selected
        assert str(reverted.pk) not in selected

    # Frozen versions read recorded availability and are unaffected.
    for config_path in (V1_PATH, V2_PATH):
        frozen = _forecast_at(
            target=target,
            peer=peer,
            prices=(target_price, peer_price),
            config_path=config_path,
            data_cutoff=after_cutoff,
            decision_time=decision_time,
        )
        assert "evidence_selection" not in frozen.calculation
        assert str(rebound.pk) in {fact["id"] for fact in frozen.calculation["input_facts"]}

    # Nothing persisted was rewritten.
    for fact, expected_value, expected_available in (
        (corrected, Decimal("35.0"), Q4_ACCEPTANCE),
        (reverted, Decimal("32.0"), Q4_ACCEPTANCE),
        (rebound, Decimal("32.0"), REBOUND_OBSERVATION),
    ):
        fact.refresh_from_db()
        assert fact.value == expected_value
        assert fact.available_at == expected_available


def test_correction_policy_is_driven_by_its_own_declared_capability() -> None:
    """The policy reads one explicit key, never an inference from the others.

    A future version must be able to adopt alias selection or joint pair
    selection *without* silently acquiring correction-availability
    resolution, so the capability is declared and hashed on its own.
    """
    v3 = load_long_forecast_config(V3_PATH)

    assert correction_availability_policy(v3) == CORRECTION_POLICY_PROVEN_OBSERVATION
    for config_path in (V1_PATH, V2_PATH):
        frozen = load_long_forecast_config(config_path)
        assert frozen.proven_observation_correction_availability is None
        assert correction_availability_policy(frozen) == CORRECTION_POLICY_RECORDED_ONLY

    # The other two capabilities alone do not turn it on.
    without = deepcopy(v3.raw)
    without.pop("proven_observation_correction_availability")
    partial = LongForecastConfig.from_mapping(without)
    assert partial.newest_quarter_anchored_homogeneous_ttm_alias_selection is True
    assert partial.joint_compatible_invested_capital_pair_selection is True
    assert correction_availability_policy(partial) == CORRECTION_POLICY_RECORDED_ONLY

    # Declaring it explicitly off is a distinct configuration and a distinct hash.
    explicit_off = deepcopy(v3.raw)
    explicit_off["proven_observation_correction_availability"] = {"enabled": False}
    disabled = LongForecastConfig.from_mapping(explicit_off)
    assert disabled.proven_observation_correction_availability is False
    assert correction_availability_policy(disabled) == CORRECTION_POLICY_RECORDED_ONLY
    assert long_forecast_config_hash(disabled) != V3_CONFIG_HASH


def test_the_same_date_combination_ceiling_is_configured_and_not_tunable() -> None:
    """256 is a reviewed bound declared by the config, accepted only as 256."""
    v3 = load_long_forecast_config(V3_PATH)

    assert v3.maximum_same_date_source_combinations == 256
    assert same_date_combination_ceiling(v3) == 256

    for bad in (255, 257, 1, 512):
        mapping = deepcopy(v3.raw)
        mapping["maximum_same_date_source_combinations"] = bad
        with pytest.raises(ValueError, match="exactly 256"):
            LongForecastConfig.from_mapping(mapping)

    # Enabling the joint search without declaring the ceiling is refused, so
    # the enumeration can never fall back to an implicit default.
    missing = deepcopy(v3.raw)
    missing.pop("maximum_same_date_source_combinations")
    with pytest.raises(ValueError, match="maximum_same_date_source_combinations"):
        LongForecastConfig.from_mapping(missing)

    # Declaring it without the joint search is equally refused.
    unused = deepcopy(load_long_forecast_config(V2_PATH).raw)
    unused["maximum_same_date_source_combinations"] = 256
    with pytest.raises(ValueError, match="only applies when"):
        LongForecastConfig.from_mapping(unused)

    # A frozen version has no ceiling at all, and asking for one is explicit.
    with pytest.raises(ValueError, match="does not declare"):
        same_date_combination_ceiling(load_long_forecast_config(V2_PATH))
