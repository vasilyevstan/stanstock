from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

import stanstock.research.long_forecasts as long_forecasts_module
from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.management.config_loader import default_us_scoring_config_path
from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    ProviderRecord,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.research.long_forecast_config import (
    LongForecastConfig,
    load_long_forecast_config,
    long_forecast_config_hash,
)
from stanstock.research.long_forecasts import build_long_forecasts
from stanstock.research.models import Prediction
from stanstock.research.opportunities import assess_opportunity
from stanstock.research.service import _long_forecast_evidence_grade, analyze_snapshot

TARGET_DATE = date(2026, 2, 27)
DECISION_TIME = datetime(2026, 3, 1, 12, tzinfo=UTC)

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
    "weighted_average_diluted_shares": ("us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"),
    "operating_cash_flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditure": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "cash_and_equivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "long_term_debt": "us-gaap:LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    "reported_long_term_debt": "us-gaap:LongTermDebt",
    "equity": "us-gaap:StockholdersEquity",
}

UNITS = {
    "diluted_eps": "USD/shares",
    "weighted_average_diluted_shares": "shares",
}


def test_long_forecast_config_is_frozen_versioned_and_stable() -> None:
    first = load_long_forecast_config()
    second = load_long_forecast_config()

    assert first.version == "us-sec-long-v1"
    assert first.probability_positive_enabled is False
    assert first.return_basis == "split_adjusted_price_return"
    assert first.dividends_included is False
    assert first.horizons["3y"].years == 3
    assert first.horizons["5y"].years == 5
    assert len(first.horizons["3y"].fade) == 3
    assert len(first.horizons["5y"].fade) == 5
    assert (
        first.growth.historical_weight + first.growth.sustainable_weight + first.growth.peer_weight
    ) == pytest.approx(1.0)
    assert first.peer.sic_prefix_levels == (4, 3, 2)
    assert first.peer.minimum_peers == {4: 3, 3: 5, 2: 8}
    assert long_forecast_config_hash(first) == long_forecast_config_hash(second)
    assert long_forecast_config_hash(
        replace(
            first,
            eligibility=replace(first.eligibility, maximum_metric_age_days=199),
        )
    ) != long_forecast_config_hash(first)

    for attribute in ("growth_delta", "reinvestment_multiplier", "peer_multiple_multiplier"):
        assert (
            getattr(first.scenarios["bear"], attribute)
            <= getattr(first.scenarios["base"], attribute)
            <= getattr(first.scenarios["bull"], attribute)
        )


def test_long_forecast_config_rejects_probability_weight_and_horizon_drift() -> None:
    config = load_long_forecast_config()

    probability = deepcopy(config.raw)
    probability["probability_positive_enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        LongForecastConfig.from_mapping(probability)

    weights = deepcopy(config.raw)
    weights["growth"]["peer_weight"] = 0.30
    with pytest.raises(ValueError, match="must sum to 1"):
        LongForecastConfig.from_mapping(weights)

    horizon = deepcopy(config.raw)
    horizon["horizons"]["3y"]["fade"] = [0.7, 0.4]
    with pytest.raises(ValueError, match="fade must contain one value per year"):
        LongForecastConfig.from_mapping(horizon)

    annual_history = deepcopy(config.raw)
    annual_history["metric_families"]["fcf_per_share"]["minimum_annual_periods"] = 1
    with pytest.raises(ValueError, match="integer of at least 2"):
        LongForecastConfig.from_mapping(annual_history)

    share_history = deepcopy(config.raw)
    share_history["eligibility"]["minimum_share_consistency_periods"] = 2
    with pytest.raises(ValueError, match="cover every selected annual period"):
        LongForecastConfig.from_mapping(share_history)

    tolerance = deepcopy(config.raw)
    tolerance["eligibility"]["share_consistency_relative_tolerance"] = 0.16
    with pytest.raises(ValueError, match=r"\(0, 0.15\]"):
        LongForecastConfig.from_mapping(tolerance)


@pytest.mark.django_db
def test_fcf_forecast_uses_exact_sic_fallback_and_complete_provenance() -> None:
    config = _small_peer_config()
    target = _listing("FCFT")
    peer = _listing("FCFP")
    target_price = _company_evidence(target, family="fcf_per_share", sic="3571")
    peer_price = _company_evidence(peer, family="fcf_per_share", sic="3572", scale=1.1)
    peer_share_class = Listing.objects.create(
        security=peer.security,
        ticker="FCFPB",
        provider_symbol="FCFPB",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    peer_share_class_price = _asset(
        provider="twelve_data",
        kind="price_history",
        subject=peer_share_class.ticker,
        retrieved_at=DECISION_TIME,
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )

    forecasts = build_long_forecasts(
        listings=[target, peer, peer_share_class],
        current_prices={
            str(target.pk): 50.0,
            str(peer.pk): 55.0,
            str(peer_share_class.pk): 55.0,
        },
        price_assets={
            str(target.pk): target_price,
            str(peer.pk): peer_price,
            str(peer_share_class.pk): peer_share_class_price,
        },
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )

    for horizon, years in (("3y", 3), ("5y", 5)):
        forecast = forecasts[str(target.pk)][horizon]
        scenario = forecast.scenario
        assert scenario.bear is not None
        assert scenario.base is not None
        assert scenario.bull is not None
        assert scenario.bear <= scenario.base <= scenario.bull
        assert scenario.probability_positive is None
        assert forecast.calculation["metric_family"] == "fcf_per_share"
        assert forecast.calculation["support"]["peer_count"] == 1
        assert forecast.calculation["support"]["sic_fallback_level"] == 3
        assert forecast.calculation["support"]["sic_prefix"] == "357"
        assert forecast.calculation["support"]["annual_periods"] == 3
        assert forecast.calculation["support"]["share_consistency_periods"] == 3
        peer_payload = forecast.calculation["peer_set"][0]
        assert peer_payload["listing_id"] == str(peer.pk)
        assert peer_payload["classification"]["available_at"]
        assert peer_payload["fact_references"]
        assert "input_facts" not in peer_payload
        assert all(
            item["available_at"] and item["source_asset_id"] and item["filing_evidence_asset_id"]
            for item in peer_payload["fact_references"]
        )
        assert all(
            "concept" in item and "value" not in item for item in peer_payload["fact_references"]
        )
        assert forecast.calculation["formula_inputs"]["peer_growth"] == pytest.approx(
            peer_payload["historical_growth"]
        )
        assert forecast.calculation["formula_inputs"]["peer_multiple"] == pytest.approx(
            peer_payload["current_multiple_capped"]
        )
        assert forecast.calculation["dividends_included"] is False
        assert forecast.calculation["fundamentals_config_version"] == ("us-sec-fundamentals-v1")
        assert forecast.calculation["fundamentals_config_hash"]
        assert forecast.calculation["config_hash"] == long_forecast_config_hash(config)
        assert forecast.calculation["target_price_asset_id"] == str(target_price.pk)
        assert len(forecast.calculation["scenario_paths"]["base"]["annual_growth"]) == years
        annualized = forecast.calculation["annualized_returns"]["base"]
        assert annualized == pytest.approx((1.0 + scenario.base) ** (1.0 / years) - 1.0)

        input_facts = forecast.calculation["input_facts"]
        assert input_facts
        assert all(item["accession"] for item in input_facts)
        assert all(item["available_at"] for item in input_facts)
        assert all(item["source_asset_id"] for item in input_facts)
        assert all(item["filing_evidence_asset_id"] for item in input_facts)
        assert all(
            {
                "provider",
                "taxonomy",
                "period_identity",
                "filing_form",
                "filing_date",
                "acceptance_at",
                "availability_basis",
                "observation_hash",
                "quality_flags",
            }
            <= set(item)
            for item in input_facts
        )
        assert forecast.calculation["target_classification"]["observed_at"]
        assert "quality_flags" in forecast.calculation["target_classification"]
        split_basis = forecast.calculation["split_basis"]
        assert split_basis["verified_through"] == "2025-12-31"
        assert split_basis["post_period_exposure_days"] == 58
        assert split_basis["maximum_exposure_days"] == 200
        assert split_basis["continuity_tolerance"] == pytest.approx(0.15)
        assert split_basis["residual_risk"] == "unverified_post_period_split"
        assert forecast.scenario_payload()["split_basis"] == split_basis
        source_ids = {str(asset.pk) for asset in forecast.source_assets}
        assert str(target_price.pk) in source_ids
        assert str(peer_price.pk) in source_ids
        assert {item["source_asset_id"] for item in input_facts} <= source_ids
        assert {item["filing_evidence_asset_id"] for item in input_facts} <= source_ids
        assert {item["source_asset_id"] for item in peer_payload["fact_references"]} <= source_ids
        assert {
            item["filing_evidence_asset_id"] for item in peer_payload["fact_references"]
        } <= source_ids


@pytest.mark.django_db
def test_return_uses_the_actual_current_multiple_when_reversion_anchor_is_bounded() -> None:
    config = _small_peer_config()
    target = _listing("VALUET")
    peer = _listing("VALUEP")
    target_price = _company_evidence(target, family="fcf_per_share", sic="3571")
    peer_price = _company_evidence(peer, family="fcf_per_share", sic="3571")

    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 10_000.0, str(peer.pk): 50.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(target.pk)]["5y"]

    inputs = forecast.calculation["formula_inputs"]
    path = forecast.calculation["scenario_paths"]["base"]
    assert inputs["current_multiple_raw"] > inputs["current_multiple_capped"]
    assert path["current_multiple_return_denominator"] == inputs["current_multiple_raw"]
    assert path["current_multiple_reversion_anchor"] == inputs["current_multiple_capped"]
    assert forecast.scenario.base == pytest.approx(
        path["fundamental_growth_factor"]
        * path["terminal_multiple"]
        / inputs["current_multiple_raw"]
        - 1.0
    )
    assert forecast.scenario.base != pytest.approx(
        path["fundamental_growth_factor"]
        * path["terminal_multiple"]
        / inputs["current_multiple_capped"]
        - 1.0
    )

    low_multiple = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 0.01, str(peer.pk): 50.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(target.pk)]["5y"]
    assert low_multiple.scenario.base is None
    assert "below the supported long-v1 minimum" in (low_multiple.scenario.insufficiency_reason)


@pytest.mark.django_db
def test_eps_branch_is_allowed_only_when_fcf_is_truly_unavailable() -> None:
    config = _small_peer_config()
    target = _listing("EPST")
    peer = _listing("EPSP")
    target_price = _company_evidence(target, family="eps_per_share", sic="3571")
    peer_price = _company_evidence(peer, family="eps_per_share", sic="3571", scale=1.1)

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
    assert forecast.calculation["metric_family"] == "eps_per_share"

    negative = _listing("NEGFCF")
    negative_price = _company_evidence(
        negative,
        family="fcf_per_share",
        sic="3571",
        negative_fcf=True,
        include_eps=True,
    )
    blocked = build_long_forecasts(
        listings=[negative, target, peer],
        current_prices={
            str(negative.pk): 25.0,
            str(target.pk): 50.0,
            str(peer.pk): 55.0,
        },
        price_assets={
            str(negative.pk): negative_price,
            str(target.pk): target_price,
            str(peer.pk): peer_price,
        },
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(negative.pk)]["3y"]

    assert blocked.scenario.base is None
    assert "FCF/share branch failed" in blocked.scenario.insufficiency_reason
    assert blocked.calculation["metric_family"] is None

    partial = _listing("PARTFCF")
    partial_price = _company_evidence(
        partial,
        family="eps_per_share",
        sic="3571",
        partial_fcf=True,
    )
    partial_blocked = build_long_forecasts(
        listings=[partial, target, peer],
        current_prices={
            str(partial.pk): 30.0,
            str(target.pk): 50.0,
            str(peer.pk): 55.0,
        },
        price_assets={
            str(partial.pk): partial_price,
            str(target.pk): target_price,
            str(peer.pk): peer_price,
        },
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(partial.pk)]["3y"]

    assert partial_blocked.scenario.base is None
    assert "FCF/share branch failed" in partial_blocked.scenario.insufficiency_reason
    assert any(
        fact["concept"] == "operating_cash_flow"
        for fact in partial_blocked.calculation["input_facts"]
    )


@pytest.mark.django_db
def test_share_basis_unsupported_sic_and_missing_terms_are_withheld() -> None:
    config = _small_peer_config()
    peer = _listing("VALID")
    peer_price = _company_evidence(peer, family="fcf_per_share", sic="3571")

    mismatch = _listing("SHARES")
    mismatch_price = _company_evidence(
        mismatch,
        family="fcf_per_share",
        sic="3571",
        reported_eps_multiplier=0.5,
    )
    financial = _listing("BANK")
    financial_price = _company_evidence(financial, family="fcf_per_share", sic="6021")
    no_sic = _listing("NOSIC")
    no_sic_price = _company_evidence(
        no_sic,
        family="fcf_per_share",
        sic=None,
    )
    no_tax = _listing("NOTAX")
    no_tax_price = _company_evidence(
        no_tax,
        family="fcf_per_share",
        sic="3571",
        include_tax=False,
    )
    stale_share_check = _listing("STALEEPS")
    stale_share_price = _company_evidence(
        stale_share_check,
        family="fcf_per_share",
        sic="3571",
        skip_latest_reported_eps=True,
    )
    ttm_share_drift = _listing("TTMSHARES")
    ttm_share_drift_price = _company_evidence(
        ttm_share_drift,
        family="fcf_per_share",
        sic="3571",
        quarter_share_multiplier=2.0,
    )
    reverse_share_drift = _listing("REVSHARES")
    reverse_share_drift_price = _company_evidence(
        reverse_share_drift,
        family="fcf_per_share",
        sic="3571",
        quarter_share_multiplier=0.5,
    )
    minor_share_drift = _listing("MINSHARES")
    minor_share_drift_price = _company_evidence(
        minor_share_drift,
        family="fcf_per_share",
        sic="3571",
        quarter_share_multiplier=0.92,
    )
    tiny_eps_mismatch = _listing("TINYEP")
    tiny_eps_mismatch_price = _company_evidence(
        tiny_eps_mismatch,
        family="fcf_per_share",
        sic="3571",
        net_income_multiplier=0.0001,
        reported_eps_multiplier=0.5,
    )

    listings = [
        peer,
        mismatch,
        financial,
        no_sic,
        no_tax,
        stale_share_check,
        ttm_share_drift,
        reverse_share_drift,
        minor_share_drift,
        tiny_eps_mismatch,
    ]
    prices = {str(listing.pk): 50.0 for listing in listings}
    price_assets = {
        str(peer.pk): peer_price,
        str(mismatch.pk): mismatch_price,
        str(financial.pk): financial_price,
        str(no_sic.pk): no_sic_price,
        str(no_tax.pk): no_tax_price,
        str(stale_share_check.pk): stale_share_price,
        str(ttm_share_drift.pk): ttm_share_drift_price,
        str(reverse_share_drift.pk): reverse_share_drift_price,
        str(minor_share_drift.pk): minor_share_drift_price,
        str(tiny_eps_mismatch.pk): tiny_eps_mismatch_price,
    }
    forecasts = build_long_forecasts(
        listings=listings,
        current_prices=prices,
        price_assets=price_assets,
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )

    assert "Share basis differs from reported diluted EPS" in (
        forecasts[str(mismatch.pk)]["3y"].scenario.insufficiency_reason
    )
    assert forecasts[str(mismatch.pk)]["3y"].calculation["input_facts"]
    assert "SEC SIC 6021 is outside" in (
        forecasts[str(financial.pk)]["3y"].scenario.insufficiency_reason
    )
    assert "No point-in-time SEC SIC classification" in (
        forecasts[str(no_sic.pk)]["3y"].scenario.insufficiency_reason
    )
    assert "Sustainable growth requires TTM income_tax_expense" in (
        forecasts[str(no_tax.pk)]["3y"].scenario.insufficiency_reason
    )
    assert forecasts[str(no_tax.pk)]["3y"].calculation["input_facts"]
    assert "selected metric history" in (
        forecasts[str(stale_share_check.pk)]["3y"].scenario.insufficiency_reason
    )
    assert "TTM diluted shares differ" in (
        forecasts[str(ttm_share_drift.pk)]["3y"].scenario.insufficiency_reason
    )
    assert "TTM diluted shares differ" in (
        forecasts[str(reverse_share_drift.pk)]["3y"].scenario.insufficiency_reason
    )
    assert forecasts[str(minor_share_drift.pk)]["3y"].scenario.base is not None
    assert "Share basis differs from reported diluted EPS" in (
        forecasts[str(tiny_eps_mismatch.pk)]["3y"].scenario.insufficiency_reason
    )


@pytest.mark.django_db
def test_price_basis_history_capital_and_peer_failures_are_explicit() -> None:
    config = _small_peer_config()
    wrong_provider = _listing("WRONGPX")
    _company_evidence(
        wrong_provider,
        family="fcf_per_share",
        sic="1010",
        create_price=False,
    )
    wrong_price = _asset(
        provider="synthetic",
        kind="price_history",
        subject=wrong_provider.ticker,
        retrieved_at=DECISION_TIME,
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )
    short_history = _listing("SHORT")
    short_price = _company_evidence(
        short_history,
        family="fcf_per_share",
        sic="2020",
        annual_period_count=2,
    )
    no_capital = _listing("NOCAP")
    no_capital_price = _company_evidence(
        no_capital,
        family="fcf_per_share",
        sic="3030",
        include_balance_sheet=False,
    )
    no_peers = _listing("LONELY")
    no_peers_price = _company_evidence(
        no_peers,
        family="fcf_per_share",
        sic="4040",
    )
    debt_basis = _listing("DEBTBASIS")
    debt_basis_price = _company_evidence(
        debt_basis,
        family="fcf_per_share",
        sic="5050",
        mismatched_debt_basis=True,
    )
    source_basis = _listing("SOURCEBASIS")
    source_basis_price = _company_evidence(
        source_basis,
        family="fcf_per_share",
        sic="5050",
        mismatched_source_basis=True,
    )
    listings = [
        wrong_provider,
        short_history,
        no_capital,
        no_peers,
        debt_basis,
        source_basis,
    ]

    forecasts = build_long_forecasts(
        listings=listings,
        current_prices={str(listing.pk): 50.0 for listing in listings},
        price_assets={
            str(wrong_provider.pk): wrong_price,
            str(short_history.pk): short_price,
            str(no_capital.pk): no_capital_price,
            str(no_peers.pk): no_peers_price,
            str(debt_basis.pk): debt_basis_price,
            str(source_basis.pk): source_basis_price,
        },
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )

    assert "does not match 'twelve_data'" in (
        forecasts[str(wrong_provider.pk)]["3y"].scenario.insufficiency_reason
    )
    assert "2/3 positive contiguous annual" in (
        forecasts[str(short_history.pk)]["3y"].scenario.insufficiency_reason
    )
    assert "Beginning invested capital unavailable" in (
        forecasts[str(no_capital.pk)]["3y"].scenario.insufficiency_reason
    )
    assert "No same-family SEC SIC peer set" in (
        forecasts[str(no_peers.pk)]["3y"].scenario.insufficiency_reason
    )
    assert forecasts[str(no_peers.pk)]["3y"].calculation["metric_family"] == ("fcf_per_share")
    assert forecasts[str(no_peers.pk)]["3y"].calculation["input_facts"]
    assert "incompatible source definitions" in (
        forecasts[str(debt_basis.pk)]["3y"].scenario.insufficiency_reason
    )
    source_reason = forecasts[str(source_basis.pk)]["3y"].scenario.insufficiency_reason
    assert "incompatible source definitions" in source_reason
    maximum_reason_length = Prediction._meta.get_field("insufficiency_reason").max_length
    assert maximum_reason_length is not None
    assert len(source_reason) <= maximum_reason_length


@pytest.mark.django_db
def test_metric_age_boundary_retains_bounded_split_exposure() -> None:
    config = _small_peer_config()
    target = _listing("AGET")
    peer = _listing("AGEP")
    target_price = _company_evidence(target, family="fcf_per_share", sic="3571")
    peer_price = _company_evidence(peer, family="fcf_per_share", sic="3571")
    target_date = date(2026, 6, 16)
    decision_time = datetime(2026, 6, 17, 12, tzinfo=UTC)

    forecast = build_long_forecasts(
        listings=[target, peer],
        current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
        price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
        asof=AsOfData(decision_time),
        data_cutoff=decision_time,
        target_date=target_date,
        config=config,
    )[str(target.pk)]["3y"]

    assert forecast.scenario.base is not None
    assert forecast.calculation["split_basis"]["post_period_exposure_days"] == 167
    assert forecast.calculation["split_basis"]["maximum_exposure_days"] == 200


@pytest.mark.django_db
def test_later_fact_and_classification_evidence_cannot_change_earlier_forecast() -> None:
    config = _small_peer_config()
    target = _listing("ASOFT")
    peer = _listing("ASOFP")
    target_price = _company_evidence(target, family="fcf_per_share", sic="3571")
    peer_price = _company_evidence(peer, family="fcf_per_share", sic="3571")
    inputs = {
        "listings": [target, peer],
        "current_prices": {str(target.pk): 50.0, str(peer.pk): 50.0},
        "price_assets": {str(target.pk): target_price, str(peer.pk): peer_price},
        "data_cutoff": DECISION_TIME,
        "target_date": TARGET_DATE,
        "config": config,
    }

    before = build_long_forecasts(
        asof=AsOfData(DECISION_TIME),
        **inputs,
    )[str(target.pk)]["3y"]

    later_time = DECISION_TIME + timedelta(days=1)
    later_companyfacts, later_filing = _sec_assets(target, retrieved_at=later_time)
    _fact(
        target,
        later_companyfacts,
        later_filing,
        concept="operating_cash_flow",
        value=Decimal("999"),
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        fiscal_period="FY",
        available_at=later_time,
        accession="late-amendment",
        source_revision=2,
    )
    classification_asset = _asset(
        provider="sec",
        kind="sec_submissions",
        subject=f"{target.ticker}-later-classification",
        retrieved_at=later_time,
    )
    CompanyClassificationObservation.objects.create(
        company=target.security.company,
        provider="sec",
        scheme="sec_sic",
        code="6021",
        observed_at=later_time,
        available_at=later_time,
        source_asset=classification_asset,
    )

    after = build_long_forecasts(
        asof=AsOfData(DECISION_TIME),
        **inputs,
    )[str(target.pk)]["3y"]

    assert after.scenario == before.scenario
    assert after.calculation == before.calculation
    assert tuple(asset.pk for asset in after.source_assets) == tuple(
        asset.pk for asset in before.source_assets
    )


@pytest.mark.django_db
def test_filing_evidence_loading_is_chunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_peer_config()
    target = _listing("CHUNKT")
    peer = _listing("CHUNKP")
    target_price = _company_evidence(target, family="fcf_per_share", sic="3571")
    peer_price = _company_evidence(peer, family="fcf_per_share", sic="3571")
    monkeypatch.setattr(long_forecasts_module, "MAX_SQL_IN_ITEMS", 2)

    with CaptureQueriesContext(connection) as queries:
        forecast = build_long_forecasts(
            listings=[target, peer],
            current_prices={str(target.pk): 50.0, str(peer.pk): 55.0},
            price_assets={str(target.pk): target_price, str(peer.pk): peer_price},
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=config,
        )[str(target.pk)]["3y"]

    evidence_queries = [
        query["sql"]
        for query in queries
        if 'SELECT "data_fundamentalfactevidence"."id"' in query["sql"]
    ]
    assert forecast.scenario.base is not None
    assert len(evidence_queries) > 1


@pytest.mark.django_db
def test_peer_provenance_stays_reference_based_and_bounded() -> None:
    config = _small_peer_config()
    listings = [_listing("PAYLOADT")]
    for index in range(8):
        listings.append(_listing(f"PAY{index}"))
    prices: dict[str, float] = {}
    price_assets: dict[str, DataAsset] = {}
    for index, listing in enumerate(listings):
        price_assets[str(listing.pk)] = _company_evidence(
            listing,
            family="fcf_per_share",
            sic="3571",
            scale=1.0 + index / 100,
        )
        prices[str(listing.pk)] = 50.0 + index

    forecast = build_long_forecasts(
        listings=listings,
        current_prices=prices,
        price_assets=price_assets,
        asof=AsOfData(DECISION_TIME),
        data_cutoff=DECISION_TIME,
        target_date=TARGET_DATE,
        config=config,
    )[str(listings[0].pk)]["3y"]

    assert forecast.calculation["support"]["peer_count"] == 8
    assert all("input_facts" not in peer for peer in forecast.calculation["peer_set"])
    assert all(
        "value" not in reference
        for peer in forecast.calculation["peer_set"]
        for reference in peer["fact_references"]
    )
    payload_size = len(
        json.dumps(
            forecast.calculation,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert payload_size < 180_000


@pytest.mark.django_db
def test_snapshot_analysis_issues_isolated_three_and_five_year_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_peer_config()
    monkeypatch.setattr(
        "stanstock.research.service.load_long_forecast_config",
        lambda _path=None: config,
    )
    monkeypatch.setattr(
        "stanstock.research.service.long_forecast_config_hash",
        lambda _config: "l" * 64,
    )
    ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    universe = Universe.objects.create(
        slug="long-integration",
        name="Long integration",
        config_version="test-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="u" * 64,
    )
    listings = [_listing("LONGA"), _listing("LONGB")]
    store = AssetStore(tmp_path)
    for index, listing in enumerate(listings):
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        _write_price_asset(
            store,
            listing,
            close=50.0 + index * 5,
        )
        _company_evidence(
            listing,
            family="fcf_per_share",
            sic="3571",
            scale=1.0 + index * 0.1,
            create_price=False,
        )

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=True,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
    )

    assert len(results) == 2
    assert Prediction.objects.count() == 6
    decision_predictions = Prediction.objects.filter(evidence_role="decision")
    long_predictions = Prediction.objects.filter(evidence_role="advisory")
    assert set(decision_predictions.values_list("horizon", flat=True)) == {"short"}
    assert set(long_predictions.values_list("horizon", flat=True)) == {"3y", "5y"}
    assert all(
        prediction.method_version == "us-sec-long-v1"
        and prediction.probability_positive is None
        and prediction.price_provider == "twelve_data"
        and prediction.calculation["metric_family"] == "fcf_per_share"
        for prediction in long_predictions
    )
    assert all(
        {asset["provider"] for asset in prediction.source_assets} == {"sec", "twelve_data"}
        for prediction in long_predictions
    )
    for result in results:
        assert result.analysis.has_explicit_long_forecasts
        assert set(result.analysis.forecast_scenarios["horizons"]) == {
            "short",
            "medium",
            "long",
            "3y",
            "5y",
        }
        assert result.analysis.three_year_forecast_scenario["evidence_grade"] == "observed"
        assert result.analysis.recommendation == result.computation.recommendation
    assert _long_forecast_evidence_grade(results[0].run) == "observed"
    results[0].run.issued_on_time = False
    assert _long_forecast_evidence_grade(results[0].run) == "research"

    analysis = results[0].analysis
    original_opportunity = assess_opportunity(analysis, price_band=None)
    analysis.forecast_scenarios["horizons"]["3y"]["base"] = 99.0
    analysis.save(update_fields=["forecast_scenarios"])
    analysis.refresh_from_db()
    assert analysis.recommendation == results[0].computation.recommendation
    assert assess_opportunity(analysis, price_band=None) == original_opportunity

    late_results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME + timedelta(days=1),
        target_date=TARGET_DATE,
        issued_on_time=False,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    late_run = late_results[0].run
    late_predictions = Prediction.objects.filter(
        analysis__run=late_run,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
    )
    assert late_predictions.count() == 4
    assert all(
        prediction.evidence_grade == UniverseSnapshot.Grade.RESEARCH
        and prediction.calculation["evidence_grade"] == UniverseSnapshot.Grade.RESEARCH
        for prediction in late_predictions
    )
    assert all(
        result.analysis.three_year_forecast_scenario["evidence_grade"]
        == UniverseSnapshot.Grade.RESEARCH
        for result in late_results
    )


def _small_peer_config() -> LongForecastConfig:
    config = load_long_forecast_config()
    return replace(
        config,
        peer=replace(
            config.peer,
            minimum_peers={4: 1, 3: 1, 2: 1},
        ),
    )


def _listing(ticker: str) -> Listing:
    company = Company.objects.create(name=f"{ticker} Company", country="US")
    security = Security.objects.create(company=company, name=f"{ticker} Common")
    return Listing.objects.create(
        security=security,
        ticker=ticker,
        provider_symbol=ticker,
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )


def _company_evidence(
    listing: Listing,
    *,
    family: str,
    sic: str | None,
    scale: float = 1.0,
    annual_shares: tuple[float, float, float] = (20.0, 20.0, 20.0),
    negative_fcf: bool = False,
    partial_fcf: bool = False,
    include_eps: bool = False,
    include_tax: bool = True,
    net_income_multiplier: float = 1.0,
    reported_eps_multiplier: float = 1.0,
    skip_latest_reported_eps: bool = False,
    quarter_share_multiplier: float = 1.0,
    annual_period_count: int = 3,
    include_balance_sheet: bool = True,
    mismatched_debt_basis: bool = False,
    mismatched_source_basis: bool = False,
    create_price: bool = True,
) -> DataAsset:
    companyfacts, filing = _sec_assets(listing, retrieved_at=DECISION_TIME - timedelta(days=1))
    if sic is not None:
        classification_asset = _asset(
            provider="sec",
            kind="sec_submissions",
            subject=f"{listing.ticker}-classification",
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
    for index, ((start, end), shares) in enumerate(
        zip(annual_periods, annual_shares, strict=True),
        start=1,
    ):
        if index <= len(annual_periods) - annual_period_count:
            continue
        annual_available = datetime(end.year + 1, 2, 15, tzinfo=UTC)
        _fact(
            listing,
            companyfacts,
            filing,
            concept="weighted_average_diluted_shares",
            value=Decimal(str(shares * scale)),
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=annual_available,
            accession=f"{listing.ticker}-shares-{index}",
        )
        if family == "fcf_per_share" or partial_fcf:
            free_cash_flow = Decimal(str((80 + index * 15) * scale))
            capex = Decimal(str((20 + index * 2) * scale))
            if negative_fcf:
                free_cash_flow = -abs(free_cash_flow)
            _fact(
                listing,
                companyfacts,
                filing,
                concept="operating_cash_flow",
                value=free_cash_flow + capex,
                start=start,
                end=end,
                fiscal_period="FY",
                available_at=annual_available,
                accession=f"{listing.ticker}-ocf-{index}",
            )
            if not partial_fcf:
                _fact(
                    listing,
                    companyfacts,
                    filing,
                    concept="capital_expenditure",
                    value=capex,
                    start=start,
                    end=end,
                    fiscal_period="FY",
                    available_at=annual_available,
                    accession=f"{listing.ticker}-capex-{index}",
                )
        net_income = Decimal(str((70 + index * 15) * scale * net_income_multiplier))
        _fact(
            listing,
            companyfacts,
            filing,
            concept="net_income",
            value=net_income,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=annual_available,
            accession=f"{listing.ticker}-income-{index}",
        )
        if not (skip_latest_reported_eps and index == len(annual_periods)):
            _fact(
                listing,
                companyfacts,
                filing,
                concept="diluted_eps",
                value=(
                    net_income
                    / Decimal(str(shares * scale))
                    * Decimal(str(reported_eps_multiplier))
                ),
                start=start,
                end=end,
                fiscal_period="FY",
                available_at=annual_available,
                accession=f"{listing.ticker}-eps-{index}",
            )

    quarters = (
        (date(2025, 1, 1), date(2025, 3, 31), "Q1"),
        (date(2025, 4, 1), date(2025, 6, 30), "Q2"),
        (date(2025, 7, 1), date(2025, 9, 30), "Q3"),
        (date(2025, 10, 1), date(2025, 12, 31), "Q4"),
    )
    for index, (start, end, fiscal_period) in enumerate(quarters, start=1):
        available_at = datetime(2026, 2, 10 + index, tzinfo=UTC)
        quarter_values = {
            "weighted_average_diluted_shares": Decimal(
                str(annual_shares[-1] * scale * quarter_share_multiplier)
            ),
            "operating_income": Decimal(str(25 * scale)),
            "pretax_income": Decimal(str(22.5 * scale)),
        }
        if include_tax:
            quarter_values["income_tax_expense"] = Decimal(str(4.5 * scale))
        if family == "fcf_per_share":
            if negative_fcf:
                quarter_values["operating_cash_flow"] = Decimal(str(4 * scale))
                quarter_values["capital_expenditure"] = Decimal(str(8 * scale))
            else:
                quarter_values["operating_cash_flow"] = Decimal(str((28 + index) * scale))
                quarter_values["capital_expenditure"] = Decimal(str((5 + index / 2) * scale))
        if family == "eps_per_share" or include_eps:
            quarter_values["net_income"] = Decimal(str(20 * scale))
        for concept, value in quarter_values.items():
            _fact(
                listing,
                companyfacts,
                filing,
                concept=concept,
                value=value,
                start=start,
                end=end,
                fiscal_period=fiscal_period,
                available_at=available_at,
                accession=f"{listing.ticker}-{concept}-{fiscal_period}",
            )

    if include_balance_sheet:
        for year, debt, equity, cash in (
            (2024, 100.0, 400.0, 50.0),
            (2025, 110.0, 450.0, 60.0),
        ):
            debt_concept = (
                "long_term_debt"
                if mismatched_debt_basis and year == 2025
                else "reported_long_term_debt"
            )
            for concept, value in (
                (debt_concept, debt),
                ("equity", equity),
                ("cash_and_equivalents", cash),
            ):
                source_concept = None
                if mismatched_source_basis and year == 2025 and concept == "equity":
                    source_concept = (
                        "us-gaap:"
                        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
                    )
                _fact(
                    listing,
                    companyfacts,
                    filing,
                    concept=concept,
                    value=Decimal(str(value * scale)),
                    start=None,
                    end=date(year, 12, 31),
                    fiscal_period="FY",
                    available_at=datetime(year + 1, 2, 15, tzinfo=UTC),
                    accession=f"{listing.ticker}-{concept}-{year}",
                    source_concept=source_concept,
                )

    if not create_price:
        return companyfacts
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
    retrieved_at: datetime,
) -> tuple[DataAsset, DataAsset]:
    companyfacts = _asset(
        provider="sec",
        kind="sec_companyfacts",
        subject=f"{listing.ticker}-{retrieved_at.isoformat()}",
        retrieved_at=retrieved_at,
    )
    filing = _asset(
        provider="sec",
        kind="sec_submissions",
        subject=f"{listing.ticker}-filing-{retrieved_at.isoformat()}",
        retrieved_at=retrieved_at,
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
) -> FundamentalFact:
    fact = FundamentalFact.objects.create(
        company=listing.security.company,
        provider="sec",
        concept=concept,
        taxonomy="us-gaap",
        source_concept=source_concept or SOURCE_CONCEPTS[concept],
        value=value,
        unit=UNITS.get(concept, "USD"),
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
        filing_date=available_at.date(),
        filed_at=available_at,
        acceptance_at=available_at,
        available_at=available_at,
        availability_basis="acceptance_datetime",
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
        relative_path=f"tests/long/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        metadata=metadata or {},
    )


def _write_price_asset(
    store: AssetStore,
    listing: Listing,
    *,
    close: float,
) -> DataAsset:
    sessions: list[date] = []
    cursor = TARGET_DATE
    while len(sessions) < 320:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor -= timedelta(days=1)
    sessions.reverse()
    frame = pl.DataFrame(
        {
            "date": sessions,
            "close": [close + index * 0.02 for index in range(len(sessions))],
            "volume": [2_000_000 + index for index in range(len(sessions))],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    stored = store.write_frame(f"long-tests/{listing.ticker}.parquet", frame)
    return register_asset(
        provider="twelve_data",
        kind="price_history",
        subject=listing.ticker,
        stored=stored,
        retrieved_at=DECISION_TIME,
        available_at=DECISION_TIME,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )
