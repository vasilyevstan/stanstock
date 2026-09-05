from __future__ import annotations

import json

import pytest
from django.core.management import call_command
from django.utils import timezone

from stanstock.data.assets import AssetStore
from stanstock.data.models import (
    Company,
    DataAsset,
    FundamentalFact,
    FxRate,
    LatestMarketData,
    Listing,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.research.fundamentals import calculate_fundamentals, inputs_from_facts
from stanstock.research.models import AnalysisRun

pytestmark = pytest.mark.django_db


def test_seed_demo_creates_expected_scale() -> None:
    call_command("seed_demo")

    assert Listing.objects.count() == 60
    assert Company.objects.count() == 60
    assert UniverseSnapshot.objects.count() == 1
    assert UniverseMembership.objects.count() == 60
    assert LatestMarketData.objects.count() == 60
    # 60 price histories + 60 latest quotes + 1 benchmark price history
    # + (3 fiscal years * 60 listings) fundamentals bundles + 1 amendment
    # bundle + weekly FX CSV bundles (2 pairs * 7 years).
    assert DataAsset.objects.filter(kind="price_history").count() == 61
    assert DataAsset.objects.filter(kind="latest_quote").count() == 60
    assert DataAsset.objects.filter(kind="fundamentals").count() == 60 * 4
    assert DataAsset.objects.filter(kind="fx_rates").count() > 0
    assert FundamentalFact.objects.count() == 60 * 3 * 4 + 60
    assert FxRate.objects.count() > 0


def test_seed_demo_uses_only_obviously_synthetic_identifiers() -> None:
    call_command("seed_demo")

    for ticker in Listing.objects.values_list("ticker", flat=True):
        assert ticker.startswith("ZZUS") or ticker.startswith("ZZEU")
    for name in Company.objects.values_list("name", flat=True):
        assert name.startswith("Synthetic ")
    assert set(DataAsset.objects.values_list("provider", flat=True)) == {"synthetic_demo"}
    assert set(FundamentalFact.objects.values_list("provider", flat=True)) == {"synthetic_demo"}


def test_seed_demo_metadata_unmistakably_says_synthetic() -> None:
    call_command("seed_demo")

    # Every DataAsset row this command writes must be identifiable as
    # synthetic from its own metadata alone, not just via the
    # `provider="synthetic_demo"` convention.
    for metadata in DataAsset.objects.values_list("metadata", flat=True):
        assert metadata.get("synthetic") is True
    for flags in FundamentalFact.objects.values_list("quality_flags", flat=True):
        assert "synthetic" in flags


def test_seed_demo_universe_snapshot_is_research_grade_not_observed() -> None:
    call_command("seed_demo")

    snapshot = UniverseSnapshot.objects.get()
    assert snapshot.grade == UniverseSnapshot.Grade.RESEARCH
    assert snapshot.grade != UniverseSnapshot.Grade.OBSERVED


def test_seed_demo_creates_one_amendment_per_listing_for_latest_fiscal_year() -> None:
    call_command("seed_demo")

    amendments = FundamentalFact.objects.filter(is_amendment=True)
    assert amendments.count() == 60
    for amendment in amendments:
        assert amendment.concept == "NetIncomeLoss"
        assert "restated" in amendment.quality_flags
        assert amendment.fiscal_year == 2025


def test_seeded_source_concepts_feed_canonical_fundamental_calculations() -> None:
    call_command("seed_demo")
    listing = Listing.objects.select_related("security__company").get(ticker="ZZUS001")
    facts = FundamentalFact.objects.filter(company=listing.security.company).order_by(
        "concept",
        "period_end",
        "available_at",
    )

    result = calculate_fundamentals(inputs_from_facts(facts))

    assert "revenue_growth" in result.values
    assert "net_income_growth" in result.values
    assert "net_margin" in result.values
    assert "roe" in result.values


def test_seed_demo_is_deterministic_across_reruns() -> None:
    call_command("seed_demo")
    price_asset = DataAsset.objects.get(
        provider="synthetic_demo", kind="price_history", subject="ZZUS001"
    )
    first_sha = price_asset.sha256
    first_available_at = price_asset.available_at
    fact = (
        FundamentalFact.objects.filter(
            provider="synthetic_demo", concept="Revenue", fiscal_year=2023
        )
        .order_by("accession")
        .first()
    )
    assert fact is not None
    first_fact_value = fact.value

    call_command("seed_demo")

    price_asset_again = DataAsset.objects.get(
        provider="synthetic_demo", kind="price_history", subject="ZZUS001"
    )
    assert price_asset_again.sha256 == first_sha
    assert price_asset_again.available_at == first_available_at
    fact_again = (
        FundamentalFact.objects.filter(
            provider="synthetic_demo", concept="Revenue", fiscal_year=2023
        )
        .order_by("accession")
        .first()
    )
    assert fact_again is not None
    assert fact_again.value == first_fact_value


def test_seed_demo_is_idempotent_on_rerun() -> None:
    call_command("seed_demo")
    counts_first = (
        Listing.objects.count(),
        Company.objects.count(),
        UniverseSnapshot.objects.count(),
        DataAsset.objects.count(),
        FundamentalFact.objects.count(),
        FxRate.objects.count(),
    )

    call_command("seed_demo")
    counts_second = (
        Listing.objects.count(),
        Company.objects.count(),
        UniverseSnapshot.objects.count(),
        DataAsset.objects.count(),
        FundamentalFact.objects.count(),
        FxRate.objects.count(),
    )

    assert counts_first == counts_second


def test_seed_demo_different_seed_keeps_current_quote_tied_to_immutable_asset() -> None:
    call_command("seed_demo")
    quote = LatestMarketData.objects.select_related("source_asset").get(listing__ticker="ZZUS001")
    original_values = (quote.observed_at, quote.close, quote.previous_close, quote.volume)
    payload = json.loads(AssetStore().resolve(quote.source_asset.relative_path).read_text())

    call_command("seed_demo", seed=1)

    quote.refresh_from_db()
    assert (quote.observed_at, quote.close, quote.previous_close, quote.volume) == original_values
    assert float(quote.close) == pytest.approx(float(payload["close"]))
    assert float(quote.previous_close) == pytest.approx(float(payload["previous_close"]))
    assert quote.volume == int(payload["volume"])


def test_seed_demo_rerun_never_deletes_or_recreates_existing_rows() -> None:
    """Reruns must reuse existing primary keys, not delete-and-recreate them.

    Identical counts alone would not catch a delete-then-recreate
    implementation (the old behaviour this test guards against); comparing
    primary keys across two runs proves the underlying rows -- and anything
    a downstream model might reference by ID -- are stable.
    """
    call_command("seed_demo")
    listing = Listing.objects.get(ticker="ZZUS001")
    snapshot = UniverseSnapshot.objects.get()
    price_asset = DataAsset.objects.get(
        provider="synthetic_demo", kind="price_history", subject="ZZUS001"
    )
    fx_rate = FxRate.objects.order_by("observation_date").first()
    assert fx_rate is not None
    fact = FundamentalFact.objects.order_by("accession").first()
    assert fact is not None
    ids_first = (listing.id, snapshot.id, price_asset.id, fx_rate.id, fact.id)

    call_command("seed_demo")

    ids_second = (
        Listing.objects.get(ticker="ZZUS001").id,
        UniverseSnapshot.objects.get().id,
        DataAsset.objects.get(
            provider="synthetic_demo", kind="price_history", subject="ZZUS001"
        ).id,
        FxRate.objects.order_by("observation_date").first().id,  # type: ignore[union-attr]
        FundamentalFact.objects.order_by("accession").first().id,  # type: ignore[union-attr]
    )
    assert ids_first == ids_second


def test_seed_demo_rerun_is_safe_even_when_snapshot_is_referenced() -> None:
    """Rerunning must never need to delete a referenced snapshot: it reuses it."""
    call_command("seed_demo")
    snapshot = UniverseSnapshot.objects.get()
    generated_at = timezone.now()
    analysis_run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=snapshot.as_of_date,
        universe_snapshot=snapshot,
        config_version="test-v1",
        config_hash="0" * 64,
        code_revision="test",
    )

    call_command("seed_demo")

    assert UniverseSnapshot.objects.count() == 1
    reused_snapshot = UniverseSnapshot.objects.get()
    assert reused_snapshot.id == snapshot.id
    analysis_run.refresh_from_db()
    assert analysis_run.universe_snapshot_id == snapshot.id


def test_price_history_covers_full_business_day_range() -> None:
    call_command("seed_demo")

    asset = DataAsset.objects.get(
        provider="synthetic_demo", kind="price_history", subject="ZZUS001"
    )
    assert asset.period_start.year == 2020
    assert asset.period_end.year == 2026
    assert asset.metadata["rows"] > 1600
