"""Base-versus-worktree proof that `us-sec-long-v1`/`v2` are still frozen.

The `us-sec-long-v3` slice edits shared code (`sec_fundamentals`,
`long_forecast_config`, `long_forecasts`). A frozen methodology contract
covers the config bytes, the effective hash, eligibility, reason wording, and
*both* the successful and the withheld calculation and scenario payloads, so
asserting a few hand-picked fields is not enough.

This module builds a fully deterministic fixture -- every model identity is a
UUID5 derived from a stable fixture key, and every asset path and checksum is
a fixed string -- runs v1 and v2 through it, and compares the complete
payloads against:

1. ``tests/data/long_frozen_base_payloads.json``, generated from the exact
   base revision by this file's own harness (always compared); and
2. the base revision's source executed live in an isolated import namespace
   (compared whenever the base object is present in the local git object
   database).

Only `CompanyClassificationObservation.ingested_at` is normalized. It is an
``auto_now_add`` local ingest clock, generated at row-creation time rather
than derived from evidence, so no committed golden could pin it; nothing else
is masked.

The committed golden is *base-produced evidence*, never a recording of the
current behavior. An ordinary run only ever reads it. Regeneration goes
through `regenerate_golden`, which executes the base revision's own modules,
refuses outright when those objects are unavailable, and records the SHA-256
of the exact base sources it ran::

    STANSTOCK_WRITE_FROZEN_GOLDEN=1 pytest tests/test_long_frozen_differential.py
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import pytest

from frozen_base import (
    BASE_MODULE_PATHS,
    BASE_SHA,
    BaseRevisionUnavailableError,
    base_long_forecast_modules,
    base_source_checksums,
    base_sources_available,
)
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
from stanstock.research.long_forecast_config import load_long_forecast_config
from stanstock.research.long_forecasts import build_long_forecasts

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "long_frozen_base_payloads.json"

#: Opt-in regeneration switch. It only ever drives `regenerate_golden`, which
#: executes the base revision's own modules; an ordinary comparison run never
#: writes anything.
WRITE_GOLDEN_ENV = "STANSTOCK_WRITE_FROZEN_GOLDEN"

TARGET_DATE = date(2026, 2, 27)
DECISION_TIME = datetime(2026, 3, 1, 12, tzinfo=UTC)

V1_PATH = Path("config/forecasts/us-sec-long-v1.yml")
V2_PATH = Path("config/forecasts/us-sec-long-v2.yml")

#: Fixed namespace so every fixture row has the same primary key on every
#: machine and in every run. Generated identifiers are the only thing that
#: would otherwise make a committed payload golden environment-specific.
FIXTURE_NAMESPACE = UUID("2f6f2a30-0000-4000-8000-5354414e5354")

INGESTED_AT_PLACEHOLDER = "<auto_now_add>"

SOURCE_CONCEPTS = {
    "net_income": "us-gaap:NetIncomeLoss",
    "diluted_eps": "us-gaap:EarningsPerShareDiluted",
    "weighted_average_diluted_shares": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
    "operating_cash_flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditure": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "operating_income": "us-gaap:OperatingIncomeLoss",
    "pretax_income": (
        "us-gaap:"
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"
        "NoncontrollingInterest"
    ),
    "income_tax_expense": "us-gaap:IncomeTaxExpenseBenefit",
    "cash_and_equivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "long_term_debt": "us-gaap:LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    "reported_long_term_debt": "us-gaap:LongTermDebt",
    "equity": "us-gaap:StockholdersEquity",
}

UNITS = {
    "diluted_eps": "USD/shares",
    "weighted_average_diluted_shares": "shares",
}

QUARTERS = (
    (date(2025, 1, 1), date(2025, 3, 31)),
    (date(2025, 4, 1), date(2025, 6, 30)),
    (date(2025, 7, 1), date(2025, 9, 30)),
    (date(2025, 10, 1), date(2025, 12, 31)),
)

ANNUAL_PERIODS = (
    (date(2023, 1, 1), date(2023, 12, 31)),
    (date(2024, 1, 1), date(2024, 12, 31)),
    (date(2025, 1, 1), date(2025, 12, 31)),
)

#: Each scenario deliberately exercises a different frozen outcome shape.
SCENARIOS = ("eligible", "share_continuity_break", "incompatible_invested_capital")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_frozen_v1_v2_payloads_match_the_committed_base_golden() -> None:
    head = _payloads_with_head()
    golden = _golden_payloads()

    assert set(golden) == set(SCENARIOS)
    for scenario in SCENARIOS:
        assert head[scenario] == golden[scenario], scenario


def test_the_committed_golden_pins_the_exact_base_source_bytes() -> None:
    """A golden labelled with a SHA must name the bytes it came from."""
    metadata = _golden_metadata()

    assert metadata["base_sha"] == BASE_SHA
    assert metadata["generated_from"] == "base_revision_execution"
    assert set(metadata["base_source_sha256"]) == {path for _name, path in BASE_MODULE_PATHS}
    for digest in metadata["base_source_sha256"].values():
        assert len(digest) == 64


@pytest.mark.skipif(
    not base_sources_available(),
    reason=f"base revision {BASE_SHA} is not present in the local git object database",
)
def test_the_committed_golden_checksums_match_the_real_base_objects() -> None:
    assert _golden_metadata()["base_source_sha256"] == base_source_checksums()


@pytest.mark.django_db
def test_regeneration_refuses_when_the_base_objects_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No base objects, no golden. There is no working-tree fallback."""
    import frozen_base

    def _unavailable(path: str) -> bytes:
        raise BaseRevisionUnavailableError(f"pruned: {path}")

    monkeypatch.setattr(frozen_base, "_read_base_source", _unavailable)
    target = tmp_path / "long_frozen_base_payloads.json"

    with pytest.raises(BaseRevisionUnavailableError):
        regenerate_golden(target)

    assert not target.exists()
    # The committed golden is also left exactly as it was.
    assert _golden_metadata()["base_sha"] == BASE_SHA


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_sources_available(),
    reason=f"base revision {BASE_SHA} is not present in the local git object database",
)
def test_changed_head_behavior_can_never_become_the_accepted_base_golden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regeneration runs base code, so head drift cannot be labelled BASE_SHA."""
    module = importlib.import_module(__name__)

    def _poisoned_head(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("regeneration must never execute working-tree behavior")

    monkeypatch.setattr(module, "build_long_forecasts", _poisoned_head)
    target = tmp_path / "long_frozen_base_payloads.json"

    regenerate_golden(target)

    # Byte-identical to the committed golden: the poisoned head contributed
    # nothing, and the committed evidence really is base-produced.
    assert target.read_bytes() == GOLDEN_PATH.read_bytes()


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_sources_available(),
    reason=f"base revision {BASE_SHA} is not present in the local git object database",
)
def test_exact_base_generation_still_compares_complete_payloads_losslessly() -> None:
    """Pooling shrinks the file; it must not drop or reshape any evidence."""
    inputs = _build_all_scenarios()
    with base_long_forecast_modules() as base:
        payloads = _payloads_for_all_scenarios(
            inputs,
            build=base.long_forecasts.build_long_forecasts,
            load_config=base.long_forecast_config.load_long_forecast_config,
        )

    pooled, pool = _pool(payloads)

    assert _expand(pooled, pool) == payloads
    assert payloads == _golden_payloads()


@pytest.mark.django_db
@pytest.mark.skipif(
    os.environ.get(WRITE_GOLDEN_ENV) != "1",
    reason=f"set {WRITE_GOLDEN_ENV}=1 to regenerate the committed base golden",
)
def test_regenerate_the_committed_golden_from_the_exact_base_revision() -> (
    None
):  # pragma: no cover - tooling
    regenerate_golden(GOLDEN_PATH)


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_sources_available(),
    reason=f"base revision {BASE_SHA} is not present in the local git object database",
)
def test_frozen_v1_v2_payloads_match_the_base_revision_executed_live() -> None:
    inputs = _build_all_scenarios()
    head = _payloads_for_all_scenarios(
        inputs,
        build=build_long_forecasts,
        load_config=load_long_forecast_config,
    )
    with base_long_forecast_modules() as base:
        # Prove we really loaded the base bytes and not the working tree.
        assert base.base_sha == BASE_SHA
        assert base.source_sha256 == base_source_checksums()
        assert base.source_sha256 == _golden_metadata()["base_source_sha256"]
        assert not hasattr(base.sec_fundamentals, "TTM_SELECTION_NEWEST_QUARTER_ALIAS")
        assert not hasattr(base.sec_fundamentals, "build_alias_instant_candidates")
        assert "ttm_selection" not in (
            inspect.signature(base.sec_fundamentals.build_sec_fundamental_series).parameters
        )
        assert not hasattr(base.long_forecasts, "audit_invested_capital_pairs")
        assert not hasattr(base.long_forecast_config, "long_forecast_config_path")
        base_payloads = _payloads_for_all_scenarios(
            inputs,
            build=base.long_forecasts.build_long_forecasts,
            load_config=base.long_forecast_config.load_long_forecast_config,
        )

    # The working-tree modules are restored, untouched, afterwards.
    assert hasattr(
        importlib.import_module("stanstock.data.sec_fundamentals"),
        "TTM_SELECTION_NEWEST_QUARTER_ALIAS",
    )

    for scenario in SCENARIOS:
        assert head[scenario] == base_payloads[scenario], scenario

    golden = _golden_payloads()
    for scenario in SCENARIOS:
        assert base_payloads[scenario] == golden[scenario], scenario


@pytest.mark.django_db
def test_the_differential_fixture_is_actually_deterministic() -> None:
    """A golden is only evidence if the fixture identities are stable."""
    payloads = _payloads_with_head()
    ids = [entry["listing_id"] for entry in payloads["eligible"]["forecasts"]]

    assert ids == [
        str(_det("listing", "eligible", "TGT")),
        str(_det("listing", "eligible", "PEER")),
    ]
    input_facts = _target_horizon(payloads, "eligible", "v1", "3y")["calculation"]["input_facts"]
    assert input_facts
    for fact in input_facts:
        assert UUID(fact["id"]) == _det(
            "fact",
            "eligible",
            "TGT",
            fact["accession"],
        )


@pytest.mark.django_db
def test_the_three_frozen_scenarios_cover_success_and_withholding() -> None:
    payloads = _payloads_with_head()

    eligible = _target_horizon(payloads, "eligible", "v2", "3y")
    assert eligible["scenario"]["base"] is not None
    assert eligible["calculation"]["scenario_paths"]

    v1_break = _target_horizon(payloads, "share_continuity_break", "v1", "3y")
    v2_break = _target_horizon(payloads, "share_continuity_break", "v2", "3y")
    assert v1_break["scenario"]["base"] is not None
    assert v2_break["scenario"]["base"] is None
    assert (
        "Adjacent annual diluted-share basis continuity is incompatible/unverified"
        in (v2_break["scenario"]["insufficiency_reason"])
    )
    assert v2_break["calculation"]["split_basis"]["assessment_status"] == (
        "incompatible_or_unverified"
    )
    assert "verified_through" not in v2_break["calculation"]["split_basis"]
    # long-v1 never evaluated that check, so its payload keeps the frozen
    # verified wording on identical evidence.
    assert v1_break["calculation"]["split_basis"]["verified_through"] == "2025-12-31"

    incompatible = _target_horizon(payloads, "incompatible_invested_capital", "v2", "3y")
    assert incompatible["scenario"]["base"] is None
    assert incompatible["scenario"]["insufficiency_reason"] == (
        "Beginning/end invested-capital evidence uses incompatible source definitions"
    )
    # Frozen versions carry no v3 selection provenance at all.
    for scenario in SCENARIOS:
        for version in ("v1", "v2"):
            for horizon in ("3y", "5y"):
                calculation = _target_horizon(payloads, scenario, version, horizon)["calculation"]
                assert "evidence_selection" not in calculation
                assert "invested_capital_selection" not in calculation["formula_inputs"]


# ---------------------------------------------------------------------------
# Payload capture
# ---------------------------------------------------------------------------


def _payloads_with_head() -> dict[str, dict[str, Any]]:
    return _payloads_for_all_scenarios(
        _build_all_scenarios(),
        build=build_long_forecasts,
        load_config=load_long_forecast_config,
    )


ScenarioInputs = tuple[list[Listing], dict[str, float], dict[str, DataAsset]]


def _build_all_scenarios() -> dict[str, ScenarioInputs]:
    """Create every fixture row exactly once per test database."""
    return {scenario: _build_scenario(scenario) for scenario in SCENARIOS}


def _payloads_for_all_scenarios(
    inputs: dict[str, ScenarioInputs],
    *,
    build: Any,
    load_config: Any,
) -> dict[str, dict[str, Any]]:
    return {
        scenario: _payloads_for_scenario(
            scenario,
            inputs[scenario],
            build=build,
            load_config=load_config,
        )
        for scenario in SCENARIOS
    }


def _payloads_for_scenario(
    scenario: str,
    inputs: ScenarioInputs,
    *,
    build: Any,
    load_config: Any,
) -> dict[str, Any]:
    listings, prices, price_assets = inputs
    versions: dict[str, Any] = {}
    for label, path in (("v1", V1_PATH), ("v2", V2_PATH)):
        config = load_config(path)
        config = replace(config, peer=replace(config.peer, minimum_peers={4: 1, 3: 1, 2: 1}))
        versions[label] = build(
            listings=listings,
            current_prices=prices,
            price_assets=price_assets,
            asof=AsOfData(DECISION_TIME),
            data_cutoff=DECISION_TIME,
            target_date=TARGET_DATE,
            config=config,
        )
    return {
        "scenario": scenario,
        "forecasts": [
            {
                "listing_id": str(listing.pk),
                "ticker": listing.ticker,
                "versions": {
                    label: {
                        horizon: _forecast_payload(versions[label][str(listing.pk)][horizon])
                        for horizon in ("3y", "5y")
                    }
                    for label in ("v1", "v2")
                },
            }
            for listing in listings
        ],
    }


def _forecast_payload(forecast: Any) -> dict[str, Any]:
    return _normalize(
        {
            "scenario": forecast.scenario.as_dict(),
            "calculation": forecast.calculation,
            "source_asset_ids": sorted(str(asset.pk) for asset in forecast.source_assets),
        }
    )


def _normalize(payload: Any) -> Any:
    """Mask only the ``auto_now_add`` ingest clock.

    It is generated when the fixture row is written and is not derived from
    any evidence, so it cannot be pinned by a committed golden. Every other
    field -- including every model identity -- is deterministic by
    construction.
    """
    if isinstance(payload, dict):
        return {
            key: (INGESTED_AT_PLACEHOLDER if key == "ingested_at" else _normalize(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_normalize(item) for item in payload]
    if isinstance(payload, tuple):
        return [_normalize(item) for item in payload]
    return payload


def _target_horizon(
    payloads: dict[str, dict[str, Any]],
    scenario: str,
    version: str,
    horizon: str,
) -> dict[str, Any]:
    target = payloads[scenario]["forecasts"][0]
    assert target["ticker"] == "TGT"
    return target["versions"][version][horizon]


POOLED_KEYS = ("input_facts", "peer_set")


def _pool(payloads: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Content-address the two repeated evidence lists.

    ``input_facts`` and ``peer_set`` are byte-identical across versions and
    horizons for a given listing, and repeating them would make the committed
    golden an order of magnitude larger without adding evidence. Pooling is
    lossless: `_expand` restores the exact original structure before any
    comparison.
    """
    pool: dict[str, Any] = {}

    def walk(node: Any, key: str | None) -> Any:
        if isinstance(node, dict):
            return {name: walk(value, name) for name, value in node.items()}
        if isinstance(node, list):
            if key in POOLED_KEYS:
                digest = hashlib.sha256(
                    json.dumps(node, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                pool[digest] = node
                return {"$pooled": digest}
            return [walk(item, None) for item in node]
        return node

    return walk(payloads, None), pool


def _expand(node: Any, pool: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if set(node) == {"$pooled"}:
            return _expand(pool[node["$pooled"]], pool)
        return {name: _expand(value, pool) for name, value in node.items()}
    if isinstance(node, list):
        return [_expand(item, pool) for item in node]
    return node


def _golden_metadata(path: Path | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = json.loads((path or GOLDEN_PATH).read_text(encoding="utf-8"))
    return metadata


def _golden_payloads(path: Path | None = None) -> dict[str, Any]:
    golden = _golden_metadata(path)
    assert golden["base_sha"] == BASE_SHA
    assert golden["generated_from"] == "base_revision_execution"
    expanded: dict[str, Any] = _expand(golden["payloads"], golden["evidence_pool"])
    return expanded


def regenerate_golden(path: Path) -> dict[str, dict[str, Any]]:
    """Rewrite ``path`` from the base revision's own executed modules.

    This is the *only* way the committed golden may be produced. It refuses
    when the base objects are unavailable rather than falling back to the
    working tree, and it records the SHA-256 of the exact base sources it
    executed so a later run can prove the file's provenance. Head payloads
    are never written under a `BASE_SHA` label.
    """
    checksums = base_source_checksums()
    inputs = _build_all_scenarios()
    with base_long_forecast_modules() as base:
        assert base.base_sha == BASE_SHA
        assert base.source_sha256 == checksums
        payloads = _payloads_for_all_scenarios(
            inputs,
            build=base.long_forecasts.build_long_forecasts,
            load_config=base.long_forecast_config.load_long_forecast_config,
        )
    _write_golden(payloads, source_sha256=checksums, path=path)
    return payloads


def _write_golden(
    payloads: dict[str, dict[str, Any]],
    *,
    source_sha256: dict[str, str],
    path: Path,
) -> None:
    pooled, pool = _pool(payloads)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "base_sha": BASE_SHA,
                "base_source_sha256": source_sha256,
                "generated_by": "tests/test_long_frozen_differential.py",
                "generated_from": "base_revision_execution",
                "normalized_fields": ["ingested_at"],
                "pooled_keys": list(POOLED_KEYS),
                "evidence_pool": pool,
                "payloads": pooled,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Deterministic fixture
# ---------------------------------------------------------------------------


def _det(*parts: str) -> UUID:
    return uuid5(FIXTURE_NAMESPACE, "/".join(parts))


def _build_scenario(scenario: str) -> tuple[list[Listing], dict[str, float], dict[str, DataAsset]]:
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    price_assets: dict[str, DataAsset] = {}
    for role, ticker, price, scale in (
        ("TGT", "TGT", 50.0, 1.0),
        ("PEER", "PEER", 55.0, 1.1),
    ):
        listing = _listing(scenario, role, ticker)
        price_asset = _company(
            scenario=scenario,
            role=role,
            listing=listing,
            scale=scale,
            break_share_continuity=(scenario == "share_continuity_break" and role == "TGT"),
            incompatible_invested_capital=(
                scenario == "incompatible_invested_capital" and role == "TGT"
            ),
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        price_assets[str(listing.pk)] = price_asset
    return listings, prices, price_assets


def _listing(scenario: str, role: str, ticker: str) -> Listing:
    company = Company.objects.create(
        id=_det("company", scenario, role),
        name=f"{ticker} {scenario} Company",
        country="US",
    )
    security = Security.objects.create(
        id=_det("security", scenario, role),
        company=company,
        name=f"{ticker} Common",
    )
    return Listing.objects.create(
        id=_det("listing", scenario, role),
        security=security,
        ticker=ticker,
        provider_symbol=ticker,
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )


def _asset(
    *,
    scenario: str,
    role: str,
    key: str,
    provider: str,
    kind: str,
    retrieved_at: datetime,
    subject: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DataAsset:
    asset_id = _det("asset", scenario, role, key)
    return DataAsset.objects.create(
        id=asset_id,
        provider=provider,
        kind=kind,
        subject=(subject or f"{scenario}-{role}-{key}")[:120],
        relative_path=f"tests/frozen-differential/{asset_id}",
        sha256=f"{asset_id.hex}{asset_id.hex}",
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        metadata=metadata or {},
    )


def _company(
    *,
    scenario: str,
    role: str,
    listing: Listing,
    scale: float,
    break_share_continuity: bool,
    incompatible_invested_capital: bool,
) -> DataAsset:
    companyfacts = _asset(
        scenario=scenario,
        role=role,
        key="companyfacts",
        provider="sec",
        kind="sec_companyfacts",
        retrieved_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    filing = _asset(
        scenario=scenario,
        role=role,
        key="filing",
        provider="sec",
        kind="sec_submissions",
        retrieved_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    classification_asset = _asset(
        scenario=scenario,
        role=role,
        key="classification",
        provider="sec",
        kind="sec_submissions",
        retrieved_at=datetime(2026, 2, 28, tzinfo=UTC),
    )
    CompanyClassificationObservation.objects.create(
        id=_det("classification", scenario, role),
        company=listing.security.company,
        provider="sec",
        scheme="sec_sic",
        code="3571",
        description="Electronic computers",
        observed_at=datetime(2026, 2, 28, tzinfo=UTC),
        available_at=datetime(2026, 2, 28, tzinfo=UTC),
        source_asset=classification_asset,
    )

    shares = 20.0 * scale
    for index, (start, end) in enumerate(ANNUAL_PERIODS, start=1):
        available = datetime(end.year + 1, 2, 15, tzinfo=UTC)
        # A 2:1 diluted-share basis break in the final adjacent annual pair is
        # exactly what long-v2 refuses and long-v1 never evaluated.
        period_shares = shares * 2.0 if (break_share_continuity and index == 3) else shares
        net_income = Decimal(str((70 + index * 15) * scale))
        if break_share_continuity and index == 3:
            net_income = net_income * 2
        _fact(
            scenario=scenario,
            role=role,
            listing=listing,
            companyfacts=companyfacts,
            filing=filing,
            concept="weighted_average_diluted_shares",
            value=Decimal(str(period_shares)),
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{role}-shares-{index}",
        )
        free_cash_flow = Decimal(str((80 + index * 15) * scale))
        capex = Decimal(str((20 + index * 2) * scale))
        _fact(
            scenario=scenario,
            role=role,
            listing=listing,
            companyfacts=companyfacts,
            filing=filing,
            concept="operating_cash_flow",
            value=free_cash_flow + capex,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{role}-ocf-{index}",
        )
        _fact(
            scenario=scenario,
            role=role,
            listing=listing,
            companyfacts=companyfacts,
            filing=filing,
            concept="capital_expenditure",
            value=capex,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{role}-capex-{index}",
        )
        _fact(
            scenario=scenario,
            role=role,
            listing=listing,
            companyfacts=companyfacts,
            filing=filing,
            concept="net_income",
            value=net_income,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{role}-income-{index}",
        )
        _fact(
            scenario=scenario,
            role=role,
            listing=listing,
            companyfacts=companyfacts,
            filing=filing,
            concept="diluted_eps",
            value=net_income / Decimal(str(period_shares)),
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available,
            accession=f"{role}-eps-{index}",
        )

    quarter_shares = shares * 2.0 if break_share_continuity else shares
    for index, (start, end) in enumerate(QUARTERS, start=1):
        available_at = datetime(2026, 2, 10 + index, tzinfo=UTC)
        for concept, value in (
            ("weighted_average_diluted_shares", Decimal(str(quarter_shares))),
            ("operating_income", Decimal(str(25 * scale))),
            ("pretax_income", Decimal(str(22.5 * scale))),
            ("income_tax_expense", Decimal(str(4.5 * scale))),
            ("operating_cash_flow", Decimal(str((28 + index) * scale))),
            ("capital_expenditure", Decimal(str((5 + index / 2) * scale))),
        ):
            _fact(
                scenario=scenario,
                role=role,
                listing=listing,
                companyfacts=companyfacts,
                filing=filing,
                concept=concept,
                value=value,
                start=start,
                end=end,
                fiscal_period=f"Q{index}",
                available_at=available_at,
                accession=f"{role}-{concept}-Q{index}",
            )

    balance_sheets = (
        (date(2024, 12, 31), "reported_long_term_debt", datetime(2025, 2, 15, tzinfo=UTC)),
        (date(2025, 12, 31), "reported_long_term_debt", datetime(2026, 2, 15, tzinfo=UTC)),
    )
    if incompatible_invested_capital:
        balance_sheets = (
            (date(2024, 12, 31), "long_term_debt", datetime(2025, 2, 15, tzinfo=UTC)),
            (date(2025, 12, 31), "reported_long_term_debt", datetime(2026, 2, 15, tzinfo=UTC)),
        )
    for period_end, debt_concept, balance_available_at in balance_sheets:
        for concept, value in (
            (debt_concept, 100.0 + (period_end.year - 2024) * 10),
            ("equity", 400.0 + (period_end.year - 2024) * 50),
            ("cash_and_equivalents", 50.0 + (period_end.year - 2024) * 10),
        ):
            _fact(
                scenario=scenario,
                role=role,
                listing=listing,
                companyfacts=companyfacts,
                filing=filing,
                concept=concept,
                value=Decimal(str(value * scale)),
                start=None,
                end=period_end,
                fiscal_period="FY",
                available_at=balance_available_at,
                accession=f"{role}-{concept}-{period_end.isoformat()}",
            )

    return _asset(
        scenario=scenario,
        role=role,
        key="price",
        provider="twelve_data",
        kind="price_history",
        retrieved_at=DECISION_TIME,
        subject=listing.ticker,
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )


def _fact(
    *,
    scenario: str,
    role: str,
    listing: Listing,
    companyfacts: DataAsset,
    filing: DataAsset,
    concept: str,
    value: Decimal,
    start: date | None,
    end: date,
    fiscal_period: str,
    available_at: datetime,
    accession: str,
) -> FundamentalFact:
    fact = FundamentalFact.objects.create(
        id=_det("fact", scenario, role, accession),
        company=listing.security.company,
        provider="sec",
        concept=concept,
        taxonomy="us-gaap",
        source_concept=SOURCE_CONCEPTS[concept],
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
        source_revision=1,
        source_asset=companyfacts,
    )
    FundamentalFactEvidence.objects.create(
        id=_det("evidence", scenario, role, accession),
        fact=fact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=filing,
    )
    return fact


def test_sec_fundamentals_config_is_unedited_by_this_slice() -> None:
    """The v3 slice pins, but never rewrites, the reviewed fundamentals config."""
    config = load_sec_fundamentals_config()

    assert config.config_version == "us-sec-fundamentals-v1"
