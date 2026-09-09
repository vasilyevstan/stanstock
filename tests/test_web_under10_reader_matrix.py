"""Generator -> JSON round-trip -> reader matrix for `us-under10-shadow-v1`.

This is the primary anti-regression gate for the Under-$10 web reader
(`stanstock.web.views`'s `_is_valid_under10_payload`/`_under10_panel`
family). Every case here calls the *real* generator
(`build_under10_assessment`) over synthetic, in-memory evidence, round-trips
the result through the exact JSON codec a persisted payload goes through,
and feeds it to the *real* reader. Every genuinely generated payload must
survive that round trip and render "recorded" (never "unsupported"); every
single-field corruption of a real payload must become "unsupported", with
no additional write and no recomputation.

Everything here is synthetic and constructed in memory: `FundamentalFact`/
`DataAsset` instances are never `.save()`d (`build_under10_assessment` is a
pure function over already-resolved evidence, exactly as
`test_research_under10.py` already relies on), so the bulk matrix touches
no database. A handful of representative cases are additionally round-
tripped through an authenticated HTTP GET against a real persisted
`StockAnalysis`, proving the same contract holds through the full view/
template stack. No provider, credential, or network access anywhere in
this file.
"""

from __future__ import annotations

import itertools
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import polars as pl
import pytest
from django.urls import reverse

from stanstock.data.models import (
    DataAsset,
    FundamentalFact,
)
from stanstock.data.provider_policy import (
    CAPABILITY_NO_REVIEWED_SOURCE,
    CAPABILITY_PLAN_NOT_ENTITLED,
    CAPABILITY_UNAVAILABLE,
    SPLIT_EVENT_CAPABILITY,
)
from stanstock.data.sec_fundamentals import MAX_ANNUAL_DAYS, MIN_ANNUAL_DAYS
from stanstock.research.indicators import (
    DOLLAR_VOLUME_DROPPED_ROWS,
    DOLLAR_VOLUME_DUPLICATE_SESSIONS,
    DOLLAR_VOLUME_INSUFFICIENT_SESSIONS,
    DOLLAR_VOLUME_INVALID_CLOSE,
    DOLLAR_VOLUME_INVALID_SESSION_DATES,
    DOLLAR_VOLUME_INVALID_VOLUME,
    DOLLAR_VOLUME_MISSING_COLUMNS,
    DOLLAR_VOLUME_NONFINITE_MEDIAN,
    DOLLAR_VOLUME_NONFINITE_PRODUCT,
)
from stanstock.research.models import StockAnalysis
from stanstock.research.under10 import (
    LIQUIDITY_BASIS_INCOMPATIBLE,
    LIQUIDITY_COMPUTED,
    LIQUIDITY_FUTURE_PRICE_SESSION,
    LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE,
    LIQUIDITY_STALE_PRICE_EVIDENCE,
    LIQUIDITY_WITHHELD,
    RUNWAY_COMPUTED,
    RUNWAY_NOT_APPLICABLE,
    RUNWAY_WITHHELD,
    SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION,
    SOLVENCY_ELEVATED_OBLIGATION_RISK,
    SOLVENCY_INSUFFICIENT_EVIDENCE,
    SOLVENCY_NO_ADVERSE_EVIDENCE,
    build_under10_assessment,
    under10_assessment_hash,
    under10_policy_hash,
)
from stanstock.web.views import (
    _ExpectedPriceAssetReference,
    _is_valid_under10_payload,
    _under10_panel,
)

# Reused verbatim from the existing in-memory generator fixture helpers --
# not duplicated, per "reuse existing helpers and patterns". `sec_config`,
# `authenticated_client`, `persisted_analysis`, and `scheduler_status` are
# shared pytest fixtures from `tests/conftest.py`; they need no import (and
# must not be imported -- a fixture parameter of the same name would
# otherwise look like an unrelated import redefinition to the linter).
from test_research_under10 import (
    ANNUAL_PERIOD,
    DATA_CUTOFF,
    INSTANT_DATE,
    LISTING_ID,
    TARGET_DATE,
    _asset,
    _fact,
    _price_asset,
)
from test_web_research import _make_under_ten

# ---------------------------------------------------------------------------
# Round-trip / reader assertion helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _preverified_under10_evidence_for_structural_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory matrix isolates cheap validation and presentation.

    Facts and assets in this module are deliberately unsaved. Real immutable
    evidence replay, including authenticated GETs, is covered by the pipeline
    integration module.
    """
    monkeypatch.setattr(
        "stanstock.web.views.under10_assessment_matches_persisted_evidence",
        lambda **_kwargs: True,
    )


def _round_trip(payload: dict[str, Any]) -> dict[str, Any]:
    """Exactly what persisting to, then reading back from, a JSON column does."""
    return json.loads(json.dumps(payload))


def _recompute_hash(payload: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``payload`` with ``assessment_hash`` corrected for its content.

    Every real payload's hash is computed by `build_under10_assessment`
    *after* every other field is already final, so a hand-mutated payload
    that wants to represent "what a genuine assessment carrying this exact
    content would look like" (rather than "storage corrupted an otherwise
    genuine row") must recompute it the same way, or the new accidental-
    corruption checksum gate would reject it for the wrong reason and stop
    exercising the semantic branch validators this matrix exists to prove.
    """
    return {**payload, "assessment_hash": under10_assessment_hash(payload)}


#: The one immutable decision-run close/currency every payload and stub
#: analysis in this module shares, so the F3 binding check never sees a
#: contrived mismatch by accident. Genuinely inside the Under-$10 band.
_STUB_REFERENCE_CLOSE = Decimal("4.250000")
_STUB_CURRENCY = "USD"
_STUB_CODE_REVISION = "0" * 40


class _DerivePriceAssetReference:
    """Sentinel type: derive the expected price-asset reference (both the
    UUID *and* content checksum) from the payload's own ``liquidity.
    price_asset`` -- the correct default for every case that is *not*
    specifically exercising the C1 transplant/checksum-mismatch guards,
    since a genuine payload's own recorded price asset and its parent
    analysis's own ``data_quality["source_assets"]`` entry are
    structurally the same `DataAsset` by construction (see
    `_is_valid_under10_payload`'s C1 note). A dedicated sentinel type
    (rather than a magic string or bare ``None``) keeps "please derive
    this" unambiguous from a genuine ``None`` override, which instead
    means "the parent analysis establishes no anchor at all" (the missing-
    parent-source-entry regression)."""


#: The one shared instance every default parameter below uses.
_DERIVE_PRICE_ASSET_REFERENCE = _DerivePriceAssetReference()

#: A resolved expected reference is either genuinely absent (``None``,
#: meaning the parent analysis establishes no anchor), an explicit
#: ``(id, sha256)`` pair, or the sentinel asking to derive one from the
#: payload under test.
_PriceAssetReferenceOverride = tuple[str, str] | None | _DerivePriceAssetReference


def _price_asset_reference_from_payload(payload: object) -> tuple[str, str] | None:
    """The payload's own ``(liquidity.price_asset.id, ...sha256)``, or ``None``.

    ``None`` both when the fields are genuinely absent (e.g. the
    ``price_provenance_unavailable`` withheld branch, which never reaches
    the C1 binding check at all) and when ``payload`` is malformed --
    either way, harmless as a derived "expected" value for a case that
    was never going to compare it.
    """
    if not isinstance(payload, dict):
        return None
    liquidity = payload.get("liquidity")
    if not isinstance(liquidity, dict):
        return None
    price_asset = liquidity.get("price_asset")
    if not isinstance(price_asset, dict):
        return None
    asset_id = price_asset.get("id")
    sha256 = price_asset.get("sha256")
    if isinstance(asset_id, str) and isinstance(sha256, str):
        return (asset_id, sha256)
    return None


def _resolve_price_asset_reference(
    override: _PriceAssetReferenceOverride, *, payload: object
) -> tuple[str, str] | None:
    if isinstance(override, _DerivePriceAssetReference):
        return _price_asset_reference_from_payload(payload)
    return override


def _stub_analysis(
    *,
    data_quality: dict[str, Any],
    price_asset_reference: _PriceAssetReferenceOverride = _DERIVE_PRICE_ASSET_REFERENCE,
    data_cutoff: datetime = DATA_CUTOFF,
    listing_id: str = LISTING_ID,
    code_revision: str = _STUB_CODE_REVISION,
) -> SimpleNamespace:
    """The minimal `StockAnalysis`-shaped surface `_under10_panel` reads.

    A plain in-memory stand-in (not a persisted model) so the bulk matrix
    exercises the *real* `_under10_panel` render path -- not merely the
    schema validator -- without touching the database per case.

    ``price_asset_reference`` seeds both ``data_quality["price_source"]
    ["asset_id"]`` and the matching entry in ``data_quality[
    "source_assets"]`` (the C1 binding anchor): by default both are
    derived from the payload's own recorded price asset, matching genuine
    same-computation behavior; pass an explicit ``(id, sha256)`` pair to
    exercise a transplant/checksum-mismatch guard, or ``None`` to exercise
    the missing-parent-anchor guard. ``data_cutoff`` is the parent
    `AnalysisRun.data_cutoff` (the other C1 anchor, now bound to *exact*
    equality); pass a different value to exercise that guard.
    ``listing_id`` is the parent `StockAnalysis.listing_id` (the F2
    anchor); pass a different value to exercise the whole-blob-transplant
    guard.
    """
    resolved_reference = _resolve_price_asset_reference(
        price_asset_reference, payload=data_quality.get("under10_assessment")
    )
    full_data_quality = dict(data_quality)
    if resolved_reference is not None:
        asset_id, sha256 = resolved_reference
        full_data_quality.setdefault("price_source", {"asset_id": asset_id})
        full_data_quality.setdefault("source_assets", [{"id": asset_id, "sha256": sha256}])
    return SimpleNamespace(
        data_quality=full_data_quality,
        current_price=_STUB_REFERENCE_CLOSE,
        run=SimpleNamespace(
            target_date=TARGET_DATE,
            data_cutoff=data_cutoff,
            code_revision=code_revision,
        ),
        listing=SimpleNamespace(currency=_STUB_CURRENCY),
        listing_id=listing_id,
    )


def _is_valid_stub_payload(
    payload: object,
    *,
    expected_listing_id: str = LISTING_ID,
    expected_target_date: date = TARGET_DATE,
    expected_data_cutoff: datetime = DATA_CUTOFF,
    expected_reference_close: Decimal = _STUB_REFERENCE_CLOSE,
    expected_currency: str = _STUB_CURRENCY,
    expected_reference_close_in_band: bool = True,
    expected_price_asset_reference: _PriceAssetReferenceOverride = _DERIVE_PRICE_ASSET_REFERENCE,
    expected_code_revision: str = _STUB_CODE_REVISION,
) -> bool:
    """`_is_valid_under10_payload`, defaulted to this module's shared stub
    decision-run context (`_stub_analysis`'s own values), so a call site
    that only varies one expectation (usually ``expected_target_date``)
    never needs to repeat the rest.

    ``expected_price_asset_reference`` defaults to deriving both the id
    and checksum from ``payload`` itself; pass an explicit ``(id,
    sha256)`` pair or ``None`` to exercise the C1 transplant/checksum/
    missing-anchor guards directly against the validator."""
    resolved_reference = _resolve_price_asset_reference(
        expected_price_asset_reference, payload=payload
    )
    expected_price_asset = (
        _ExpectedPriceAssetReference(id=resolved_reference[0], sha256=resolved_reference[1])
        if resolved_reference is not None
        else None
    )
    return _is_valid_under10_payload(
        payload,
        expected_listing_id=expected_listing_id,
        expected_target_date=expected_target_date,
        expected_data_cutoff=expected_data_cutoff,
        expected_reference_close=expected_reference_close,
        expected_currency=expected_currency,
        expected_reference_close_in_band=expected_reference_close_in_band,
        expected_price_asset=expected_price_asset,
        expected_code_revision=expected_code_revision,
    )


def _assert_readable(payload: dict[str, Any]) -> dict[str, Any]:
    """A genuinely generated payload must round-trip and render "recorded"."""
    round_tripped = _round_trip(payload)
    assert _is_valid_stub_payload(round_tripped) is True, (
        f"genuine generator output rejected by the reader: {round_tripped}"
    )
    panel = _under10_panel(
        _stub_analysis(data_quality={"under10_assessment": round_tripped}),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "recorded"
    return panel


def _assert_unsupported(payload: dict[str, Any]) -> None:
    """A corrupted payload must be refused entirely -- never a favorable render.

    Checked in *both* the accidental-corruption checksum's stale-hash form
    (``payload`` as given -- for a hand-mutated payload this is normally a
    stale hash, trivially caught by `_is_valid_assessment_hash` alone) and
    with the hash recomputed to match the mutated content (which must
    still be rejected, this time only by the semantic branch validators; a
    matching hash must never substitute for them). Every corruption test in
    this module goes through here rather than repeating both variants at
    each call site, so a hash mismatch alone can never quietly stand in for
    proving *why* a specific corruption is refused.
    """
    for candidate in (payload, _recompute_hash(payload)):
        round_tripped = _round_trip(candidate)
        assert _is_valid_stub_payload(round_tripped) is False
        panel = _under10_panel(
            _stub_analysis(data_quality={"under10_assessment": round_tripped}),
            current_price_band=None,
        )
        assert panel is not None
        assert panel["state"] == "unsupported"


# ---------------------------------------------------------------------------
# Synthetic, in-memory solvency evidence -- per-concept (value, period_end)
# control, so a period mismatch can be constructed precisely.
# ---------------------------------------------------------------------------


def _solvency_facts(
    *,
    cash: tuple[str, date] | None = ("1000.00000000", INSTANT_DATE),
    short_term_debt: tuple[str, date] | None = ("100.00000000", INSTANT_DATE),
    current_long_term_debt: tuple[str, date] | None = ("50.00000000", INSTANT_DATE),
    current_assets: tuple[str, date] | None = ("2000.00000000", INSTANT_DATE),
    current_liabilities: tuple[str, date] | None = ("1000.00000000", INSTANT_DATE),
    operating_cash_flow: str | None = "400.00000000",
    capital_expenditure: str | None = "100.00000000",
    annual_period: tuple[date, date] = ANNUAL_PERIOD,
    asset: DataAsset | None = None,
) -> list[FundamentalFact]:
    source = asset or _asset()
    rows: list[FundamentalFact] = []
    instants: dict[str, tuple[str, date] | None] = {
        "cash_and_equivalents": cash,
        "short_term_debt": short_term_debt,
        "current_long_term_debt": current_long_term_debt,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
    }
    for concept, spec in instants.items():
        if spec is None:
            continue
        value, period_end = spec
        rows.append(
            _fact(
                concept,
                value,
                asset=source,
                period_end=period_end,
                fact_key=f"{concept}:{period_end.isoformat()}:{value}",
            )
        )
    durations: dict[str, str | None] = {
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
                fact_key=f"{concept}:{annual_period[1].isoformat()}:{value}",
            )
        )
    return rows


def _quarter_periods(*, end: date, lengths: tuple[int, int, int, int]) -> list[tuple[date, date]]:
    """Four contiguous ``(start, end)`` pairs ending at ``end``.

    ``lengths`` gives each quarter's exact inclusive day count
    (``(end - start).days + 1``), oldest quarter first, with no gap between
    a quarter's end and the following quarter's start -- exactly what
    `_quarters_contiguous` in `stanstock.data.sec_fundamentals` requires.
    """
    periods: list[tuple[date, date]] = []
    current_end = end
    for length in reversed(lengths):
        current_start = current_end - timedelta(days=length - 1)
        periods.append((current_start, current_end))
        current_end = current_start - timedelta(days=1)
    periods.reverse()
    return periods


def _ttm_solvency_facts(
    *,
    cash: tuple[str, date] | None = ("1000.00000000", INSTANT_DATE),
    short_term_debt: tuple[str, date] | None = ("100.00000000", INSTANT_DATE),
    current_long_term_debt: tuple[str, date] | None = ("50.00000000", INSTANT_DATE),
    current_assets: tuple[str, date] | None = ("2000.00000000", INSTANT_DATE),
    current_liabilities: tuple[str, date] | None = ("1000.00000000", INSTANT_DATE),
    quarter_end: date = date(2025, 12, 31),
    quarter_lengths: tuple[int, int, int, int] = (91, 91, 92, 91),
    quarter_operating_cash_flow: str = "100.00000000",
    quarter_capital_expenditure: str = "25.00000000",
    asset: DataAsset | None = None,
) -> list[FundamentalFact]:
    """Genuine TTM-basis solvency facts: the same instants `_solvency_facts`
    builds, plus four directly reported (never YTD-derived) contiguous
    quarters -- exactly the ``direct`` path `_quarter_series` selects --
    so `_resolve_free_cash_flow` reports ``duration_basis == "ttm"``.
    """
    source = asset or _asset()
    rows = _solvency_facts(
        cash=cash,
        short_term_debt=short_term_debt,
        current_long_term_debt=current_long_term_debt,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        operating_cash_flow=None,
        capital_expenditure=None,
        asset=source,
    )
    quarter_periods = _quarter_periods(end=quarter_end, lengths=quarter_lengths)
    for index, (start, end) in enumerate(quarter_periods):
        rows.append(
            _fact(
                "operating_cash_flow",
                quarter_operating_cash_flow,
                asset=source,
                period_start=start,
                period_end=end,
                accession=f"0000000000-25-ttmo{index}",
                fact_key=f"ttm-ocf-q{index}:{start.isoformat()}:{end.isoformat()}",
            )
        )
        rows.append(
            _fact(
                "capital_expenditure",
                quarter_capital_expenditure,
                asset=source,
                period_start=start,
                period_end=end,
                accession=f"0000000000-25-ttmc{index}",
                fact_key=f"ttm-capex-q{index}:{start.isoformat()}:{end.isoformat()}",
            )
        )
    return rows


def _price_frame(
    *,
    sessions: int = 252,
    close: float = 4.0,
    volume: float = 1_000_000.0,
    last_session: date = TARGET_DATE,
    duplicate_last: bool = False,
) -> pl.DataFrame:
    dates = [last_session - timedelta(days=index) for index in range(sessions)][::-1]
    if duplicate_last:
        dates[-2] = dates[-1]
    return pl.DataFrame(
        {
            "date": dates,
            "close": [close] * sessions,
            "volume": [volume] * sessions,
        }
    )


def _build_payload(sec_config, **overrides: Any) -> dict[str, Any]:
    facts = overrides.pop("facts", None)
    if facts is None:
        facts = _solvency_facts()
    price_frame = overrides.pop("price_frame", None)
    if price_frame is None:
        price_frame = _price_frame()
    price_asset = overrides.pop("price_asset", "__unset__")
    if price_asset == "__unset__":
        price_asset = _price_asset()
    price_source = overrides.pop("price_source", "__unset__")
    if price_source == "__unset__":
        price_source = {"asset_id": str(price_asset.id)} if price_asset is not None else None
    kwargs: dict[str, Any] = {
        "facts": facts,
        "sec_config": sec_config,
        "price_frame": price_frame,
        "price_asset": price_asset,
        "price_source": price_source,
        "reference_close": _STUB_REFERENCE_CLOSE,
        "target_date": TARGET_DATE,
        "data_cutoff": DATA_CUTOFF,
        "code_revision_value": _STUB_CODE_REVISION,
        "provider": "twelve_data",
        "provider_plan": "basic",
        "evidence_cutoff_safe": True,
        "company_identity_present": True,
        "invalid_session_date_rows": 0,
        "listing_id": LISTING_ID,
    }
    kwargs.update(overrides)
    return build_under10_assessment(**kwargs)


# ---------------------------------------------------------------------------
# R2: fixed v1 policy fields and canonical SEC lineage.
# ---------------------------------------------------------------------------

_R2_FIXED_AND_LINEAGE_CORRUPTIONS = (
    "missing_gates",
    "all_true_gates",
    "wrong_gate_shape",
    "extra_gate",
    "numeric_false_gates",
    "missing_activated",
    "activated_true",
    "activated_zero",
    "missing_shadow_only",
    "shadow_only_false",
    "shadow_only_one",
    "missing_activation_eligible",
    "activation_eligible_true",
    "activation_eligible_zero",
    "missing_blocking_reasons",
    "wrong_blocking_reason",
    "extra_blocking_reason",
    "missing_assessed_fact_ids",
    "missing_assessed_assets",
    "complete_favorable_empty_lineage",
    "malformed_fact_id",
    "unsorted_fact_ids",
    "duplicate_fact_id",
    "noncanonical_fact_id",
    "malformed_asset_ref",
    "extra_asset_ref_field",
    "unsorted_asset_refs",
    "duplicate_asset_ref",
    "noncanonical_asset_id",
    "uppercase_asset_checksum",
    "facts_empty_assets_present",
    "facts_present_assets_empty",
    "missing_price_band",
    "wrong_price_band",
    "missing_date_basis",
    "wrong_date_basis",
    "missing_code_revision",
    "empty_code_revision",
    "list_code_revision",
    "mismatched_code_revision",
    "missing_allocation",
    "nonzero_allocation",
    "false_allocation",
    "true_allocation",
)


def _r2_corrupted_payload(
    sec_config,
    corruption: str,
    *,
    code_revision_value: str = _STUB_CODE_REVISION,
) -> dict[str, Any]:
    """Return one semantic corruption with its assessment checksum recomputed."""
    payload = _round_trip(_build_payload(sec_config, code_revision_value=code_revision_value))
    solvency = payload["solvency"]
    if corruption == "missing_gates":
        del payload["gates"]
    elif corruption == "all_true_gates":
        payload["gates"] = {
            "solvency_obligation": True,
            "dollar_liquidity_252": True,
            "verified_split_evidence": True,
        }
    elif corruption == "wrong_gate_shape":
        payload["gates"] = {
            "solvency_obligation": False,
            "dollar_liquidity_252": False,
        }
    elif corruption == "extra_gate":
        payload["gates"]["unreviewed_extra_gate"] = False
    elif corruption == "numeric_false_gates":
        payload["gates"] = {
            "solvency_obligation": 0,
            "dollar_liquidity_252": 0,
            "verified_split_evidence": 0,
        }
    elif corruption == "missing_activated":
        del payload["activated"]
    elif corruption == "activated_true":
        payload["activated"] = True
    elif corruption == "activated_zero":
        payload["activated"] = 0
    elif corruption == "missing_shadow_only":
        del payload["shadow_only"]
    elif corruption == "shadow_only_false":
        payload["shadow_only"] = False
    elif corruption == "shadow_only_one":
        payload["shadow_only"] = 1
    elif corruption == "missing_activation_eligible":
        del payload["activation_eligible"]
    elif corruption == "activation_eligible_true":
        payload["activation_eligible"] = True
    elif corruption == "activation_eligible_zero":
        payload["activation_eligible"] = 0
    elif corruption == "missing_blocking_reasons":
        del payload["blocking_reasons"]
    elif corruption == "wrong_blocking_reason":
        payload["blocking_reasons"] = ["no_reviewed_corporate_actions_source"]
    elif corruption == "extra_blocking_reason":
        payload["blocking_reasons"].append("unreviewed_extra_reason")
    elif corruption == "missing_assessed_fact_ids":
        del solvency["assessed_fact_ids"]
    elif corruption == "missing_assessed_assets":
        del solvency["assessed_assets"]
    elif corruption == "complete_favorable_empty_lineage":
        assert solvency["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
        solvency["assessed_fact_ids"] = []
        solvency["assessed_assets"] = []
    elif corruption == "malformed_fact_id":
        solvency["assessed_fact_ids"] = ["not-a-uuid"]
    elif corruption == "unsorted_fact_ids":
        fact_ids = solvency["assessed_fact_ids"]
        assert fact_ids == sorted(fact_ids) and len(fact_ids) > 1
        solvency["assessed_fact_ids"] = list(reversed(fact_ids))
    elif corruption == "duplicate_fact_id":
        solvency["assessed_fact_ids"].append(solvency["assessed_fact_ids"][-1])
    elif corruption == "noncanonical_fact_id":
        solvency["assessed_fact_ids"] = ["AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"]
    elif corruption == "malformed_asset_ref":
        solvency["assessed_assets"] = [{"id": "not-a-uuid", "sha256": "a" * 64}]
    elif corruption == "extra_asset_ref_field":
        solvency["assessed_assets"][0]["provider"] = "sec"
    elif corruption == "unsorted_asset_refs":
        second = {
            "id": "00000000-0000-4000-8000-000000000001",
            "sha256": "a" * 64,
        }
        ordered = sorted([*solvency["assessed_assets"], second], key=lambda item: item["id"])
        solvency["assessed_assets"] = list(reversed(ordered))
    elif corruption == "duplicate_asset_ref":
        solvency["assessed_assets"].append(dict(solvency["assessed_assets"][0]))
    elif corruption == "noncanonical_asset_id":
        solvency["assessed_assets"] = [
            {
                "id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
                "sha256": "a" * 64,
            }
        ]
    elif corruption == "uppercase_asset_checksum":
        solvency["assessed_assets"][0]["sha256"] = "A" * 64
    elif corruption == "facts_empty_assets_present":
        solvency["assessed_fact_ids"] = []
    elif corruption == "facts_present_assets_empty":
        solvency["assessed_assets"] = []
    elif corruption == "missing_price_band":
        del payload["evaluated_for"]["price_band"]
    elif corruption == "wrong_price_band":
        payload["evaluated_for"]["price_band"] = "10_to_50"
    elif corruption == "missing_date_basis":
        del payload["evaluated_for"]["date_basis"]
    elif corruption == "wrong_date_basis":
        payload["evaluated_for"]["date_basis"] = "latest_market"
    elif corruption == "missing_code_revision":
        del payload["code_revision"]
    elif corruption == "empty_code_revision":
        payload["code_revision"] = ""
    elif corruption == "list_code_revision":
        payload["code_revision"] = [code_revision_value]
    elif corruption == "mismatched_code_revision":
        payload["code_revision"] = f"{code_revision_value}-different"
    elif corruption == "missing_allocation":
        del payload["new_allocation_percent"]
    elif corruption == "nonzero_allocation":
        payload["new_allocation_percent"] = 1
    elif corruption == "false_allocation":
        payload["new_allocation_percent"] = False
    elif corruption == "true_allocation":
        payload["new_allocation_percent"] = True
    else:
        raise AssertionError(f"Unknown R2 corruption case: {corruption}")
    payload["assessment_hash"] = under10_assessment_hash(payload)
    return payload


@pytest.mark.parametrize("corruption", _R2_FIXED_AND_LINEAGE_CORRUPTIONS)
def test_r2_fixed_policy_and_lineage_corruptions_are_unsupported_directly_and_in_panel(
    sec_config,
    corruption: str,
) -> None:
    """A matching checksum never substitutes for the canonical v1 contract."""
    corrupted = _r2_corrupted_payload(sec_config, corruption)
    assert corrupted["assessment_hash"] == under10_assessment_hash(corrupted)
    _assert_unsupported(corrupted)


@pytest.mark.django_db
@pytest.mark.parametrize("corruption", _R2_FIXED_AND_LINEAGE_CORRUPTIONS)
def test_r2_fixed_policy_and_lineage_corruptions_render_http_200_unsupported(
    sec_config,
    authenticated_client,
    persisted_analysis,
    django_assert_max_num_queries,
    corruption: str,
) -> None:
    """The authenticated detail reader fails closed, read-only, and query-bounded."""
    corrupted = _r2_corrupted_payload(
        sec_config,
        corruption,
        code_revision_value=persisted_analysis.run.code_revision,
    )
    _align_run_target_date(persisted_analysis, TARGET_DATE)
    _make_under_ten(
        persisted_analysis,
        assessment=corrupted,
        align_code_revision=False,
    )
    stored_before = StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality
    stored_payload = stored_before["under10_assessment"]
    assert stored_payload["assessment_hash"] == under10_assessment_hash(stored_payload)

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    assert detail.context["under10_panel"]["state"] == "unsupported"
    assert StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality == stored_before


@pytest.mark.parametrize(
    "branch",
    ("evidence_free_insufficient", "evidence_backed_insufficient", "complete"),
)
def test_r2_genuine_lineage_branches_remain_recorded(sec_config, branch: str) -> None:
    if branch == "evidence_free_insufficient":
        payload = _build_payload(sec_config, facts=[])
        assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
        assert payload["solvency"]["assessed_fact_ids"] == []
        assert payload["solvency"]["assessed_assets"] == []
    elif branch == "evidence_backed_insufficient":
        payload = _build_payload(sec_config, evidence_cutoff_safe=False)
        assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
        assert payload["solvency"]["assessed_fact_ids"]
        assert payload["solvency"]["assessed_assets"]
        assert all(value is None for value in payload["solvency"]["inputs"].values())
    else:
        payload = _build_payload(sec_config)
        assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
        assert payload["solvency"]["assessed_fact_ids"]
        assert payload["solvency"]["assessed_assets"]
    panel = _assert_readable(payload)
    assert panel["state"] == "recorded"


# ---------------------------------------------------------------------------
# Solvency matrix: every genuinely generated combination must be readable.
# ---------------------------------------------------------------------------

_CASH_CHOICES: dict[str, tuple[str, date] | None] = {
    "missing": None,
    "positive": ("1000.00000000", INSTANT_DATE),
    "negative": ("-1.00000000", INSTANT_DATE),
    "zero": ("0.00000000", INSTANT_DATE),
}
_DEBT_CHOICES: dict[str, tuple[tuple[str, date] | None, tuple[str, date] | None]] = {
    "both": (("100.00000000", INSTANT_DATE), ("50.00000000", INSTANT_DATE)),
    "missing_short": (None, ("50.00000000", INSTANT_DATE)),
    "missing_long": (("100.00000000", INSTANT_DATE), None),
    "missing_both": (None, None),
}
_ASSETS_LIABILITIES_CHOICES: dict[str, tuple[tuple[str, date] | None, tuple[str, date] | None]] = {
    "assets_above": (("2000.00000000", INSTANT_DATE), ("1000.00000000", INSTANT_DATE)),
    "assets_below": (("500.00000000", INSTANT_DATE), ("1000.00000000", INSTANT_DATE)),
    "assets_equal": (("1000.00000000", INSTANT_DATE), ("1000.00000000", INSTANT_DATE)),
    "liabilities_zero": (("2000.00000000", INSTANT_DATE), ("0.00000000", INSTANT_DATE)),
    "liabilities_negative": (("2000.00000000", INSTANT_DATE), ("-100.00000000", INSTANT_DATE)),
    "missing_assets": (None, ("1000.00000000", INSTANT_DATE)),
    "missing_liabilities": (("2000.00000000", INSTANT_DATE), None),
}
_FCF_CHOICES: dict[str, tuple[str | None, str | None]] = {
    "positive": ("400.00000000", "100.00000000"),
    "negative": ("100.00000000", "400.00000000"),
    "zero": ("100.00000000", "100.00000000"),
    "missing": (None, None),
}

_SOLVENCY_GRID = list(
    itertools.product(
        _CASH_CHOICES.items(),
        _DEBT_CHOICES.items(),
        _ASSETS_LIABILITIES_CHOICES.items(),
        _FCF_CHOICES.items(),
    )
)


@pytest.mark.parametrize(
    "cash_choice, debt_choice, assets_liabilities_choice, fcf_choice",
    _SOLVENCY_GRID,
    ids=[f"{c[0]}-{d[0]}-{a[0]}-{f[0]}" for c, d, a, f in _SOLVENCY_GRID],
)
def test_every_genuinely_generated_solvency_combination_is_readable(
    sec_config,
    cash_choice: tuple[str, tuple[str, date] | None],
    debt_choice: tuple[str, tuple[tuple[str, date] | None, tuple[str, date] | None]],
    assets_liabilities_choice: tuple[str, tuple[tuple[str, date] | None, tuple[str, date] | None]],
    fcf_choice: tuple[str, tuple[str | None, str | None]],
) -> None:
    _cash_name, cash = cash_choice
    _debt_name, (short_term_debt, current_long_term_debt) = debt_choice
    _al_name, (current_assets, current_liabilities) = assets_liabilities_choice
    _fcf_name, (operating_cash_flow, capital_expenditure) = fcf_choice
    facts = _solvency_facts(
        cash=cash,
        short_term_debt=short_term_debt,
        current_long_term_debt=current_long_term_debt,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        operating_cash_flow=operating_cash_flow,
        capital_expenditure=capital_expenditure,
    )
    payload = _build_payload(sec_config, facts=facts)
    panel = _assert_readable(payload)
    solvency = panel["solvency"]
    assert solvency["state"] in ("assessed", "withheld")


def test_solvency_grid_reaches_every_state_and_runway_branch(sec_config) -> None:
    """Guards the grid itself: it must not vacuously pass by only ever
    reaching one state or one runway branch."""
    seen_states: set[str] = set()
    seen_runway: set[str] = set()
    for cash_choice, debt_choice, assets_liabilities_choice, fcf_choice in _SOLVENCY_GRID:
        cash = cash_choice[1]
        short, long = debt_choice[1]
        assets, liab = assets_liabilities_choice[1]
        ocf, capex = fcf_choice[1]
        facts = _solvency_facts(
            cash=cash,
            short_term_debt=short,
            current_long_term_debt=long,
            current_assets=assets,
            current_liabilities=liab,
            operating_cash_flow=ocf,
            capital_expenditure=capex,
        )
        payload = _build_payload(sec_config, facts=facts)
        seen_states.add(payload["solvency"]["status"])
        seen_runway.add(payload["solvency"]["runway"]["status"])
    assert seen_states == {
        SOLVENCY_NO_ADVERSE_EVIDENCE,
        SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION,
        SOLVENCY_ELEVATED_OBLIGATION_RISK,
        SOLVENCY_INSUFFICIENT_EVIDENCE,
    }
    assert seen_runway == {RUNWAY_COMPUTED, RUNWAY_NOT_APPLICABLE, RUNWAY_WITHHELD}


@pytest.mark.parametrize(
    "mismatch_concept",
    [
        "cash_and_equivalents",
        "short_term_debt",
        "current_long_term_debt",
        "current_assets",
        "current_liabilities",
    ],
)
def test_a_genuine_instant_period_mismatch_is_readable_and_withheld(
    sec_config, mismatch_concept: str
) -> None:
    shifted = INSTANT_DATE + timedelta(days=30)
    base = {
        "cash": ("1000.00000000", INSTANT_DATE),
        "short_term_debt": ("100.00000000", INSTANT_DATE),
        "current_long_term_debt": ("50.00000000", INSTANT_DATE),
        "current_assets": ("2000.00000000", INSTANT_DATE),
        "current_liabilities": ("1000.00000000", INSTANT_DATE),
    }
    concept_to_kwarg = {
        "cash_and_equivalents": "cash",
        "short_term_debt": "short_term_debt",
        "current_long_term_debt": "current_long_term_debt",
        "current_assets": "current_assets",
        "current_liabilities": "current_liabilities",
    }
    kwarg = concept_to_kwarg[mismatch_concept]
    value, _period_end = base[kwarg]
    base[kwarg] = (value, shifted)
    facts = _solvency_facts(**base)
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert "instant_period_mismatch" in payload["solvency"]["reasons"]
    _assert_readable(payload)


# ---------------------------------------------------------------------------
# Liquidity matrix: 14 branches, every one readable.
# ---------------------------------------------------------------------------


def test_liquidity_computed_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["liquidity"]["status"] == LIQUIDITY_COMPUTED
    _assert_readable(payload)


def test_liquidity_missing_anchor_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config, price_asset=None, price_source=None)
    assert payload["liquidity"]["reason"] == LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE
    assert payload["liquidity"]["price_asset"] is None
    _assert_readable(payload)


def test_liquidity_mismatched_anchor_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config, price_source={"asset_id": str(uuid4())})
    assert payload["liquidity"]["reason"] == LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE
    assert payload["liquidity"]["price_asset"] is None
    _assert_readable(payload)


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "interval": "1week",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "currency": "USD",
        },
        {
            "interval": "1day",
            "adjustment": "unadjusted",
            "return_definition": "split_adjusted_price_return",
            "currency": "USD",
        },
        {
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "total_return",
            "currency": "USD",
        },
        {
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "currency": "EUR",
        },
        {},
    ],
    ids=["bad-interval", "bad-adjustment", "bad-return-definition", "bad-currency-only", "empty"],
)
def test_liquidity_incompatible_metadata_branch_is_readable(
    sec_config, metadata: dict[str, str]
) -> None:
    payload = _build_payload(sec_config, price_asset=_price_asset(metadata=metadata))
    assert payload["liquidity"]["reason"] == LIQUIDITY_BASIS_INCOMPATIBLE
    assert payload["liquidity"]["price_asset"] is not None
    _assert_readable(payload)


def test_liquidity_invalid_session_date_rows_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config, invalid_session_date_rows=1)
    assert payload["liquidity"]["status"] == LIQUIDITY_WITHHELD
    _assert_readable(payload)


def test_liquidity_insufficient_sessions_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=251))
    assert payload["liquidity"]["reason"] == "insufficient_sessions"
    _assert_readable(payload)


def test_liquidity_duplicate_sessions_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(duplicate_last=True))
    assert payload["liquidity"]["reason"] == "duplicate_sessions"
    _assert_readable(payload)


def test_liquidity_invalid_close_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(close=0.0))
    assert payload["liquidity"]["reason"] == "invalid_close_values"
    _assert_readable(payload)


def test_liquidity_future_price_session_branch_is_readable(sec_config) -> None:
    future = TARGET_DATE + timedelta(days=5)
    payload = _build_payload(sec_config, price_frame=_price_frame(last_session=future))
    assert payload["liquidity"]["reason"] == LIQUIDITY_FUTURE_PRICE_SESSION
    _assert_readable(payload)


def test_liquidity_stale_price_evidence_branch_is_readable(sec_config) -> None:
    stale = TARGET_DATE - timedelta(days=30)
    payload = _build_payload(sec_config, price_frame=_price_frame(last_session=stale))
    assert payload["liquidity"]["reason"] == LIQUIDITY_STALE_PRICE_EVIDENCE
    _assert_readable(payload)


def test_liquidity_branches_reach_at_least_fourteen_distinct_shapes(sec_config) -> None:
    """Guards the liquidity matrix itself: must not vacuously pass on one shape."""
    seen: set[tuple[str, str | None]] = set()
    scenarios = [
        _build_payload(sec_config),
        _build_payload(sec_config, price_asset=None, price_source=None),
        _build_payload(sec_config, price_source={"asset_id": str(uuid4())}),
        _build_payload(
            sec_config,
            price_asset=_price_asset(
                metadata={
                    "interval": "1week",
                    "adjustment": "splits",
                    "return_definition": "split_adjusted_price_return",
                    "currency": "USD",
                }
            ),
        ),
        _build_payload(
            sec_config,
            price_asset=_price_asset(
                metadata={
                    "interval": "1day",
                    "adjustment": "unadjusted",
                    "return_definition": "split_adjusted_price_return",
                    "currency": "USD",
                }
            ),
        ),
        _build_payload(
            sec_config,
            price_asset=_price_asset(
                metadata={
                    "interval": "1day",
                    "adjustment": "splits",
                    "return_definition": "total_return",
                    "currency": "USD",
                }
            ),
        ),
        _build_payload(
            sec_config,
            price_asset=_price_asset(
                metadata={
                    "interval": "1day",
                    "adjustment": "splits",
                    "return_definition": "split_adjusted_price_return",
                    "currency": "EUR",
                }
            ),
        ),
        _build_payload(sec_config, price_asset=_price_asset(metadata={})),
        _build_payload(sec_config, invalid_session_date_rows=1),
        _build_payload(sec_config, price_frame=_price_frame(sessions=251)),
        _build_payload(sec_config, price_frame=_price_frame(duplicate_last=True)),
        _build_payload(sec_config, price_frame=_price_frame(close=0.0)),
        _build_payload(
            sec_config, price_frame=_price_frame(last_session=TARGET_DATE + timedelta(days=5))
        ),
        _build_payload(
            sec_config, price_frame=_price_frame(last_session=TARGET_DATE - timedelta(days=30))
        ),
    ]
    for payload in scenarios:
        liquidity = payload["liquidity"]
        seen.add((liquidity["status"], liquidity["reason"]))
        _assert_readable(payload)
    assert len(scenarios) == 14
    # Several scenarios legitimately collapse to the same (status, reason)
    # shape (e.g. every incompatible-metadata variant is `basis_incompatible`;
    # missing and mismatched anchors are both `price_provenance_unavailable`),
    # so this is not 14 -- but it must still span most of the actual
    # `_build_liquidity`/`median_dollar_volume` reason vocabulary, not
    # collapse to one or two shapes.
    assert len(seen) >= 9


# ---------------------------------------------------------------------------
# F1: the generator's genuinely signed runway domain must be accepted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cash_value",
    ["-1.00000000", "0.00000000", "1000.00000000"],
    ids=["negative-cash", "zero-cash", "positive-cash"],
)
def test_a_legitimate_negative_runway_with_complete_inputs_is_readable(
    sec_config, cash_value: str
) -> None:
    """Cash=-1, FCF=-100 must emit a readable ``computed`` runway of
    ``"-0.0400"`` -- the generator applies no cash eligibility rule to
    ``_build_runway``, and a reader that requires nonnegative quarters
    incorrectly rejects a real, signed generator output."""
    facts = _solvency_facts(
        cash=(cash_value, INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    runway = payload["solvency"]["runway"]
    assert runway["status"] == RUNWAY_COMPUTED
    assert runway["reason"] is None
    if cash_value.startswith("-"):
        assert runway["quarters"].startswith("-")
    _assert_readable(payload)


def test_a_legitimate_negative_runway_survives_an_unrelated_missing_obligation_input(
    sec_config,
) -> None:
    """The independent cash/negative-FCF runway diagnostic stays computed even
    when an unrelated obligation input (here, both debt components) is
    missing -- `_build_runway` reads only ``cash``/``free_cash_flow``."""
    facts = _solvency_facts(
        cash=("-1.00000000", INSTANT_DATE),
        short_term_debt=None,
        current_long_term_debt=None,
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    runway = payload["solvency"]["runway"]
    assert runway["status"] == RUNWAY_COMPUTED
    assert runway["quarters"] == "-0.0400"
    panel = _assert_readable(payload)
    assert panel["solvency"]["runway"]["quarters"] == "-0.0400"


def test_reader_rejects_a_negative_runway_only_before_this_fix(sec_config) -> None:
    """Mutation-style documentation: proves the exact prior defect. A
    deliberately reintroduced ``>= 0`` requirement (mirrored inline here,
    not by editing production) would have rejected the payload above."""
    facts = _solvency_facts(
        cash=("-1.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    quarters = Decimal(payload["solvency"]["runway"]["quarters"])
    assert quarters < 0, "fixture no longer reproduces a negative runway"


# ---------------------------------------------------------------------------
# RI-3: computed runway's sign must be consistent with cash's own sign bit,
# including a genuine signed negative zero -- never recomputed magnitude.
# ---------------------------------------------------------------------------


def test_a_genuine_signed_negative_zero_runway_is_readable(sec_config) -> None:
    """A cash value negative enough to be real, but tiny enough that
    ``4 * cash / abs(fcf)`` quantizes to a *signed* ``"-0.0000"`` (not the
    plain unsigned ``"0.0000"`` a zero cash produces): `Decimal.is_signed()`
    is what correctly reads this as "cash was recorded negative", never a
    numeric ``< 0`` comparison, which cannot tell the two zeros apart."""
    facts = _solvency_facts(
        cash=("-0.00000001", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    runway = payload["solvency"]["runway"]
    assert runway["status"] == RUNWAY_COMPUTED
    assert runway["quarters"] == "-0.0000"
    assert Decimal(runway["quarters"]).is_signed()
    panel = _assert_readable(payload)
    assert panel["solvency"]["runway"]["quarters"] == "-0.0000"


@pytest.mark.parametrize(
    ("cash_value", "genuine_quarters", "opposite_sign_quarters"),
    [
        ("-1.00000000", "-0.0400", "0.0400"),
        ("1000.00000000", "40.0000", "-40.0000"),
        ("-0.00000001", "-0.0000", "0.0000"),
    ],
    ids=["negative-cash", "positive-cash", "signed-negative-zero-cash"],
)
def test_a_runway_sign_opposite_to_cash_is_rejected(
    sec_config, cash_value: str, genuine_quarters: str, opposite_sign_quarters: str
) -> None:
    """RI-3: flipping only the stored quarters' sign (leaving cash, FCF, and
    every other field genuine) must be caught even though the reader never
    recomputes the quotient's magnitude -- only its sign bit against cash's
    own, via `Decimal.is_signed()`."""
    facts = _solvency_facts(
        cash=(cash_value, INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    assert payload["solvency"]["runway"]["quarters"] == genuine_quarters
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["runway"]["quarters"] = opposite_sign_quarters
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# Residual: an impossible withheld-runway combination must never be
# accepted. `_build_runway` only ever reaches `withheld` when free cash
# flow is itself unavailable, or when it is negative and cash is
# unavailable -- never when both are genuinely present and finite (its
# only other branch there, a defensive precision-64 quotient-finiteness
# guard, is confirmed unreachable below).
# ---------------------------------------------------------------------------


def test_runway_withheld_when_free_cash_flow_is_missing_is_readable(sec_config) -> None:
    facts = _solvency_facts(operating_cash_flow=None, capital_expenditure=None)
    payload = _build_payload(sec_config, facts=facts)
    runway = payload["solvency"]["runway"]
    assert runway["status"] == RUNWAY_WITHHELD
    assert runway["quarters"] is None
    assert runway["reason"] == "free_cash_flow_missing"
    _assert_readable(payload)


def test_runway_withheld_when_cash_is_missing_with_negative_fcf_is_readable(sec_config) -> None:
    facts = _solvency_facts(
        cash=None,
        operating_cash_flow="100.00000000",
        capital_expenditure="400.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    runway = payload["solvency"]["runway"]
    assert runway["status"] == RUNWAY_WITHHELD
    assert runway["quarters"] is None
    assert runway["reason"] == "cash_and_equivalents_missing"
    _assert_readable(payload)


def test_the_defensive_nonfinite_quotient_runway_branch_is_confirmed_unreachable() -> None:
    """Documents the due-diligence behind accepting only two withheld
    sub-cases, rather than guessing a third.

    `_build_runway`'s own third branch (negative free cash flow, cash
    present, but ``4 * cash / abs(free_cash_flow)`` itself non-finite) can
    only be reached by overflowing the ambient Decimal context's ``Emax``
    exponent bound. Even the most extreme combination representable under
    this reader's own canonical fixed-point boundary -- the widest
    quantizable cash value divided by the smallest nonzero quantizable
    free cash flow -- reaches an exponent orders of magnitude short of
    that bound, and any attempt to push further instead raises inside
    `_build_runway`'s own quantization (a separate, pre-existing generator
    concern, out of this reader-only slice's scope) before ever reaching a
    withheld return. So the reader correctly never special-cases it.
    """
    from decimal import ROUND_HALF_EVEN, Decimal, localcontext

    widest_cash = Decimal("9" * 56 + ".00000000")
    smallest_nonzero_fcf_magnitude = Decimal("0.00000001")
    with localcontext() as context:
        context.prec = 64
        context.rounding = ROUND_HALF_EVEN
        quotient = (Decimal(4) * widest_cash) / smallest_nonzero_fcf_magnitude
    assert quotient.is_finite()
    # `Emax` defaults to 999999; the widest reachable exponent here is 64 --
    # nowhere near enough to overflow to Infinity through this division.
    assert quotient.adjusted() < 1000


@pytest.mark.parametrize(
    "arbitrary_reason", ["cash_and_equivalents_missing", "totally_forged_reason"]
)
def test_a_computed_runway_relabeled_withheld_is_rejected(
    sec_config, arbitrary_reason: str
) -> None:
    """The exact residual counterexample: a genuine `computed` runway (both
    free cash flow and cash finitely present) relabeled `withheld` with
    `quarters=None` must never pass, in either hash variant."""
    facts = _solvency_facts(
        cash=("-1.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    assert payload["solvency"]["inputs"]["cash_and_equivalents"] == "-1.00000000"
    assert payload["solvency"]["inputs"]["free_cash_flow"] == "-100.00000000"
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["runway"] = {
        "status": RUNWAY_WITHHELD,
        "quarters": None,
        "reason": arbitrary_reason,
    }
    _assert_unsupported(corrupted)


def test_a_not_applicable_runway_relabeled_withheld_is_rejected(sec_config) -> None:
    """The sibling counterexample from a genuine `not_applicable_positive_fcf`
    runway (non-negative free cash flow) relabeled `withheld`."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_NOT_APPLICABLE
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["runway"] = {
        "status": RUNWAY_WITHHELD,
        "quarters": None,
        "reason": "totally_forged_reason",
    }
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# F2: contradictory solvency inputs/favorable wording must never be accepted.
# ---------------------------------------------------------------------------


def test_not_applicable_runway_paired_with_negative_fcf_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_NOT_APPLICABLE
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["free_cash_flow"] = "-1.00000000"
    _assert_unsupported(corrupted)


def test_not_applicable_runway_paired_with_missing_fcf_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["free_cash_flow"] = None
    corrupted["solvency"]["status"] = SOLVENCY_INSUFFICIENT_EVIDENCE
    corrupted["solvency"]["reasons"] = ["free_cash_flow_missing"]
    # runway.status stays "not_applicable_positive_fcf" -- the corruption.
    _assert_unsupported(corrupted)


def test_computed_runway_paired_with_nonnegative_fcf_is_rejected(sec_config) -> None:
    facts = _solvency_facts(
        cash=("-1.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["free_cash_flow"] = "100.00000000"
    _assert_unsupported(corrupted)


def test_zero_current_liabilities_cannot_coexist_with_a_computed_ratio(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["solvency"]["inputs"]["current_ratio"] is not None
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["current_liabilities"] = "0.00000000"
    _assert_unsupported(corrupted)


def test_negative_current_liabilities_cannot_be_presented_usable(sec_config) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["current_liabilities"] = "-1.00000000"
    _assert_unsupported(corrupted)


def test_a_ratio_without_a_present_liabilities_figure_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["status"] = SOLVENCY_INSUFFICIENT_EVIDENCE
    corrupted["solvency"]["reasons"] = ["current_liabilities_missing"]
    corrupted["solvency"]["inputs"]["current_liabilities"] = None
    # current_ratio stays populated -- the corruption.
    _assert_unsupported(corrupted)


def test_a_ratio_without_a_present_current_assets_figure_is_rejected(sec_config) -> None:
    """RI-5: the ratio's numerator, corrupted independently of its
    denominator -- both operands must be present, not merely liabilities."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["status"] = SOLVENCY_INSUFFICIENT_EVIDENCE
    corrupted["solvency"]["reasons"] = ["current_assets_missing"]
    corrupted["solvency"]["inputs"]["current_assets"] = None
    # current_ratio (and current_liabilities) stay populated -- the corruption.
    _assert_unsupported(corrupted)


def test_a_ratio_survives_an_unrelated_missing_debt_input(sec_config) -> None:
    """A legitimate ratio (both operands genuinely present) must not be
    rejected merely because an unrelated obligation input is missing."""
    facts = _solvency_facts(short_term_debt=None, current_long_term_debt=None)
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["inputs"]["current_ratio"] == "2.0000"
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    panel = _assert_readable(payload)
    rendered_ratio = next(
        item for item in panel["solvency"]["inputs"] if item["label"].startswith("Current ratio")
    )
    assert rendered_ratio["value"] == "2.0000"


def test_a_partial_duration_tuple_is_rejected(sec_config) -> None:
    """Isolates the duration-tuple invariant specifically: `current_liabilities`
    missing makes the overall state `insufficient_evidence` (which does not,
    by itself, require every period field non-null), so only the dedicated
    all-or-nothing duration-tuple check can catch a partial tuple here."""
    facts = _solvency_facts(current_liabilities=None)
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["periods"]["instant_date"] is None
    assert payload["solvency"]["periods"]["duration_start"] is not None
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["periods"]["duration_start"] = None
    _assert_unsupported(corrupted)


def test_no_adverse_state_with_a_nonempty_reasons_list_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["reasons"] = ["near_term_debt_exceeds_cash"]
    _assert_unsupported(corrupted)


def test_adverse_state_without_its_necessary_reason_is_rejected(sec_config) -> None:
    facts = _solvency_facts(
        cash=("0.00000000", INSTANT_DATE),
        current_assets=("500.00000000", INSTANT_DATE),
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["reasons"] = ["current_assets_below_current_liabilities"]
    _assert_unsupported(corrupted)


def test_elevated_state_with_an_unrecognized_reason_is_rejected(sec_config) -> None:
    facts = _solvency_facts(operating_cash_flow="100.00000000", capital_expenditure="400.00000000")
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["reasons"] = ["totally_forged_reason"]
    _assert_unsupported(corrupted)


def test_elevated_debt_alone_status_changed_to_adverse_is_rejected(sec_config) -> None:
    """RI-4: `near_term_debt_exceeds_cash` alone is only ever elevated --
    `_classify_solvency`'s adverse conjunction additionally requires assets
    below liabilities or a short runway. Genuinely reached with only cash
    zeroed out (assets stay above liabilities, FCF stays positive)."""
    facts = _solvency_facts(cash=("0.00000000", INSTANT_DATE))
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_ELEVATED_OBLIGATION_RISK
    assert payload["solvency"]["reasons"] == ["near_term_debt_exceeds_cash"]
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["status"] = SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    _assert_unsupported(corrupted)


def test_adverse_conjunction_status_downgraded_to_elevated_is_rejected(sec_config) -> None:
    """RI-4: the mirror-image corruption -- a genuine adverse conjunction
    (debt exceeds cash *and* assets below liabilities) relabeled elevated."""
    facts = _solvency_facts(
        cash=("0.00000000", INSTANT_DATE),
        current_assets=("500.00000000", INSTANT_DATE),
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    assert set(payload["solvency"]["reasons"]) == {
        "near_term_debt_exceeds_cash",
        "current_assets_below_current_liabilities",
    }
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["status"] = SOLVENCY_ELEVATED_OBLIGATION_RISK
    _assert_unsupported(corrupted)


def test_runway_headline_never_claims_fcf_is_non_negative_for_a_negative_fcf(
    sec_config,
) -> None:
    """Panel copy must describe the *validated* runway/FCF branch: a payload
    the reader accepts with a negative FCF must never render "FCF is
    non-negative", regardless of overall solvency state."""
    facts = _solvency_facts(
        cash=("-1.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    panel = _assert_readable(payload)
    assert "FCF is non-negative" not in panel["solvency"]["runway"]["headline"]
    assert "Cash runway - -0.0400 quarters" in panel["solvency"]["runway"]["headline"]


# ---------------------------------------------------------------------------
# F3: affirmative price-basis wording requires the complete validated branch.
# ---------------------------------------------------------------------------


def test_a_computed_payload_with_a_forged_null_anchor_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["liquidity"]["status"] == LIQUIDITY_COMPUTED
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["price_asset"] = None
    _assert_unsupported(corrupted)


def test_a_computed_payload_with_a_malformed_anchor_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["price_asset"] = {"id": "not-a-uuid", "sha256": "z" * 64}
    _assert_unsupported(corrupted)


def test_provenance_unavailable_with_a_forged_compatible_basis_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config, price_asset=None, price_source=None)
    assert payload["liquidity"]["reason"] == LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["basis"] = {
        "interval": "1day",
        "adjustment": "splits",
        "return_definition": "split_adjusted_price_return",
        "volume_basis": "provider_reported_unverified_split_basis",
    }
    _assert_unsupported(corrupted)


def test_currency_only_incompatible_basis_never_renders_the_proof_claim(sec_config) -> None:
    """The exact F3 counterexample: interval/adjustment/return_definition all
    happen to match, but currency alone was wrong -- must still read
    `basis_incompatible` and never claim the price basis is confirmed."""
    payload = _build_payload(
        sec_config,
        price_asset=_price_asset(
            metadata={
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "currency": "EUR",
            }
        ),
    )
    assert payload["liquidity"]["reason"] == LIQUIDITY_BASIS_INCOMPATIBLE
    assert payload["liquidity"]["basis"]["interval"] == "1day"
    assert payload["liquidity"]["basis"]["adjustment"] == "splits"
    assert payload["liquidity"]["basis"]["return_definition"] == "split_adjusted_price_return"
    panel = _assert_readable(payload)
    assert "Split-only price basis is confirmed" not in panel["liquidity"]["volume_basis_caveat"]
    assert "not established" in panel["liquidity"]["volume_basis_caveat"]


@pytest.mark.parametrize(
    ("candidate_reason", "expect_valid"),
    [
        # These three share `basis_incompatible`'s exact stored shape --
        # valid anchor, a basis dict that (for this currency-only mismatch)
        # is byte-identical to the fully compatible constant, and a fully
        # null session window -- because a currency mismatch is the one
        # incompatibility never separately recorded anywhere else in the
        # payload. The reader cannot tell a genuine one of these three
        # apart from a forged relabeling of this exact currency-only
        # mismatch using stored fields alone, so it must not be the more
        # permissive of the two readings: the payload is still a
        # structurally legitimate *withheld* shape (so overall validity
        # does not collapse), but it must never affirm the price basis --
        # see the dedicated caveat-wording assertion below.
        (DOLLAR_VOLUME_MISSING_COLUMNS, True),
        (DOLLAR_VOLUME_INVALID_SESSION_DATES, True),
        (DOLLAR_VOLUME_DUPLICATE_SESSIONS, True),
        # Every other recognized reason requires a session-window shape
        # (null, or a nonzero count with real dates) that this base
        # payload's fully-null window does not satisfy, so relabeling onto
        # one of these is a structural contradiction, not merely an
        # unproven claim.
        (DOLLAR_VOLUME_INSUFFICIENT_SESSIONS, False),
        (DOLLAR_VOLUME_INVALID_CLOSE, False),
        (DOLLAR_VOLUME_INVALID_VOLUME, False),
        (DOLLAR_VOLUME_NONFINITE_PRODUCT, False),
        (DOLLAR_VOLUME_DROPPED_ROWS, False),
        (DOLLAR_VOLUME_NONFINITE_MEDIAN, False),
        (LIQUIDITY_STALE_PRICE_EVIDENCE, False),
        (LIQUIDITY_FUTURE_PRICE_SESSION, False),
        # `price_provenance_unavailable` requires a null anchor; this base
        # payload's anchor is valid.
        (LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE, False),
        # Never a recognized reason at all.
        ("totally_forged_reason", False),
        ("", False),
    ],
)
def test_reason_relabeling_from_a_currency_only_incompatible_basis(
    sec_config, candidate_reason: str, expect_valid: bool
) -> None:
    """RI-1: enumerate recognized/unknown reasons from the one base shape
    that is genuinely ambiguous with three other reasons, proving the
    reader's response is the exact, deliberate structural contract above --
    never a blacklist of two reasons, and never a false affirmation."""
    payload = _build_payload(
        sec_config,
        price_asset=_price_asset(
            metadata={
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "currency": "EUR",
            }
        ),
    )
    assert payload["liquidity"]["reason"] == LIQUIDITY_BASIS_INCOMPATIBLE
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["reason"] = candidate_reason
    recomputed = _recompute_hash(corrupted)
    is_valid = _is_valid_stub_payload(_round_trip(recomputed))
    assert is_valid is expect_valid
    if is_valid:
        panel = _under10_panel(
            _stub_analysis(data_quality={"under10_assessment": _round_trip(recomputed)}),
            current_price_band=None,
        )
        assert panel["state"] == "recorded"
        assert (
            "Split-only price basis is confirmed" not in panel["liquidity"]["volume_basis_caveat"]
        )
    else:
        _assert_unsupported(corrupted)


def test_insufficient_sessions_with_a_confirmed_anchor_still_affirms_the_price_basis(
    sec_config,
) -> None:
    """Acceptable per F3 item 3: a withheld-for-an-unrelated-reason result may
    still affirm the price basis once the generator actually established it."""
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=251))
    panel = _assert_readable(payload)
    assert "Split-only price basis is confirmed" in panel["liquidity"]["volume_basis_caveat"]


# ---------------------------------------------------------------------------
# F4: an impossible 252-session calendar window must never be accepted.
# ---------------------------------------------------------------------------


def test_a_252_session_window_spanning_two_calendar_days_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["liquidity"]["sessions_used"] == 252
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["first_session"] = "2026-03-01"
    corrupted["liquidity"]["last_session"] = "2026-03-02"
    _assert_unsupported(corrupted)


def test_a_252_session_window_at_the_exact_251_day_boundary_is_readable(sec_config) -> None:
    """Necessary, not sufficient: exactly 251 calendar days apart is the
    tightest span 252 *distinct* dates can occupy, and must not be rejected
    outright (only shorter, arithmetically impossible spans are refused)."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["first_session"] = "2025-06-25"
    corrupted["liquidity"]["last_session"] = "2026-03-02"
    assert (date(2026, 3, 2) - date(2025, 6, 25)).days == 250
    # 250 days is one short of the necessary bound -- still impossible.
    _assert_unsupported(corrupted)
    corrupted["liquidity"]["first_session"] = "2025-06-24"
    assert (date(2026, 3, 2) - date(2025, 6, 24)).days == 251
    _assert_readable(_recompute_hash(corrupted))


def test_equal_first_and_last_session_withheld_diagnostics_remain_readable(sec_config) -> None:
    """A withheld branch legitimately reporting a single (or zero) observed
    session is not subject to the 252-session feasibility bound at all."""
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=1))
    assert payload["liquidity"]["status"] == LIQUIDITY_WITHHELD
    assert payload["liquidity"]["sessions_used"] == 1
    assert payload["liquidity"]["first_session"] == payload["liquidity"]["last_session"]
    _assert_readable(payload)


# ---------------------------------------------------------------------------
# RI-2: per-reason session-window/temporal contracts, mutated one field at a
# time from a genuine payload -- never an exchange-calendar or price read.
# ---------------------------------------------------------------------------


def test_a_computed_last_session_after_the_target_date_is_rejected(sec_config) -> None:
    """The exact RI-2 counterexample: a computed result's last session must
    never be reported after the decision's own evaluated target date."""
    payload = _build_payload(sec_config)
    assert payload["liquidity"]["status"] == LIQUIDITY_COMPUTED
    corrupted = json.loads(json.dumps(payload))
    original_last = date.fromisoformat(corrupted["liquidity"]["last_session"])
    corrupted["liquidity"]["last_session"] = (original_last + timedelta(days=1)).isoformat()
    _assert_unsupported(corrupted)


def test_a_computed_reversed_session_window_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["first_session"], corrupted["liquidity"]["last_session"] = (
        corrupted["liquidity"]["last_session"],
        corrupted["liquidity"]["first_session"],
    )
    _assert_unsupported(corrupted)


def test_a_computed_result_carrying_a_withholding_reason_is_rejected(sec_config) -> None:
    """A `computed` result never carries a `reason` -- `_build_liquidity`
    only ever sets one on a `withheld` result."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["reason"] = LIQUIDITY_STALE_PRICE_EVIDENCE
    _assert_unsupported(corrupted)


@pytest.mark.parametrize("sessions_used", [-1])
def test_insufficient_sessions_negative_count_is_rejected(sec_config, sessions_used: int) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=200))
    assert payload["liquidity"]["reason"] == DOLLAR_VOLUME_INSUFFICIENT_SESSIONS
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["sessions_used"] = sessions_used
    corrupted["liquidity"]["first_session"] = None
    corrupted["liquidity"]["last_session"] = None
    _assert_unsupported(corrupted)


@pytest.mark.parametrize("sessions_used", [252, 253])
def test_insufficient_sessions_at_or_above_the_full_window_is_rejected(
    sec_config, sessions_used: int
) -> None:
    """Isolates the reason-specific upper bound (``<= 251``) from the
    separate calendar-feasibility check: this window's real dates span
    comfortably enough calendar days for 252+ sessions, so only the
    ``insufficient_sessions``-specific bound can be what rejects it."""
    payload = _build_payload(sec_config)
    assert payload["liquidity"]["status"] == LIQUIDITY_COMPUTED
    assert payload["liquidity"]["sessions_used"] == 252
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["status"] = LIQUIDITY_WITHHELD
    corrupted["liquidity"]["value"] = None
    corrupted["liquidity"]["reason"] = DOLLAR_VOLUME_INSUFFICIENT_SESSIONS
    corrupted["liquidity"]["sessions_used"] = sessions_used
    _assert_unsupported(corrupted)


def test_insufficient_sessions_zero_count_with_non_null_dates_is_rejected(sec_config) -> None:
    """The empty-frame sub-case (`sessions_used=0`) always pairs with a fully
    null window -- a zero count claiming real dates is a partial pair."""
    empty_frame = pl.DataFrame({"date": [], "close": [], "volume": []})
    payload = _build_payload(sec_config, price_frame=empty_frame)
    assert payload["liquidity"]["sessions_used"] == 0
    assert payload["liquidity"]["first_session"] is None
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["first_session"] = TARGET_DATE.isoformat()
    corrupted["liquidity"]["last_session"] = TARGET_DATE.isoformat()
    _assert_unsupported(corrupted)


def test_insufficient_sessions_positive_count_with_a_null_window_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=200))
    assert payload["liquidity"]["sessions_used"] == 200
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["first_session"] = None
    corrupted["liquidity"]["last_session"] = None
    _assert_unsupported(corrupted)


def test_insufficient_sessions_with_a_reversed_window_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=200))
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["first_session"], corrupted["liquidity"]["last_session"] = (
        corrupted["liquidity"]["last_session"],
        corrupted["liquidity"]["first_session"],
    )
    _assert_unsupported(corrupted)


def test_insufficient_sessions_with_a_future_last_session_is_rejected(sec_config) -> None:
    """Unlike `future_price_session`, `insufficient_sessions` has no
    exception to the "last session on or before target date" rule."""
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=200))
    corrupted = json.loads(json.dumps(payload))
    last = date.fromisoformat(corrupted["liquidity"]["last_session"])
    corrupted["liquidity"]["last_session"] = (last + timedelta(days=1)).isoformat()
    _assert_unsupported(corrupted)


def test_insufficient_sessions_with_an_impossible_calendar_span_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=200))
    corrupted = json.loads(json.dumps(payload))
    last = date.fromisoformat(corrupted["liquidity"]["last_session"])
    corrupted["liquidity"]["first_session"] = (last - timedelta(days=50)).isoformat()
    _assert_unsupported(corrupted)


def test_a_null_window_reason_with_a_forged_nonnull_window_is_rejected(sec_config) -> None:
    """`basis_incompatible`'s branch never inspects a window at all; a
    forged nonnull window paired with it is a structural contradiction."""
    payload = _build_payload(sec_config, price_asset=_price_asset(metadata={}))
    assert payload["liquidity"]["reason"] == LIQUIDITY_BASIS_INCOMPATIBLE
    assert payload["liquidity"]["sessions_used"] is None
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["sessions_used"] = 5
    corrupted["liquidity"]["first_session"] = "2026-02-25"
    corrupted["liquidity"]["last_session"] = "2026-03-01"
    _assert_unsupported(corrupted)


@pytest.mark.parametrize("sessions_used", [0, 253])
def test_a_windowed_reason_outside_its_one_to_252_bound_is_rejected(
    sec_config, sessions_used: int
) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(close=0.0))
    assert payload["liquidity"]["reason"] == DOLLAR_VOLUME_INVALID_CLOSE
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["sessions_used"] = sessions_used
    if sessions_used == 0:
        corrupted["liquidity"]["first_session"] = None
        corrupted["liquidity"]["last_session"] = None
    _assert_unsupported(corrupted)


@pytest.mark.parametrize("sessions_used", [251, 253])
def test_the_full_window_reason_off_exactly_252_is_rejected(sec_config, sessions_used: int) -> None:
    """`nonfinite_median` is only ever reached once a *complete*
    ``UNDER10_LIQUIDITY_SESSIONS``-session window was already confirmed --
    never a partial or an over-wide one."""
    payload = _build_payload(
        sec_config,
        facts=_solvency_facts(),
        # Each row's own `close * volume` stays finite (9e307); only the
        # even-count *median* -- which numpy computes by averaging its two
        # middle values -- overflows, since their sum exceeds float64 max.
        price_frame=_price_frame(close=9e307, volume=1.0),
    )
    assert payload["liquidity"]["reason"] == DOLLAR_VOLUME_NONFINITE_MEDIAN
    assert payload["liquidity"]["sessions_used"] == 252
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["sessions_used"] = sessions_used
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    "last_session",
    ["2026-03-02", "2026-02-28"],
    ids=["on-target-date", "before-target-date"],
)
def test_future_price_session_without_a_genuinely_future_last_session_is_rejected(
    sec_config, last_session: str
) -> None:
    """`future_price_session`'s entire premise is a last session strictly
    after ``target_date``; on or before it is a contradiction, never the
    same withheld shape wearing a different label."""
    payload = _build_payload(
        sec_config, price_frame=_price_frame(last_session=TARGET_DATE + timedelta(days=5))
    )
    assert payload["liquidity"]["reason"] == LIQUIDITY_FUTURE_PRICE_SESSION
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["last_session"] = last_session
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("target_date", "not-a-date"), ("target_date", "2026-03-32"), ("target_date", 20260302)],
)
def test_a_malformed_evaluated_for_target_date_is_rejected(
    sec_config, field: str, bad_value: object
) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["evaluated_for"][field] = bad_value
    _assert_unsupported(corrupted)


def test_a_target_date_mismatched_against_the_actual_analysis_run_is_rejected(sec_config) -> None:
    """RI-2: a stored `evaluated_for.target_date` must equal the persisted
    `AnalysisRun.target_date` it is rendered against -- a forged/stale
    target date must never masquerade as this decision's own, even though
    every other field is genuine and its own hash matches."""
    payload = _build_payload(sec_config)
    round_tripped = _round_trip(payload)
    assert _is_valid_stub_payload(round_tripped, expected_target_date=TARGET_DATE) is True
    mismatched_run_date = TARGET_DATE + timedelta(days=1)
    assert _is_valid_stub_payload(round_tripped, expected_target_date=mismatched_run_date) is False
    panel = _under10_panel(
        _stub_analysis(data_quality={"under10_assessment": round_tripped}),
        current_price_band=None,
    )
    # `_stub_analysis` itself carries `TARGET_DATE`, matching the payload --
    # this proves the cross-check via the validator directly, above, and
    # confirms the matching-date case still renders through the same panel.
    assert panel["state"] == "recorded"


# ---------------------------------------------------------------------------
# F5: extreme Decimal exponents (either sign) must never be accepted; the
# generator's own canonical fixed-point boundary outputs must still pass.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        "1e100000000",
        "-1e100000000",
        "1e-100000000",
        "-1e-100000000",
        "1" * 200,
        "NaN",
        "Infinity",
        "-Infinity",
        "1.123456789",  # one decimal place too many for an 8-place field
        "5",  # missing the required 8 decimal places entirely
        5.0,  # not a string at all
        True,
    ],
)
def test_an_extreme_or_malformed_monetary_string_is_rejected(sec_config, bad_value) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["cash_and_equivalents"] = bad_value
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    "bad_value",
    ["1e-100000000", "-1e-100000000", "1e100000000", "NaN"],
)
def test_an_extreme_reference_close_is_rejected(sec_config, bad_value) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["evaluated_for"]["reference_close"] = bad_value
    _assert_unsupported(corrupted)


@pytest.mark.parametrize("bad_value", ["1e-100000000", "-1e-100000000", "1e100000000"])
def test_an_extreme_runway_quarters_is_rejected(sec_config, bad_value) -> None:
    facts = _solvency_facts(
        cash=("-1.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["runway"]["quarters"] = bad_value
    _assert_unsupported(corrupted)


def test_the_generators_actual_boundary_outputs_still_pass(sec_config) -> None:
    """Verifies the digit-count bound is derived from precision-64, not
    guessed: the widest monetary value the generator can actually quantize
    (55 leading nines before the decimal point, `_UNDER10_DECIMAL_PRECISION`
    significant digits total) must still be readable."""
    from stanstock.web.views import _parse_canonical_decimal

    widest_integer_part = "9" * 56
    widest_value = f"{widest_integer_part}.00000000"
    assert _parse_canonical_decimal(widest_value, places=8) is not None
    facts = _solvency_facts(cash=(widest_value, INSTANT_DATE))
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["inputs"]["cash_and_equivalents"] == widest_value
    _assert_readable(payload)

    one_digit_too_wide = "9" * 57 + ".00000000"
    assert _parse_canonical_decimal(one_digit_too_wide, places=8) is None


def test_valid_negative_zero_monetary_value_is_readable(sec_config) -> None:
    """`_quantized_text` can genuinely emit "-0.00000000" (a tiny negative
    source value rounding to zero); this must render as an explicit,
    honest negative-zero string, not be rejected."""
    from stanstock.web.views import _parse_canonical_decimal

    assert _parse_canonical_decimal("-0.00000000", places=8) is not None
    facts = _solvency_facts(cash=("-0.000000001", INSTANT_DATE))
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["inputs"]["cash_and_equivalents"] == "-0.00000000"
    _assert_readable(payload)


# ---------------------------------------------------------------------------
# F6: valid withheld opaque metadata must remain readable, never affirmative.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opaque_value",
    [7, ["splits"], {"nested": True}, True, None, 3.14],
    ids=["int", "list", "dict", "bool", "null", "float"],
)
def test_withheld_basis_incompatible_preserves_opaque_metadata_types(
    sec_config, opaque_value: object
) -> None:
    payload = _build_payload(sec_config, price_asset=_price_asset(metadata={}))
    assert payload["liquidity"]["reason"] == LIQUIDITY_BASIS_INCOMPATIBLE
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["basis"]["interval"] = opaque_value
    corrupted["liquidity"]["basis"]["adjustment"] = opaque_value
    panel = _assert_readable(_recompute_hash(corrupted))
    # The rest of a genuinely readable assessment (solvency in particular)
    # must not be hidden merely because the liquidity basis carries opaque,
    # honestly preserved evidence.
    assert panel["solvency"]["state"] in ("assessed", "withheld")
    assert "Split-only price basis is confirmed" not in panel["liquidity"]["volume_basis_caveat"]


def test_a_forged_volume_basis_string_is_still_rejected_even_when_opaque_elsewhere(
    sec_config,
) -> None:
    """`volume_basis` is never observed-copied (always the one fixed
    constant); unlike `interval`/`adjustment`/`return_definition`, a wrong
    value there is never legitimate, opaque or not."""
    payload = _build_payload(sec_config, price_asset=_price_asset(metadata={}))
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["basis"]["volume_basis"] = "totally_forged_basis"
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# Representative authenticated-GET round trips: the same contract through
# the full view/template stack, against a genuinely persisted StockAnalysis.
# ---------------------------------------------------------------------------


def _align_run_target_date(analysis: StockAnalysis, target_date: date) -> None:
    """Make the persisted `AnalysisRun.target_date` match a payload's own claim.

    `_under10_panel` cross-checks a stored assessment's `evaluated_for.
    target_date` against the *actual* analysis run it is attached to (a
    forged/stale target date must never masquerade as this decision's own),
    so an HTTP round-trip test built around the fixed, well-understood
    `TARGET_DATE` test constant needs the persisted run to genuinely carry
    that same date -- `persisted_analysis` otherwise always uses "today".
    """
    analysis.run.target_date = target_date
    analysis.run.save(update_fields=["target_date"])


@pytest.mark.django_db
def test_a_genuine_computed_payload_renders_recorded_over_http(
    sec_config, authenticated_client, persisted_analysis, django_assert_max_num_queries
) -> None:
    payload = _build_payload(sec_config)
    _align_run_target_date(persisted_analysis, TARGET_DATE)
    _make_under_ten(persisted_analysis, assessment=payload)
    before = StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    assert detail.context["under10_panel"]["state"] == "recorded"
    assert StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality == before


@pytest.mark.django_db
def test_a_genuine_negative_runway_payload_renders_recorded_over_http(
    sec_config, authenticated_client, persisted_analysis, django_assert_max_num_queries
) -> None:
    facts = _solvency_facts(
        cash=("-1.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    _align_run_target_date(persisted_analysis, TARGET_DATE)
    _make_under_ten(persisted_analysis, assessment=payload)

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    panel = detail.context["under10_panel"]
    assert panel["state"] == "recorded"
    assert panel["solvency"]["runway"]["quarters"] == "-0.0400"
    content = " ".join(detail.content.decode().split())
    assert "Cash runway - -0.0400 quarters" in content
    assert "FCF is non-negative" not in content


@pytest.mark.django_db
@pytest.mark.parametrize(
    "corrupt",
    [
        "not_applicable_with_negative_fcf",
        "zero_liabilities_with_ratio",
        "partial_duration_tuple",
        "impossible_session_window",
        "extreme_negative_exponent",
        "computed_with_null_anchor",
        "annual_span_outside_window",
    ],
)
def test_every_targeted_corruption_is_unsupported_over_http(
    sec_config,
    authenticated_client,
    persisted_analysis,
    django_assert_max_num_queries,
    corrupt: str,
) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    if corrupt == "not_applicable_with_negative_fcf":
        corrupted["solvency"]["inputs"]["free_cash_flow"] = "-1.00000000"
    elif corrupt == "zero_liabilities_with_ratio":
        corrupted["solvency"]["inputs"]["current_liabilities"] = "0.00000000"
    elif corrupt == "partial_duration_tuple":
        corrupted["solvency"]["periods"]["duration_start"] = None
    elif corrupt == "impossible_session_window":
        corrupted["liquidity"]["first_session"] = "2026-03-01"
        corrupted["liquidity"]["last_session"] = "2026-03-02"
    elif corrupt == "extreme_negative_exponent":
        corrupted["solvency"]["inputs"]["cash_and_equivalents"] = "1e-100000000"
    elif corrupt == "computed_with_null_anchor":
        corrupted["liquidity"]["price_asset"] = None
    elif corrupt == "annual_span_outside_window":
        end = date.fromisoformat(corrupted["solvency"]["periods"]["duration_end"])
        corrupted["solvency"]["periods"]["duration_start"] = end.isoformat()
    _align_run_target_date(persisted_analysis, TARGET_DATE)
    _make_under_ten(persisted_analysis, assessment=corrupted)
    before = StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    panel = detail.context["under10_panel"]
    assert panel["state"] == "unsupported"
    content = " ".join(detail.content.decode().split())
    assert "Assessed -" not in content
    assert "New allocation remains 0%" in content
    assert StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality == before


# ---------------------------------------------------------------------------
# P3: canonical fixed-point strings never carry a multi-digit leading zero
# (Python's own `Decimal.__format__("f")` never emits one, so a stored
# string that does is never genuine).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed",
    [
        "00.00000000",
        "-00.00000000",
        "007.00000000",
        "-007.00000000",
        "01.00000000",
    ],
)
def test_a_leading_zero_monetary_string_is_rejected(sec_config, malformed: str) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["cash_and_equivalents"] = malformed
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    "malformed",
    ["00.0000", "-00.0000", "004.0000"],
)
def test_a_leading_zero_reported_ratio_string_is_rejected(sec_config, malformed: str) -> None:
    payload = _build_payload(sec_config)
    assert payload["solvency"]["inputs"]["current_ratio"] is not None
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["current_ratio"] = malformed
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    "malformed",
    ["00.000000", "-00.000000", "004.250000"],
)
def test_a_leading_zero_reference_close_string_is_rejected(sec_config, malformed: str) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["evaluated_for"]["reference_close"] = malformed
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    "malformed",
    ["00.0000", "-00.0000", "004.0000"],
)
def test_a_leading_zero_runway_quarters_string_is_rejected(sec_config, malformed: str) -> None:
    facts = _solvency_facts(
        cash=("1000.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    assert payload["solvency"]["runway"]["quarters"] == "40.0000"
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["runway"]["quarters"] = malformed
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    ("value", "places"),
    [
        ("0.00000000", 8),
        ("-0.00000000", 8),
        ("10.00000000", 8),
        ("100.00000000", 8),
        ("0.0000", 4),
        ("10.0000", 4),
        ("0.000000", 6),
    ],
)
def test_genuine_zero_and_multi_digit_canonical_forms_still_parse(value: str, places: int) -> None:
    """The tightened leading-zero regex must not also reject legitimate
    single-``0`` and ordinary nonzero-leading forms at every scale."""
    from stanstock.web.views import _parse_canonical_decimal

    assert _parse_canonical_decimal(value, places=places) is not None


# ---------------------------------------------------------------------------
# Root hardening: `assessment_hash` is an explicit, total (never-raising)
# accidental-corruption checksum gate -- a recomputation check, not tamper
# protection, and it must never substitute for any semantic branch check.
# ---------------------------------------------------------------------------


def test_a_genuine_payloads_own_hash_is_accepted(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["assessment_hash"] == under10_assessment_hash(payload)
    _assert_readable(payload)


def test_a_single_flipped_hash_character_is_rejected(sec_config) -> None:
    """The narrowest possible accidental bit-level corruption: everything
    else -- including every semantic branch -- stays completely genuine."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    original = corrupted["assessment_hash"]
    flipped_first_char = "0" if original[0] != "0" else "1"
    corrupted["assessment_hash"] = flipped_first_char + original[1:]
    round_tripped = _round_trip(corrupted)
    assert _is_valid_stub_payload(round_tripped) is False
    panel = _under10_panel(
        _stub_analysis(data_quality={"under10_assessment": round_tripped}),
        current_price_band=None,
    )
    assert panel["state"] == "unsupported"


@pytest.mark.parametrize(
    "malformed_hash",
    [
        "not-hex" + "0" * 56,
        "A" * 64,  # uppercase hex is never what the generator emits
        "0" * 63,  # one character short
        "0" * 65,  # one character too many
        "",
        None,
        1234,
        True,
    ],
)
def test_a_malformed_format_hash_is_rejected(sec_config, malformed_hash: object) -> None:
    """Deliberately bypasses `_assert_unsupported`'s recomputed-hash variant:
    recomputing the hash here would *replace* the very malformed value this
    test exists to prove is refused, rather than corroborate it."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["assessment_hash"] = malformed_hash
    round_tripped = _round_trip(corrupted)
    assert _is_valid_stub_payload(round_tripped) is False
    panel = _under10_panel(
        _stub_analysis(data_quality={"under10_assessment": round_tripped}),
        current_price_band=None,
    )
    assert panel["state"] == "unsupported"


def test_a_missing_hash_key_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    del corrupted["assessment_hash"]
    assert _is_valid_stub_payload(_round_trip(corrupted)) is False


def test_the_hash_gate_alone_never_substitutes_for_semantic_validation(sec_config) -> None:
    """The two-variant contract, made explicit for one representative
    corruption: a stale (pre-mutation) hash is rejected by the checksum
    gate alone, while the *same* corruption with the hash correctly
    recomputed is still rejected -- this time only because the semantic
    branch validators refuse the contradiction itself."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["reasons"] = ["near_term_debt_exceeds_cash"]

    stale = _round_trip(corrupted)
    assert stale["assessment_hash"] == payload["assessment_hash"]
    assert _is_valid_stub_payload(stale) is False

    recomputed = _round_trip(_recompute_hash(corrupted))
    assert recomputed["assessment_hash"] != payload["assessment_hash"]
    assert _is_valid_stub_payload(recomputed) is False


def test_recompute_assessment_hash_is_total_and_never_raises() -> None:
    """Direct unit coverage for the wrapper itself: a value `json.dumps` can
    never serialize must produce ``None``, never an uncaught exception --
    the reader's own checksum recomputation must never be the thing that
    turns a malformed stored payload into an unhandled 500."""
    from stanstock.web.views import _recompute_assessment_hash

    assert _recompute_assessment_hash({"assessment_hash": "0" * 64, "value": object()}) is None
    assert _recompute_assessment_hash({"assessment_hash": "0" * 64, "value": float("nan")}) is None
    assert _recompute_assessment_hash({"assessment_hash": "0" * 64, "value": float("inf")}) is None


def test_an_unknown_top_level_extra_key_is_preserved_when_the_hash_matches(sec_config) -> None:
    """`under10_assessment_hash` covers every top-level key except its own,
    so an extra key is only genuine when it is itself included in the hash
    computation -- and, once it is, this reader must not reject the
    payload merely for carrying a field nothing here reads."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["a_future_schema_field_nothing_here_reads"] = {"anything": ["at", "all"]}
    recomputed = _recompute_hash(corrupted)
    _assert_readable(recomputed)


# ---------------------------------------------------------------------------
# Round-6 F1: derived financial claims must equal their recorded canonical
# operands exactly (precision-64/ROUND_HALF_EVEN), not merely look
# structurally plausible -- near_term_debt vs cash, current_ratio, computed
# runway quarters, and the frozen state/reason classification itself.
# ---------------------------------------------------------------------------


def test_a_forged_near_term_debt_against_cash_is_rejected(sec_config) -> None:
    """The exact F1 counterexample: near_term_debt 150 -> 1500 while cash
    stays 1000 must not survive with the favorable status unchanged --
    `debt_exceeds_cash` is recomputed from these same recorded operands."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    assert payload["solvency"]["inputs"]["cash_and_equivalents"] == "1000.00000000"
    assert payload["solvency"]["inputs"]["near_term_debt"] == "150.00000000"
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["near_term_debt"] = "1500.00000000"
    _assert_unsupported(corrupted)


def test_a_negative_current_ratio_with_a_positive_quotient_is_rejected(sec_config) -> None:
    """The exact F1 counterexample: assets=2000/liabilities=1000 genuinely
    quantizes to a positive ``"2.0000"``; a forged negative ratio can never
    match the recomputed quotient, no tolerance or float conversion."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["inputs"]["current_ratio"] == "2.0000"
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["current_ratio"] = "-2.0000"
    _assert_unsupported(corrupted)


def test_a_forged_runway_magnitude_with_the_same_sign_is_rejected(sec_config) -> None:
    """The exact F1 counterexample: a forged ``"999999.0000"`` shares
    ``"40.0000"``'s sign (so RI-3's sign-only check alone would have missed
    it) but is not the recomputed ``4 * cash / abs(FCF)`` quotient."""
    facts = _solvency_facts(
        cash=("1000.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    assert payload["solvency"]["runway"]["quarters"] == "40.0000"
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["runway"]["quarters"] = "999999.0000"
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    ("operand", "forged_value"),
    [
        # Downward, not upward: `debt_exceeds_cash` (150 > cash) is the only
        # predicate that reads cash at all in this baseline (runway is
        # `not_applicable` here, independent of cash's value) -- inflating
        # cash upward keeps every recorded relationship internally
        # consistent (a *real*, if different, cash figure could equally
        # have produced this same output), so only a *downward* forgery
        # actually contradicts the recorded near_term_debt/status pairing.
        ("cash_and_equivalents", "0.00000000"),
        ("near_term_debt", "9000.00000000"),
        ("current_assets", "1.00000000"),
        ("current_liabilities", "999999.00000000"),
        ("free_cash_flow", "-999999.00000000"),
    ],
)
def test_each_solvency_operand_mutated_independently_is_rejected(
    sec_config, operand: str, forged_value: str
) -> None:
    """F1 point 5: every operand, mutated on its own from a genuine complete
    payload, must be caught -- either by the state/reason recomputation, the
    current-ratio recomputation, or the runway recomputation, whichever
    reads it."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"][operand] = forged_value
    _assert_unsupported(corrupted)


def test_a_forged_status_alone_with_genuine_operands_is_rejected(sec_config) -> None:
    """A genuine no-adverse payload's status forged to adverse, with every
    operand and the (correctly empty) reasons list left untouched."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["status"] = SOLVENCY_ELEVATED_OBLIGATION_RISK
    _assert_unsupported(corrupted)


def test_a_reordered_reason_list_is_rejected(sec_config) -> None:
    """The frozen classification emits reasons in one fixed order (debt,
    assets, free cash flow, runway); the same *set* in a different order is
    not the shape the generator produces."""
    facts = _solvency_facts(
        cash=("0.00000000", INSTANT_DATE),
        current_assets=("500.00000000", INSTANT_DATE),
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    reasons = payload["solvency"]["reasons"]
    assert len(reasons) >= 2
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["reasons"] = list(reversed(reasons))
    _assert_unsupported(corrupted)


def test_every_genuine_grid_case_matches_its_own_recomputed_classification(sec_config) -> None:
    """Vacuity guard for the F1 recomputation itself: every complete-input
    case in the existing 448-case genuine grid must independently satisfy
    the exact recomputed state/reason classification -- not merely round-
    trip as *some* recognized shape."""
    from stanstock.web.views import _recompute_solvency_classification

    checked_complete_cases = 0
    for cash_choice, debt_choice, assets_liabilities_choice, fcf_choice in _SOLVENCY_GRID:
        cash = cash_choice[1]
        short, long = debt_choice[1]
        assets, liab = assets_liabilities_choice[1]
        ocf, capex = fcf_choice[1]
        facts = _solvency_facts(
            cash=cash,
            short_term_debt=short,
            current_long_term_debt=long,
            current_assets=assets,
            current_liabilities=liab,
            operating_cash_flow=ocf,
            capital_expenditure=capex,
        )
        payload = _build_payload(sec_config, facts=facts)
        solvency = payload["solvency"]
        if solvency["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE:
            continue
        checked_complete_cases += 1
        inputs = solvency["inputs"]
        expected_status, expected_reasons = _recompute_solvency_classification(
            cash=Decimal(inputs["cash_and_equivalents"]),
            near_term_debt=Decimal(inputs["near_term_debt"]),
            current_assets=Decimal(inputs["current_assets"]),
            current_liabilities=Decimal(inputs["current_liabilities"]),
            free_cash_flow=Decimal(inputs["free_cash_flow"]),
        )
        assert solvency["status"] == expected_status
        assert solvency["reasons"] == expected_reasons
    assert checked_complete_cases > 0


# ---------------------------------------------------------------------------
# Round-6 F2: solvency instant/duration freshness and liquidity staleness,
# each bound relative to this decision's own target date -- future/stale
# recorded facts must never render as an assessed (favorable or adverse)
# state, but may still be legitimately withheld for audit.
# ---------------------------------------------------------------------------


def _facts_with_shared_instant_date(instant_date: date, **overrides: Any) -> list[FundamentalFact]:
    kwargs: dict[str, Any] = {
        "cash": ("1000.00000000", instant_date),
        "short_term_debt": ("100.00000000", instant_date),
        "current_long_term_debt": ("50.00000000", instant_date),
        "current_assets": ("2000.00000000", instant_date),
        "current_liabilities": ("1000.00000000", instant_date),
    }
    kwargs.update(overrides)
    return _solvency_facts(**kwargs)


@pytest.mark.parametrize("age_days", [199, 200])
def test_a_genuine_instant_date_within_the_freshness_window_is_readable(
    sec_config, age_days: int
) -> None:
    instant_date = TARGET_DATE - timedelta(days=age_days)
    payload = _build_payload(sec_config, facts=_facts_with_shared_instant_date(instant_date))
    assert payload["solvency"]["status"] != SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["periods"]["instant_date"] == instant_date.isoformat()
    _assert_readable(payload)


def test_a_genuine_instant_date_one_day_past_the_freshness_window_is_insufficient(
    sec_config,
) -> None:
    instant_date = TARGET_DATE - timedelta(days=201)
    payload = _build_payload(sec_config, facts=_facts_with_shared_instant_date(instant_date))
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["inputs"]["cash_and_equivalents"] is None
    # The stale shared instant date is still preserved for audit even though
    # every operand behind it is withheld.
    assert payload["solvency"]["periods"]["instant_date"] == instant_date.isoformat()
    _assert_readable(payload)


def test_a_genuine_future_instant_date_is_insufficient(sec_config) -> None:
    instant_date = TARGET_DATE + timedelta(days=1)
    payload = _build_payload(sec_config, facts=_facts_with_shared_instant_date(instant_date))
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["inputs"]["cash_and_equivalents"] is None
    _assert_readable(payload)


@pytest.mark.parametrize("age_days", [201, -1])
def test_a_forged_stale_or_future_instant_date_with_operands_kept_present_is_rejected(
    sec_config, age_days: int
) -> None:
    """The exact F2 counterexample: a genuine complete payload's shared
    instant date forged stale/future while every operand it backs stays
    present is a contradiction `_validate_instant_fact`'s own per-fact
    freshness rule cannot produce."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] != SOLVENCY_INSUFFICIENT_EVIDENCE
    corrupted = json.loads(json.dumps(payload))
    forged_instant_date = TARGET_DATE - timedelta(days=age_days)
    corrupted["solvency"]["periods"]["instant_date"] = forged_instant_date.isoformat()
    _assert_unsupported(corrupted)


@pytest.mark.parametrize("age_days", [199, 200])
def test_a_genuine_duration_end_within_the_freshness_window_is_readable(
    sec_config, age_days: int
) -> None:
    duration_end = TARGET_DATE - timedelta(days=age_days)
    annual_period = (duration_end - timedelta(days=364), duration_end)
    payload = _build_payload(sec_config, facts=_solvency_facts(annual_period=annual_period))
    assert payload["solvency"]["inputs"]["free_cash_flow"] is not None
    assert payload["solvency"]["periods"]["duration_end"] == duration_end.isoformat()
    _assert_readable(payload)


@pytest.mark.parametrize("age_days", [201, -1])
def test_a_genuine_stale_or_future_duration_end_withholds_free_cash_flow(
    sec_config, age_days: int
) -> None:
    duration_end = TARGET_DATE - timedelta(days=age_days)
    annual_period = (duration_end - timedelta(days=364), duration_end)
    payload = _build_payload(sec_config, facts=_solvency_facts(annual_period=annual_period))
    assert payload["solvency"]["inputs"]["free_cash_flow"] is None
    assert payload["solvency"]["runway"]["status"] == RUNWAY_WITHHELD
    # The stale/future duration window is still preserved for audit.
    assert payload["solvency"]["periods"]["duration_end"] == duration_end.isoformat()
    _assert_readable(payload)


@pytest.mark.parametrize("age_days", [201, -1])
def test_a_forged_stale_or_future_duration_end_with_fcf_kept_usable_is_rejected(
    sec_config, age_days: int
) -> None:
    """The F2 sibling counterexample for the flow window: forging
    ``duration_end`` stale/future while ``free_cash_flow`` and the runway
    derived from it stay a *usable* claim is a contradiction."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["inputs"]["free_cash_flow"] is not None
    corrupted = json.loads(json.dumps(payload))
    forged_duration_end = TARGET_DATE - timedelta(days=age_days)
    corrupted["solvency"]["periods"]["duration_end"] = forged_duration_end.isoformat()
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# F2 (span): a populated annual/TTM duration triple must independently span
# the same inclusive `MIN_ANNUAL_DAYS`-`MAX_ANNUAL_DAYS` (350-380 day) window
# `stanstock.data.sec_fundamentals` enforces for both bases. Only
# ``duration_start`` is forged in the "outside the window" cases below --
# ``duration_end`` stays the genuine, freshness-eligible value the payload
# already carries -- so each case isolates the span defect from the
# already-covered F2 staleness checks above.
# ---------------------------------------------------------------------------

_FORGED_SPAN_DAYS = pytest.mark.parametrize(
    "span_days",
    [1, MIN_ANNUAL_DAYS - 1, MAX_ANNUAL_DAYS + 1],
    ids=["one-day", "349-day", "381-day"],
)
_GENUINE_BOUNDARY_SPAN_DAYS = pytest.mark.parametrize(
    "span_days",
    [MIN_ANNUAL_DAYS, MAX_ANNUAL_DAYS],
    ids=["350-day", "380-day"],
)


@_FORGED_SPAN_DAYS
def test_a_forged_annual_duration_span_outside_350_380_days_is_rejected(
    sec_config, span_days: int
) -> None:
    """Complete solvency, annual basis: a checksum-recomputed one-day,
    349-day, or 381-day flow window can never be a genuine annual
    construction, which `_annual_series` only ever admits inside
    350-380 days."""
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    assert payload["solvency"]["periods"]["duration_basis"] == "annual"
    corrupted = json.loads(json.dumps(payload))
    end = date.fromisoformat(corrupted["solvency"]["periods"]["duration_end"])
    forged_start = end - timedelta(days=span_days - 1)
    corrupted["solvency"]["periods"]["duration_start"] = forged_start.isoformat()
    _assert_unsupported(corrupted)


@_FORGED_SPAN_DAYS
def test_a_forged_annual_duration_span_outside_350_380_days_is_rejected_with_insufficient_evidence(
    sec_config, span_days: int
) -> None:
    """The same span forgery against a partial `insufficient_evidence`
    payload that still reports a genuine, usable, numeric runway (missing
    debt components are the only withheld operand): the flow-window span
    invariant applies independently of the overall solvency completeness."""
    facts = _solvency_facts(
        cash=("1000.00000000", INSTANT_DATE),
        short_term_debt=None,
        current_long_term_debt=None,
        operating_cash_flow="0.00000000",
        capital_expenditure="100.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    assert payload["solvency"]["runway"]["quarters"] == "40.0000"
    assert payload["solvency"]["periods"]["duration_basis"] == "annual"
    corrupted = json.loads(json.dumps(payload))
    end = date.fromisoformat(corrupted["solvency"]["periods"]["duration_end"])
    forged_start = end - timedelta(days=span_days - 1)
    corrupted["solvency"]["periods"]["duration_start"] = forged_start.isoformat()
    _assert_unsupported(corrupted)


@_FORGED_SPAN_DAYS
def test_a_forged_ttm_duration_span_outside_350_380_days_is_rejected(
    sec_config, span_days: int
) -> None:
    """Complete solvency, TTM basis: the same forged span is rejected when
    the genuine payload's flow window came from four contiguous quarters
    instead of one annual fact."""
    payload = _build_payload(sec_config, facts=_ttm_solvency_facts())
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    assert payload["solvency"]["periods"]["duration_basis"] == "ttm"
    corrupted = json.loads(json.dumps(payload))
    end = date.fromisoformat(corrupted["solvency"]["periods"]["duration_end"])
    forged_start = end - timedelta(days=span_days - 1)
    corrupted["solvency"]["periods"]["duration_start"] = forged_start.isoformat()
    _assert_unsupported(corrupted)


@_FORGED_SPAN_DAYS
def test_a_forged_ttm_duration_span_outside_350_380_days_is_rejected_with_insufficient_evidence(
    sec_config, span_days: int
) -> None:
    """The TTM sibling of the annual `insufficient_evidence`-with-numeric-
    runway case above: missing debt components leave the overall status
    `insufficient_evidence` while cash/free_cash_flow (built from four
    contiguous quarters) still back a usable, computed runway."""
    facts = _ttm_solvency_facts(
        cash=("1000.00000000", INSTANT_DATE),
        short_term_debt=None,
        current_long_term_debt=None,
        quarter_operating_cash_flow="0.00000000",
        quarter_capital_expenditure="25.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    assert payload["solvency"]["runway"]["quarters"] == "40.0000"
    assert payload["solvency"]["periods"]["duration_basis"] == "ttm"
    corrupted = json.loads(json.dumps(payload))
    end = date.fromisoformat(corrupted["solvency"]["periods"]["duration_end"])
    forged_start = end - timedelta(days=span_days - 1)
    corrupted["solvency"]["periods"]["duration_start"] = forged_start.isoformat()
    _assert_unsupported(corrupted)


@_GENUINE_BOUNDARY_SPAN_DAYS
def test_a_genuine_annual_duration_span_at_the_350_380_day_boundary_is_readable(
    sec_config, span_days: int
) -> None:
    """The 350/380-day boundary itself stays accepted -- only a span
    strictly outside it is refused."""
    end = date(2025, 12, 31)
    start = end - timedelta(days=span_days - 1)
    facts = _solvency_facts(annual_period=(start, end))
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["periods"]["duration_basis"] == "annual"
    assert payload["solvency"]["periods"]["duration_start"] == start.isoformat()
    assert payload["solvency"]["periods"]["duration_end"] == end.isoformat()
    _assert_readable(payload)


@pytest.mark.parametrize(
    ("span_days", "quarter_lengths"),
    [
        (MIN_ANNUAL_DAYS, (87, 88, 87, 88)),
        (MAX_ANNUAL_DAYS, (95, 95, 95, 95)),
    ],
    ids=["350-day", "380-day"],
)
def test_a_genuine_ttm_duration_span_at_the_350_380_day_boundary_is_readable(
    sec_config, span_days: int, quarter_lengths: tuple[int, int, int, int]
) -> None:
    """The TTM sibling of the boundary test above, built from four genuine
    contiguous quarters (each independently inside `MIN_QUARTER_DAYS`-
    `MAX_QUARTER_DAYS`) summing to exactly 350/380 days."""
    end = date(2025, 12, 31)
    start = end - timedelta(days=span_days - 1)
    facts = _ttm_solvency_facts(quarter_end=end, quarter_lengths=quarter_lengths)
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["periods"]["duration_basis"] == "ttm"
    assert payload["solvency"]["periods"]["duration_start"] == start.isoformat()
    assert payload["solvency"]["periods"]["duration_end"] == end.isoformat()
    _assert_readable(payload)


@pytest.mark.parametrize("age_days", [6, 7])
def test_a_genuine_liquidity_last_session_within_the_staleness_window_is_readable(
    sec_config, age_days: int
) -> None:
    payload = _build_payload(
        sec_config, price_frame=_price_frame(last_session=TARGET_DATE - timedelta(days=age_days))
    )
    assert payload["liquidity"]["status"] == LIQUIDITY_COMPUTED
    _assert_readable(payload)


def test_a_genuine_liquidity_last_session_eight_days_stale_is_withheld(sec_config) -> None:
    payload = _build_payload(
        sec_config, price_frame=_price_frame(last_session=TARGET_DATE - timedelta(days=8))
    )
    assert payload["liquidity"]["status"] == LIQUIDITY_WITHHELD
    assert payload["liquidity"]["reason"] == LIQUIDITY_STALE_PRICE_EVIDENCE
    _assert_readable(payload)


def test_a_forged_eight_day_stale_liquidity_relabeled_computed_is_rejected(sec_config) -> None:
    """The exact F2 counterexample: a genuine 8-day-stale withheld result
    relabeled `computed` (value restored, reason cleared) must never pass
    the 7-day staleness bound, even with the hash recomputed."""
    payload = _build_payload(
        sec_config, price_frame=_price_frame(last_session=TARGET_DATE - timedelta(days=8))
    )
    assert payload["liquidity"]["reason"] == LIQUIDITY_STALE_PRICE_EVIDENCE
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["status"] = LIQUIDITY_COMPUTED
    corrupted["liquidity"]["value"] = 4_000_000.0
    corrupted["liquidity"]["reason"] = None
    _assert_unsupported(corrupted)


def test_a_genuine_seven_day_stale_computed_relabeled_stale_reason_is_rejected(sec_config) -> None:
    """The mirror-image relabeling: a genuine, exactly-7-day-old `computed`
    result relabeled `withheld`/`stale_price_evidence` must also fail --
    it is not actually stale by the exact margin the reason claims."""
    payload = _build_payload(
        sec_config, price_frame=_price_frame(last_session=TARGET_DATE - timedelta(days=7))
    )
    assert payload["liquidity"]["status"] == LIQUIDITY_COMPUTED
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["status"] = LIQUIDITY_WITHHELD
    corrupted["liquidity"]["value"] = None
    corrupted["liquidity"]["reason"] = LIQUIDITY_STALE_PRICE_EVIDENCE
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# Round-6 F3: `evaluated_for.reference_close`/`currency` must bind exactly
# to the already-loaded, immutable decision-run context -- never a mutable
# latest-market read, and never a transplant from a different listing/run.
# ---------------------------------------------------------------------------


def test_a_genuine_payload_is_readable_when_bound_to_its_own_decision_close(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["evaluated_for"]["reference_close"] == "4.250000"
    _assert_readable(payload)


@pytest.mark.parametrize(
    "forged_close",
    ["-4.250000", "9999.250000", "3.500000", "0.000000", "10.000000"],
)
def test_a_reference_close_not_bound_to_the_decision_run_is_rejected(
    sec_config, forged_close: str
) -> None:
    """The exact F3 counterexample: negative, far outside the band, and a
    *different*-but-still-under-$10 close (a transplant from another
    listing/run entirely) must all be refused -- the binding is exact
    equality to the loaded decision close, never a range check alone."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["evaluated_for"]["reference_close"] = forged_close
    _assert_unsupported(corrupted)


def test_a_currency_transplanted_from_a_different_listing_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["evaluated_for"]["currency"] = "EUR"
    _assert_unsupported(corrupted)


@pytest.mark.parametrize(
    "out_of_band_close", [Decimal("-4.250000"), Decimal("0.000000"), Decimal("10.000000")]
)
def test_a_decision_close_outside_the_under10_band_is_rejected(
    sec_config, out_of_band_close: Decimal
) -> None:
    """Even with a byte-matching recorded ``reference_close`` and a matching
    hash, a decision run whose own bound close is not genuinely inside the
    Under-$10 band (as the caller's own band classification already
    determined) can never render a recorded Under-$10 assessment."""
    payload = _build_payload(sec_config, reference_close=out_of_band_close)
    round_tripped = _round_trip(payload)
    assert (
        _is_valid_stub_payload(
            round_tripped,
            expected_reference_close=out_of_band_close,
            expected_reference_close_in_band=False,
        )
        is False
    )


def test_a_decision_close_genuinely_in_band_but_not_usd_currency_is_rejected(sec_config) -> None:
    """`classify_price_band` itself refuses a non-USD currency (returns
    ``None``), so a listing recorded in another currency can never satisfy
    `expected_reference_close_in_band` even if the raw number looks
    plausible -- reusing the caller's own band flag, rather than a bare
    ``0 < x < 10`` check, is what closes this for free."""
    payload = _build_payload(sec_config)
    round_tripped = _round_trip(payload)
    assert _is_valid_stub_payload(round_tripped, expected_reference_close_in_band=False) is False


# ---------------------------------------------------------------------------
# NB-1: an impossible complete `insufficient_evidence` must never pass --
# `_build_solvency` only reaches `insufficient_evidence` when at least one
# of its five candidate operands is unusable; a complete set always reaches
# `_classify_solvency` (adverse/elevated/no-adverse) instead.
# ---------------------------------------------------------------------------


def test_a_complete_no_adverse_payload_relabeled_insufficient_is_rejected(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["solvency"]["status"] == SOLVENCY_NO_ADVERSE_EVIDENCE
    assert all(
        payload["solvency"]["inputs"][key] is not None
        for key in (
            "cash_and_equivalents",
            "near_term_debt",
            "current_assets",
            "current_liabilities",
            "free_cash_flow",
        )
    )
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["status"] = SOLVENCY_INSUFFICIENT_EVIDENCE
    corrupted["solvency"]["reasons"] = ["stale_metric"]
    _assert_unsupported(corrupted)


def test_a_complete_adverse_payload_relabeled_insufficient_is_rejected(sec_config) -> None:
    facts = _solvency_facts(
        cash=("0.00000000", INSTANT_DATE), current_assets=("500.00000000", INSTANT_DATE)
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["status"] = SOLVENCY_INSUFFICIENT_EVIDENCE
    corrupted["solvency"]["reasons"] = ["near_term_debt_components_missing"]
    _assert_unsupported(corrupted)


def test_a_genuine_partial_insufficient_evidence_case_remains_readable(sec_config) -> None:
    """Vacuity guard for NB-1: a genuinely partial case (one operand
    missing, independent of the others) must not be swept up by the new
    complete-set check."""
    facts = _solvency_facts(current_liabilities=None)
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["inputs"]["current_liabilities"] is None
    _assert_readable(payload)


# ---------------------------------------------------------------------------
# NB-2: a non-dict `analysis.data_quality` must never raise -- it is a
# schema-less `JSONField`, so a malformed row can carry any JSON value.
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_data_quality",
    # ``None`` is deliberately absent here: `data_quality` is a `NOT NULL`
    # column (Django's `JSONField(default=dict)` default), so it can never
    # actually reach a persisted row -- `.save()` itself raises
    # `IntegrityError` first. It is still covered directly, below, exactly
    # like the existing NaN/Infinity liquidity precedent.
    [5, ["under10_assessment"], "under10_assessment", True],
    ids=["int", "list", "string", "bool"],
)
def test_a_non_dict_data_quality_never_raises_over_http(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
    malformed_data_quality: object,
) -> None:
    persisted_analysis.data_quality = malformed_data_quality
    persisted_analysis.save(update_fields=["data_quality"])
    before = StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    panel = detail.context["under10_panel"]
    assert panel is None or panel["state"] == "unsupported"
    assert StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality == before


def test_a_non_dict_data_quality_never_raises_directly() -> None:
    """Direct validator-path coverage, independent of the HTTP round trip."""
    from stanstock.web.views import _under10_panel

    for malformed in (None, 5, ["under10_assessment"], "under10_assessment", True):
        analysis = SimpleNamespace(
            data_quality=malformed,
            current_price=_STUB_REFERENCE_CLOSE,
            run=SimpleNamespace(target_date=TARGET_DATE, data_cutoff=DATA_CUTOFF),
            listing=SimpleNamespace(currency=_STUB_CURRENCY),
        )
        panel = _under10_panel(analysis, current_price_band=None)
        assert panel is None or panel["state"] == "unsupported"


# ---------------------------------------------------------------------------
# NB-3: the listing's own currency is compared case-insensitively (matching
# `classify_price_band`'s own `.upper()` normalization and this module's
# existing `listing__currency__iexact` filters), while the *recorded*
# payload currency stays held to the generator's exact canonical constant.
# ---------------------------------------------------------------------------


def test_a_genuine_payload_is_readable_against_a_lowercase_usd_listing(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["evaluated_for"]["currency"] == "USD"
    round_tripped = _round_trip(payload)
    assert (
        _is_valid_stub_payload(
            round_tripped, expected_currency="usd", expected_reference_close_in_band=True
        )
        is True
    )


def test_a_non_usd_listing_currency_is_still_rejected_regardless_of_case(sec_config) -> None:
    payload = _build_payload(sec_config)
    round_tripped = _round_trip(payload)
    for candidate_currency in ("eur", "EUR", "Eur"):
        assert _is_valid_stub_payload(round_tripped, expected_currency=candidate_currency) is False


def test_a_recorded_non_canonical_currency_is_still_rejected_even_against_a_matching_case(
    sec_config,
) -> None:
    """The recorded payload's own currency is never weakened to a mere
    case-insensitive match against the expected listing currency: it must
    stay the generator's exact canonical constant, even in the one case
    that could otherwise slip through a same-case-but-non-canonical
    comparison (both recorded and expected happen to share a non-USD,
    non-canonical value)."""
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["evaluated_for"]["currency"] = "EUR"
    recomputed = _recompute_hash(corrupted)
    round_tripped = _round_trip(recomputed)
    assert (
        _is_valid_stub_payload(
            round_tripped, expected_currency="eur", expected_reference_close_in_band=True
        )
        is False
    )


# ---------------------------------------------------------------------------
# NB-4: `_build_liquidity` resolves staleness before ever inspecting
# `median_dollar_volume`'s own status/reason, so every recognized reason
# with a real reported window -- other than `future_price_session` itself,
# and `stale_price_evidence`'s own inverted bound -- can only genuinely be
# reached when that window's last session is not stale.
# ---------------------------------------------------------------------------


def test_liquidity_invalid_volume_branch_is_readable(sec_config) -> None:
    payload = _build_payload(sec_config, price_frame=_price_frame(volume=-1.0))
    assert payload["liquidity"]["reason"] == "invalid_volume_values"
    _assert_readable(payload)


@pytest.mark.parametrize(
    "price_frame_kwargs",
    [{"close": 0.0}, {"volume": -1.0}, {"sessions": 100}, {"close": 9e307, "volume": 1.0}],
    ids=["invalid_close", "invalid_volume", "insufficient_sessions", "nonfinite_median"],
)
def test_staleness_takes_precedence_over_every_other_windowed_reason(
    sec_config, price_frame_kwargs: dict[str, Any]
) -> None:
    """Confirms the real generator's own precedence (staleness is resolved
    *before* `median_dollar_volume`'s own reason is ever consulted): an
    otherwise-reason-worthy window that is *also* 8 days stale is reported
    as `stale_price_evidence`, never the other reason -- exactly why every
    other windowed reason's own genuine window can never itself be stale."""
    payload = _build_payload(
        sec_config,
        price_frame=_price_frame(
            last_session=TARGET_DATE - timedelta(days=8), **price_frame_kwargs
        ),
    )
    assert payload["liquidity"]["reason"] == LIQUIDITY_STALE_PRICE_EVIDENCE
    _assert_readable(payload)


@pytest.mark.parametrize(
    "target_reason",
    [
        "invalid_close_values",
        "invalid_volume_values",
        "invalid_rows_dropped_during_preparation",
        "nonfinite_median",
    ],
)
def test_a_genuine_stale_window_relabeled_to_another_windowed_reason_is_rejected(
    sec_config, target_reason: str
) -> None:
    """Starting from a genuine `stale_price_evidence` result (last session
    8 days old, sessions_used already 252 since the default price frame is
    a full window), relabeling it to any other windowed reason -- keeping
    its stale window unchanged -- must still fail: none of those reasons
    can genuinely carry a stale window either."""
    payload = _build_payload(
        sec_config, price_frame=_price_frame(last_session=TARGET_DATE - timedelta(days=8))
    )
    assert payload["liquidity"]["reason"] == LIQUIDITY_STALE_PRICE_EVIDENCE
    assert payload["liquidity"]["sessions_used"] == 252
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["reason"] = target_reason
    _assert_unsupported(corrupted)


def test_a_genuine_insufficient_sessions_nonempty_window_survives_eight_day_staleness_check(
    sec_config,
) -> None:
    """Vacuity guard: a genuine, non-stale `insufficient_sessions` window
    (last session on target date) remains readable -- the new NB-4 bound
    does not reject a genuinely fresh partial window."""
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=100))
    assert payload["liquidity"]["reason"] == "insufficient_sessions"
    assert payload["liquidity"]["sessions_used"] == 100
    _assert_readable(payload)


def test_a_genuine_insufficient_sessions_window_staled_by_mutation_is_rejected(sec_config) -> None:
    """The mutation-side counterexample: a genuine, fresh, non-empty
    `insufficient_sessions` window mutated to be 8 days stale (reason kept
    unchanged) is a contradiction -- a genuinely stale window here would
    have been reported as `stale_price_evidence` instead (as the
    precedence test above already proves)."""
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=100))
    assert payload["liquidity"]["reason"] == "insufficient_sessions"
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    stale_last_session = TARGET_DATE - timedelta(days=8)
    corrupted["liquidity"]["last_session"] = stale_last_session.isoformat()
    corrupted["liquidity"]["first_session"] = (stale_last_session - timedelta(days=99)).isoformat()
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# NB-5: the converse ratio requirement -- both current_assets and a
# strictly positive current_liabilities present always yields a computed
# current_ratio, independent of any unrelated missing debt/FCF input.
# ---------------------------------------------------------------------------


def test_both_ratio_operands_present_requires_a_computed_ratio(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["solvency"]["inputs"]["current_ratio"] == "2.0000"
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["current_ratio"] = None
    _assert_unsupported(corrupted)


def test_both_ratio_operands_present_with_unrelated_missing_debt_still_requires_a_ratio(
    sec_config,
) -> None:
    """The exact NB-5 counterexample: a partial `insufficient_evidence`
    assessment (missing debt components, unrelated to the ratio) still
    genuinely computes the ratio -- nulling it out is still a
    contradiction, independent of the overall status."""
    facts = _solvency_facts(short_term_debt=None, current_long_term_debt=None)
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["inputs"]["current_ratio"] == "2.0000"
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["inputs"]["current_ratio"] = None
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# NB-7: the reader's own canonical decimal scales must match the
# generator's own published policy document exactly -- a direct parity
# regression, so drift fails clearly rather than only through the large
# corruption matrix.
# ---------------------------------------------------------------------------


def test_reader_decimal_scales_match_the_policy_document_exactly(sec_config) -> None:
    from stanstock.research.under10 import under10_policy_document
    from stanstock.web.views import (
        _MONETARY_DECIMAL_PLACES,
        _MONETARY_QUANTUM,
        _REFERENCE_CLOSE_DECIMAL_PLACES,
        _REFERENCE_CLOSE_QUANTUM,
        _REPORTED_DECIMAL_PLACES,
        _REPORTED_QUANTUM,
        _UNDER10_DECIMAL_PRECISION,
    )

    serialization = under10_policy_document(sec_config)["serialization"]
    assert serialization["monetary_decimal_places"] == _MONETARY_DECIMAL_PLACES
    assert serialization["reported_decimal_places"] == _REPORTED_DECIMAL_PLACES
    assert serialization["reference_close_decimal_places"] == _REFERENCE_CLOSE_DECIMAL_PLACES
    assert serialization["decimal_precision"] == _UNDER10_DECIMAL_PRECISION
    assert serialization["rounding"] == "ROUND_HALF_EVEN"
    # The quantization exponents themselves must have exactly as many
    # fractional digits as the digit-count constants above claim.
    assert -_MONETARY_QUANTUM.as_tuple().exponent == _MONETARY_DECIMAL_PLACES
    assert -_REPORTED_QUANTUM.as_tuple().exponent == _REPORTED_DECIMAL_PLACES
    assert -_REFERENCE_CLOSE_QUANTUM.as_tuple().exponent == _REFERENCE_CLOSE_DECIMAL_PLACES


# ---------------------------------------------------------------------------
# C1: assessment identity is bound to the parent analysis's own recorded
# price asset -- both the UUID (`data_quality["price_source"]["asset_id"]`)
# *and* content checksum, cross-derived from `data_quality["source_assets"]`
# -- and must be evaluated under *exactly* the parent `AnalysisRun.
# data_cutoff`, never merely at or before it. A matching date/reference
# close/currency alone is not sufficient identity -- two genuine, unrelated
# candidates can share all three on the same day.
# ---------------------------------------------------------------------------


def _other_price_asset() -> DataAsset:
    """A second, genuinely different `DataAsset` (distinct id/sha256) from
    the shared `_price_asset()` fixture -- simulates a different analysis's
    own recorded price evidence for the C1 transplant regressions."""
    return _asset(
        "twelve-data-price-transplant",
        provider="twelve_data",
        kind="price_history",
        metadata={
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "currency": "USD",
        },
    )


def test_c1_genuine_payload_is_rejected_against_a_different_analysis_price_asset(
    sec_config,
) -> None:
    """The exact C1 reproduction: a correctly checksummed, otherwise
    genuine payload naming one price asset must never be accepted by a
    reader bound to a *different* analysis's own recorded price-asset
    UUID+checksum -- even though every other field (date, reference
    close, currency) matches."""
    payload = _build_payload(sec_config)
    round_tripped = _round_trip(payload)
    assert _is_valid_stub_payload(round_tripped) is True
    other_asset = _other_price_asset()
    other_reference = (str(other_asset.id), other_asset.sha256)
    assert other_reference[0] != round_tripped["liquidity"]["price_asset"]["id"]
    assert (
        _is_valid_stub_payload(round_tripped, expected_price_asset_reference=other_reference)
        is False
    )
    panel = _under10_panel(
        _stub_analysis(
            data_quality={"under10_assessment": round_tripped},
            price_asset_reference=other_reference,
        ),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "unsupported"


def test_c1_two_same_date_same_price_analyses_with_different_evidence_cannot_swap_assessments(
    sec_config,
) -> None:
    """The full C1 scenario: two genuine analyses share the exact same
    decision date/reference close/currency -- the only identity the
    payload's own `evaluated_for` carries -- but differ in their own
    recorded price evidence and solvency facts. Each analysis's own
    genuine assessment is readable against itself, but neither survives
    being transplanted onto the other."""
    asset_a = _price_asset()
    asset_b = _other_price_asset()
    reference_a: tuple[str, str] = (str(asset_a.id), asset_a.sha256)
    reference_b: tuple[str, str] = (str(asset_b.id), asset_b.sha256)
    payload_a = _round_trip(_build_payload(sec_config, price_asset=asset_a))
    payload_b = _round_trip(
        _build_payload(
            sec_config,
            price_asset=asset_b,
            facts=_solvency_facts(cash=("2000.00000000", INSTANT_DATE)),
        )
    )
    assert (
        payload_a["liquidity"]["price_asset"]["id"] != payload_b["liquidity"]["price_asset"]["id"]
    )
    assert (
        payload_a["solvency"]["inputs"]["cash_and_equivalents"]
        != payload_b["solvency"]["inputs"]["cash_and_equivalents"]
    )

    def _panel_for(payload: dict[str, Any], *, reference: tuple[str, str]) -> dict[str, Any]:
        panel = _under10_panel(
            _stub_analysis(
                data_quality={"under10_assessment": payload}, price_asset_reference=reference
            ),
            current_price_band=None,
        )
        assert panel is not None
        return panel

    assert _panel_for(payload_a, reference=reference_a)["state"] == "recorded"
    assert _panel_for(payload_b, reference=reference_b)["state"] == "recorded"
    # Transplants in both directions must be refused.
    assert _panel_for(payload_a, reference=reference_b)["state"] == "unsupported"
    assert _panel_for(payload_b, reference=reference_a)["state"] == "unsupported"


def test_c1_same_uuid_but_changed_payload_checksum_is_rejected(sec_config) -> None:
    """The exact checksum-binding gap: keeping the correct price-asset UUID
    but claiming different asset *content* (a mutated ``sha256`` on the
    payload's own ``liquidity.price_asset``, with `assessment_hash`
    recomputed to match) must never be accepted -- the parent analysis's
    own recorded checksum for that UUID is authoritative, not the
    payload's unverified claim about itself.

    This cannot go through the shared `_assert_unsupported` (which derives
    its "expected" reference from the very payload under test, so it
    would trivially self-match any sha256 the mutation picks): the parent
    analysis's own genuine anchor must stay fixed while only the payload's
    own claim changes."""
    asset = _price_asset()
    payload = _build_payload(sec_config, price_asset=asset)
    genuine_reference = (str(asset.id), asset.sha256)
    round_tripped = _round_trip(payload)
    assert (
        _is_valid_stub_payload(round_tripped, expected_price_asset_reference=genuine_reference)
        is True
    )
    genuine_sha256 = round_tripped["liquidity"]["price_asset"]["sha256"]
    forged_sha256 = ("0" if genuine_sha256[0] != "0" else "1") + genuine_sha256[1:]
    assert forged_sha256 != genuine_sha256
    forged_stale = json.loads(json.dumps(round_tripped))
    forged_stale["liquidity"]["price_asset"]["sha256"] = forged_sha256
    for candidate in (forged_stale, _recompute_hash(forged_stale)):
        forged = _round_trip(candidate)
        assert (
            _is_valid_stub_payload(forged, expected_price_asset_reference=genuine_reference)
            is False
        )
        panel = _under10_panel(
            _stub_analysis(
                data_quality={"under10_assessment": forged},
                price_asset_reference=genuine_reference,
            ),
            current_price_band=None,
        )
        assert panel is not None
        assert panel["state"] == "unsupported"


def test_c1_changed_parent_source_asset_checksum_is_rejected(sec_config) -> None:
    """The other side of the same gap: the payload's own recorded price
    asset is genuinely self-consistent, but the *parent analysis's* own
    ``data_quality["source_assets"]`` entry for that exact UUID carries a
    different checksum than what the payload claims -- e.g. the asset was
    later found to have different content under the same id. This must
    never be accepted either, since the parent's own recorded evidence
    checksum is what is authoritative."""
    asset = _price_asset()
    payload = _round_trip(_build_payload(sec_config, price_asset=asset))
    assert _is_valid_stub_payload(payload) is True
    genuine_sha256 = payload["liquidity"]["price_asset"]["sha256"]
    forged_parent_sha256 = ("0" if genuine_sha256[0] != "0" else "1") + genuine_sha256[1:]
    assert forged_parent_sha256 != genuine_sha256
    forged_parent_reference = (str(asset.id), forged_parent_sha256)
    assert (
        _is_valid_stub_payload(payload, expected_price_asset_reference=forged_parent_reference)
        is False
    )
    panel = _under10_panel(
        _stub_analysis(
            data_quality={"under10_assessment": payload},
            price_asset_reference=forged_parent_reference,
        ),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "unsupported"


def test_c1_missing_parent_source_asset_entry_is_rejected(sec_config) -> None:
    """A parent analysis whose own ``data_quality`` establishes no anchor
    at all for the payload's claimed price asset (e.g. ``source_assets``
    is empty, or the payload's own ``price_source`` is simply absent) must
    never render a favorable state -- there is nothing here to bind to."""
    payload = _build_payload(sec_config)
    round_tripped = _round_trip(payload)
    assert _is_valid_stub_payload(round_tripped, expected_price_asset_reference=None) is False
    panel = _under10_panel(
        _stub_analysis(
            data_quality={"under10_assessment": round_tripped}, price_asset_reference=None
        ),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "unsupported"


def test_c1_ambiguous_duplicate_parent_source_asset_entries_is_rejected(sec_config) -> None:
    """A malformed/forged parent `data_quality` carrying *two* different
    `source_assets` entries for the same id (a shape `_dedupe_assets`
    never produces for genuine data) can never unambiguously establish an
    anchor either -- this must be refused exactly like a missing one,
    never resolved by picking either candidate."""
    asset = _price_asset()
    payload = _round_trip(_build_payload(sec_config, price_asset=asset))
    assert _is_valid_stub_payload(payload) is True
    genuine_sha256 = payload["liquidity"]["price_asset"]["sha256"]
    other_sha256 = ("0" if genuine_sha256[0] != "0" else "1") + genuine_sha256[1:]
    ambiguous_data_quality = {
        "under10_assessment": payload,
        "price_source": {"asset_id": str(asset.id)},
        "source_assets": [
            {"id": str(asset.id), "sha256": genuine_sha256},
            {"id": str(asset.id), "sha256": other_sha256},
        ],
    }
    panel = _under10_panel(
        _stub_analysis(data_quality=ambiguous_data_quality),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "unsupported"


def test_c1_genuine_exact_price_asset_reference_is_readable(sec_config) -> None:
    """Vacuity guard: a genuine payload's own price-asset UUID+checksum,
    matched exactly by the parent analysis's own recorded anchor, really
    does render -- proving the corruption/mismatch tests above reject a
    genuine *difference*, not merely the presence of any binding check."""
    asset = _price_asset()
    payload = _build_payload(sec_config, price_asset=asset)
    reference = (str(asset.id), asset.sha256)
    round_tripped = _round_trip(payload)
    assert round_tripped["liquidity"]["price_asset"] == {"id": reference[0], "sha256": reference[1]}
    assert _is_valid_stub_payload(round_tripped, expected_price_asset_reference=reference) is True
    panel = _under10_panel(
        _stub_analysis(
            data_quality={"under10_assessment": round_tripped}, price_asset_reference=reference
        ),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "recorded"


def test_c1_assessment_evaluated_after_the_parent_run_cutoff_is_rejected(sec_config) -> None:
    """The other half of C1: even with identical price-asset identity, an
    assessment recorded with a `data_cutoff` *after* the parent analysis
    run's own cutoff (e.g. transplanted from a later re-run, or forged
    forward) must never be trusted -- only exact equality with the run's
    own cutoff is legitimate."""
    payload = _build_payload(sec_config)
    round_tripped = _round_trip(payload)
    assert _is_valid_stub_payload(round_tripped, expected_data_cutoff=DATA_CUTOFF) is True
    earlier_run_cutoff = DATA_CUTOFF - timedelta(hours=1)
    assert _is_valid_stub_payload(round_tripped, expected_data_cutoff=earlier_run_cutoff) is False
    panel = _under10_panel(
        _stub_analysis(
            data_quality={"under10_assessment": round_tripped}, data_cutoff=earlier_run_cutoff
        ),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "unsupported"


def test_c1_assessment_evaluated_before_a_later_parent_run_cutoff_is_rejected(sec_config) -> None:
    """The converse direction: an assessment recorded with a `data_cutoff`
    *earlier* than a later parent run's own cutoff (e.g. transplanted from
    an earlier re-run of the same target date) is just as much a
    contradiction under exact-equality binding -- the generator is always
    called with the exact same `decision_time` that becomes the run's own
    cutoff, so a genuine payload's recorded value is never independently
    earlier either."""
    payload = _build_payload(sec_config)
    round_tripped = _round_trip(payload)
    later_run_cutoff = DATA_CUTOFF + timedelta(hours=1)
    assert _is_valid_stub_payload(round_tripped, expected_data_cutoff=later_run_cutoff) is False
    panel = _under10_panel(
        _stub_analysis(
            data_quality={"under10_assessment": round_tripped}, data_cutoff=later_run_cutoff
        ),
        current_price_band=None,
    )
    assert panel is not None
    assert panel["state"] == "unsupported"


# ---------------------------------------------------------------------------
# C2: whenever `free_cash_flow` is usable/present, its complete duration
# triple must also be present -- a usable FCF value can never genuinely
# lack a resolved flow period in real generator output.
# ---------------------------------------------------------------------------


def test_c2_usable_fcf_requires_a_complete_duration_triple(sec_config) -> None:
    """The exact C2 reproduction: a genuine partial (`insufficient_evidence`,
    missing debt components) assessment with negative FCF still computes
    an independent runway. Nulling the duration triple entirely (with the
    checksum recomputed to match) must never let that runway keep
    rendering, since a usable/present FCF value can never genuinely lack
    its own resolved duration in real generator output."""
    facts = _solvency_facts(
        short_term_debt=None,
        current_long_term_debt=None,
        operating_cash_flow="100.00000000",
        capital_expenditure="400.00000000",
    )
    payload = _build_payload(sec_config, facts=facts)
    assert payload["solvency"]["status"] == SOLVENCY_INSUFFICIENT_EVIDENCE
    assert payload["solvency"]["runway"]["status"] == RUNWAY_COMPUTED
    assert payload["solvency"]["periods"]["duration_end"] is not None
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["solvency"]["periods"] = {
        **corrupted["solvency"]["periods"],
        "duration_start": None,
        "duration_end": None,
        "duration_basis": None,
    }
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# C3: the recorded policy hash must be the one authoritative, recognized
# hash for this policy version -- not merely a non-empty string alongside a
# correctly recomputed assessment checksum.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "corrupt_policy_hash", ["f" * 64, None, "not-even-hex", "0" * 63 + "G", ""]
)
def test_c3_unrecognized_policy_hash_is_rejected(sec_config, corrupt_policy_hash: object) -> None:
    """The exact C3 reproduction: an unknown, missing, or malformed policy
    hash must never be accepted, even alongside a correctly recomputed
    assessment checksum -- only the one authoritative hash for this policy
    version is trusted."""
    payload = _build_payload(sec_config)
    assert payload["policy_hash"] != corrupt_policy_hash
    corrupted = json.loads(json.dumps(payload))
    corrupted["policy_hash"] = corrupt_policy_hash
    _assert_unsupported(corrupted)


def test_c3_genuine_policy_hash_matches_the_authoritative_generator_value(sec_config) -> None:
    """Vacuity guard: the genuine payload's own `policy_hash` really is the
    authoritative value the reader now requires, computed the same way by
    the generator itself -- proving the corruption tests above reject a
    genuinely *wrong* value, not merely any value that happens to differ
    from what the reader was seeded with."""
    payload = _build_payload(sec_config)
    assert payload["policy_hash"] == under10_policy_hash(sec_config)
    _assert_readable(payload)


# ---------------------------------------------------------------------------
# C4: split-refusal metadata's provider/plan/status/reason relationships
# must be internally consistent -- claiming Twelve Data's Basic-plan
# refusal for a provider or plan that was never actually recorded is a
# contradiction, even with a correctly recomputed checksum.
# ---------------------------------------------------------------------------


def test_c4_plan_not_entitled_reason_requires_the_twelve_data_provider(sec_config) -> None:
    payload = _build_payload(sec_config)  # provider="twelve_data", provider_plan="basic"
    assert payload["split_verification"]["reason"] == CAPABILITY_PLAN_NOT_ENTITLED
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["split_verification"]["provider"] = "other_provider"
    _assert_unsupported(corrupted)


def test_c4_plan_not_entitled_reason_requires_a_recorded_plan(sec_config) -> None:
    payload = _build_payload(sec_config)
    assert payload["split_verification"]["plan_recorded"] is True
    corrupted = json.loads(json.dumps(payload))
    corrupted["split_verification"]["plan_recorded"] = False
    _assert_unsupported(corrupted)


def test_c4_genuine_generic_no_reviewed_source_branch_still_passes(sec_config) -> None:
    """Preserves the generic refusal branch for a genuinely non-Twelve-Data
    provider (or non-entitled-plan combination that the generator itself
    classifies as the generic reason) -- must not be swept up by the C4
    provider/plan tightening, and must never claim a plan that was not
    actually recorded."""
    payload = _build_payload(sec_config, provider="other_provider", provider_plan=None)
    assert payload["split_verification"]["reason"] == CAPABILITY_NO_REVIEWED_SOURCE
    assert payload["split_verification"]["plan_recorded"] is False
    _assert_readable(payload)


# ---------------------------------------------------------------------------
# F1: a stored `split_verification.reason` that is a JSON list/dict (not
# `None`, but unhashable) must never raise `TypeError` from the recognized-
# reasons `frozenset` membership test -- rejected as unsupported, exactly
# like any other malformed reason, never a 500.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("malformed_reason", [["a", "list"], {"a": "dict"}, [], {}])
def test_f1_split_reason_as_list_or_dict_is_rejected_without_raising(
    sec_config, malformed_reason: object
) -> None:
    payload = _build_payload(sec_config)
    corrupted = json.loads(json.dumps(payload))
    corrupted["split_verification"]["reason"] = malformed_reason
    _assert_unsupported(corrupted)


@pytest.mark.parametrize("malformed_reason", [["a", "list"], {"a": "dict"}])
def test_f1_is_valid_under10_split_never_raises_for_list_or_dict_reason(
    malformed_reason: object,
) -> None:
    """Direct validator-function coverage, isolating the exact reproduction:
    calling `_is_valid_under10_split` with an otherwise well-formed split
    block whose ``reason`` is a JSON list/dict must return ``False``, never
    raise."""
    from stanstock.web.views import _is_valid_under10_split

    split = {
        "status": CAPABILITY_UNAVAILABLE,
        "capability": SPLIT_EVENT_CAPABILITY,
        "inference_prohibited": True,
        "reason": malformed_reason,
        "provider": "twelve_data",
        "plan_recorded": True,
    }
    assert _is_valid_under10_split(split) is False


@pytest.mark.django_db
@pytest.mark.parametrize("malformed_reason", [["a", "list"], {"a": "dict"}])
def test_f1_split_reason_as_list_or_dict_renders_unsupported_over_http(
    sec_config,
    authenticated_client,
    persisted_analysis,
    django_assert_max_num_queries,
    malformed_reason: object,
) -> None:
    """The exact F1 reproduction over the full view/template stack: a
    stored list/dict split reason (with `assessment_hash` recomputed to
    match) must render HTTP 200 with an ``unsupported`` panel -- never a
    500, and never a favorable render."""
    payload = _build_payload(sec_config, listing_id=str(persisted_analysis.listing_id))
    payload["split_verification"]["reason"] = malformed_reason
    _align_run_target_date(persisted_analysis, TARGET_DATE)
    _make_under_ten(persisted_analysis, assessment=payload)

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.parametrize("malformed_reason", [["a", "list"], {"a": "dict"}])
def test_f1_liquidity_reason_as_list_or_dict_is_rejected_without_raising(
    sec_config, malformed_reason: object
) -> None:
    """Guard regression for the one equivalent membership site the F1 audit
    found: `_is_valid_under10_liquidity`'s own withheld-reason check
    (`_RECOGNIZED_LIQUIDITY_WITHHELD_REASONS`, also a `frozenset`) is
    already protected by an `isinstance(reason, str)` guard evaluated
    first via short-circuit `or` -- pinned here with a genuine withheld
    payload so a future refactor can never silently drop that ordering."""
    payload = _build_payload(sec_config, price_frame=_price_frame(sessions=100))
    assert payload["liquidity"]["status"] == LIQUIDITY_WITHHELD
    assert payload["liquidity"]["reason"] == "insufficient_sessions"
    corrupted = json.loads(json.dumps(payload))
    corrupted["liquidity"]["reason"] = malformed_reason
    _assert_unsupported(corrupted)


# ---------------------------------------------------------------------------
# F2: assessment identity is bound to the permanent `Listing.id`, not just
# the run target date/cutoff/reference close/currency and the immutable
# price asset UUID+checksum. Copying an entire genuine `data_quality` blob
# (the payload plus its sibling `price_source`/`source_assets` anchors)
# from one analysis to another moves every other internal anchor along
# with it, so only the permanent listing id -- read from the payload,
# never re-derived from the copied blob -- can catch that transplant.
# ---------------------------------------------------------------------------


def _complete_data_quality(payload: dict[str, Any], *, price_asset: DataAsset) -> dict[str, Any]:
    """A genuine analysis's complete Under-$10-relevant `data_quality`
    blob: the assessment itself, plus the sibling `price_source`/
    `source_assets` anchors `compute_listing_analysis` populates from the
    exact same `DataAsset` -- everything F2 proved moves together when an
    entire blob is transplanted."""
    return {
        "under10_assessment": payload,
        "price_source": {"asset_id": str(price_asset.id)},
        "source_assets": [{"id": str(price_asset.id), "sha256": price_asset.sha256}],
    }


def test_f2_whole_data_quality_blob_transplanted_between_listings_is_rejected(
    sec_config,
) -> None:
    """The exact F2 reproduction: two genuine analyses share the identical
    run target date/cutoff/reference close/currency but belong to
    different listings and carry different price assets and solvency
    evidence. Each analysis's own complete `data_quality` blob (payload
    plus its own `price_source`/`source_assets`) renders against itself,
    but transplanting either blob whole-cloth onto the other listing's
    identity must be rejected in both directions."""
    listing_a_id = LISTING_ID
    listing_b_id = str(uuid4())
    assert listing_a_id != listing_b_id
    asset_a = _price_asset()
    asset_b = _other_price_asset()
    payload_a = _build_payload(sec_config, price_asset=asset_a, listing_id=listing_a_id)
    payload_b = _build_payload(
        sec_config,
        price_asset=asset_b,
        listing_id=listing_b_id,
        facts=_solvency_facts(cash=("2000.00000000", INSTANT_DATE)),
    )
    assert payload_a["evaluated_for"]["listing_id"] != payload_b["evaluated_for"]["listing_id"]
    assert (
        payload_a["liquidity"]["price_asset"]["id"] != payload_b["liquidity"]["price_asset"]["id"]
    )
    assert (
        payload_a["solvency"]["inputs"]["cash_and_equivalents"]
        != payload_b["solvency"]["inputs"]["cash_and_equivalents"]
    )
    blob_a = _complete_data_quality(_round_trip(payload_a), price_asset=asset_a)
    blob_b = _complete_data_quality(_round_trip(payload_b), price_asset=asset_b)

    def _panel_for(data_quality: dict[str, Any], *, listing_id: str) -> dict[str, Any]:
        panel = _under10_panel(
            _stub_analysis(data_quality=data_quality, listing_id=listing_id),
            current_price_band=None,
        )
        assert panel is not None
        return panel

    # Each analysis's own genuine, complete blob renders.
    assert _panel_for(blob_a, listing_id=listing_a_id)["state"] == "recorded"
    assert _panel_for(blob_b, listing_id=listing_b_id)["state"] == "recorded"
    # Transplanting the *entire* blob (payload plus its own internal
    # anchors, unchanged) onto the other listing's identity must fail in
    # both directions -- the internal anchors all moved together, only the
    # permanent listing id catches it.
    assert _panel_for(blob_a, listing_id=listing_b_id)["state"] == "unsupported"
    assert _panel_for(blob_b, listing_id=listing_a_id)["state"] == "unsupported"


def test_f2_recorded_listing_id_must_match_exactly(sec_config) -> None:
    """Direct validator coverage: a genuine payload's own recorded
    ``evaluated_for.listing_id`` mutated to a different (but well-formed)
    UUID, with `assessment_hash` recomputed to match, is unsupported;
    the genuine value is readable."""
    payload = _build_payload(sec_config)
    _assert_readable(payload)
    corrupted = json.loads(json.dumps(payload))
    corrupted["evaluated_for"]["listing_id"] = str(uuid4())
    _assert_unsupported(corrupted)


@pytest.mark.django_db
def test_f2_whole_data_quality_blob_transplant_is_rejected_over_http(
    sec_config, authenticated_client, persisted_analysis, django_assert_max_num_queries
) -> None:
    """The full F2 scenario over the HTTP/view/template stack: a second,
    unrelated listing's genuine, complete `data_quality` blob (a different
    price asset and solvency evidence, but the identical run target date/
    cutoff/reference close/currency this fixture always uses) is
    transplanted whole-cloth onto `persisted_analysis` -- which belongs to
    a *different* listing. Even though every internal anchor inside the
    blob is mutually self-consistent, it must render unsupported."""
    _align_run_target_date(persisted_analysis, TARGET_DATE)
    other_listing_id = str(uuid4())
    assert other_listing_id != str(persisted_analysis.listing_id)
    asset = _other_price_asset()
    foreign_payload = _build_payload(
        sec_config,
        price_asset=asset,
        listing_id=other_listing_id,
        facts=_solvency_facts(cash=("2000.00000000", INSTANT_DATE)),
    )
    _make_under_ten(persisted_analysis, assessment=foreign_payload)
    # `_make_under_ten` (correctly, per its own F2 alignment) would rebind
    # `evaluated_for.listing_id` to `persisted_analysis`'s own listing --
    # exactly the realignment a genuine payload would have. Overwriting it
    # back to the foreign listing id (recomputing the checksum to match)
    # is what actually reproduces "an entire foreign blob, unmodified,
    # attached to this analysis".
    stored = StockAnalysis.objects.get(pk=persisted_analysis.pk)
    recorded = dict(stored.data_quality["under10_assessment"])
    recorded["evaluated_for"] = {**recorded["evaluated_for"], "listing_id": other_listing_id}
    recorded["assessment_hash"] = under10_assessment_hash(recorded)
    quality = {
        **stored.data_quality,
        "under10_assessment": recorded,
        "price_source": {"asset_id": str(asset.id)},
        "source_assets": [{"id": str(asset.id), "sha256": asset.sha256}],
    }
    stored.data_quality = quality
    stored.save(update_fields=["data_quality"])

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    assert detail.context["under10_panel"]["state"] == "unsupported"
