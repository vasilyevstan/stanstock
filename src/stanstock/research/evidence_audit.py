"""Read-only offline audit of SEC evidence selection for long forecasts.

This module answers the operator questions that `us-sec-long-v3`'s evidence
selection depends on, using only already-persisted immutable rows:

- which source alias anchors the newest eligible quarter for each canonical
  TTM concept, and which real filed fact controls that choice;
- whether that alias supplies a homogeneous four-quarter tail;
- which other aliases collide at that quarter, and which of them are merely
  stale-but-complete alternatives;
- whether a compatible beginning/end invested-capital pair exists, and
  whether the frozen independent nearest-date choices would have missed it.

It performs no HTTP or provider call, writes no facts, assets, models, or
files, and never reads a provider's current state.

**The audit mirrors the forecast's three separate boundaries.** A forecast
run has a historical *target date* (which reporting periods may enter the
window), a *data cutoff* controlling fact availability, and an as-of
*decision time* controlling which evidence rows and source assets are visible
at all. Collapsing them into one "as of" argument would let the audit report
facts a forecast could never have read -- or hide facts it did read -- so all
three are required explicitly here and are never guessed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from uuid import UUID

from stanstock.data.asof import AsOfData
from stanstock.data.models import FundamentalFact, Listing
from stanstock.data.sec_config import SecFundamentalsConfig, load_sec_fundamentals_config
from stanstock.data.sec_fundamentals import (
    NoncanonicalInstantIdentityError,
    SecFundamentalSeries,
    build_sec_fundamental_series,
    source_concept_priority,
)
from stanstock.research.long_forecast_config import LongForecastConfig
from stanstock.research.long_forecasts import (
    LONG_FORECAST_CONCEPTS,
    audit_invested_capital_pairs,
    invested_capital_alias_candidates_enabled,
    ttm_selection_policy,
)

#: 3 adds the explicit ``assessed_withheld`` listing status for conflicting
#: instant period identities and the invested-capital combination-ceiling
#: assessment. The audit is read-only diagnostic output, not persisted
#: evidence, so the version simply tells an operator which fields to expect.
AUDIT_SCHEMA_VERSION = 3

#: Canonical TTM concepts an operator needs to reason about before a long
#: forecast can be produced at all.
AUDITED_TTM_CONCEPTS = (
    "operating_cash_flow",
    "capital_expenditure",
    "net_income",
    "operating_income",
    "pretax_income",
    "income_tax_expense",
    "weighted_average_diluted_shares",
)


class AmbiguousListingSymbolError(ValueError):
    """Raised when a requested ticker matches more than one listing.

    A ticker is not an identity. The same symbol can be listed on two
    exchanges, and an exchange can reuse a symbol for a different company
    after a delisting. Silently keeping one match would attribute one
    company's SEC evidence to another, so the caller must supply the
    immutable listing ID instead.
    """


@dataclass(frozen=True, slots=True)
class AuditTarget:
    requested_listing_id: str | None
    requested_symbol: str | None
    listing: Listing | None
    status: str
    reason: str


def audit_long_evidence(
    *,
    listing_ids: Sequence[str | UUID] = (),
    symbols: Sequence[str] = (),
    target_date: date,
    available_through: datetime,
    decision_time: datetime,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig | None = None,
    asof: AsOfData | None = None,
) -> dict[str, Any]:
    """Return a deterministic, provider-free evidence-selection report.

    ``target_date`` bounds which reporting periods may enter the forecast
    window, ``available_through`` is the historical data cutoff applied to
    fact availability, and ``decision_time`` is the as-of boundary for
    evidence and source-asset visibility. They are separate on purpose and
    are exactly the three boundaries `build_long_forecasts` applies.
    """
    fundamentals = sec_config or load_sec_fundamentals_config()
    if (
        config.fundamentals_config_version is not None
        and config.fundamentals_config_version != fundamentals.config_version
    ):
        raise ValueError(
            f"Long forecast config {config.version!r} binds SEC fundamentals "
            f"{config.fundamentals_config_version!r}, but the loaded fundamentals "
            f"configuration is {fundamentals.config_version!r}"
        )
    _validate_boundaries(
        target_date=target_date,
        available_through=available_through,
        decision_time=decision_time,
    )
    if asof is not None and asof.decision_time != decision_time:
        raise ValueError(
            "The supplied AsOfData reader uses decision time "
            f"{asof.decision_time.isoformat()}, which is not the requested audit "
            f"decision time {decision_time.isoformat()}"
        )
    if not listing_ids and not symbols:
        raise ValueError("Auditing requires at least one listing ID or ticker symbol")
    reader = asof or AsOfData(decision_time)
    ttm_selection = ttm_selection_policy(config)
    alias_candidates = invested_capital_alias_candidates_enabled(config)
    targets = resolve_audit_targets(listing_ids=listing_ids, symbols=symbols)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "target_date": target_date.isoformat(),
        "available_through": available_through.isoformat(),
        "decision_time": decision_time.isoformat(),
        "long_forecast_config_version": config.version,
        "fundamentals_config_version": fundamentals.config_version,
        "fundamentals_config_hash": fundamentals.config_hash,
        "ttm_selection_policy": ttm_selection,
        "invested_capital_alias_candidates": alias_candidates,
        "balance_sheet_date_tolerance_days": (config.eligibility.balance_sheet_date_tolerance_days),
        "listings": [
            _listing_report(
                target=target,
                reader=reader,
                target_date=target_date,
                available_through=available_through,
                config=config,
                sec_config=fundamentals,
                ttm_selection=ttm_selection,
                alias_candidates=alias_candidates,
            )
            for target in targets
        ],
    }


def _validate_boundaries(
    *,
    target_date: date,
    available_through: datetime,
    decision_time: datetime,
) -> None:
    for label, value in (
        ("decision_time", decision_time),
        ("available_through", available_through),
    ):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError(
                f"{label} must be timezone-aware so the point-in-time boundary is unambiguous"
            )
    if available_through > decision_time:
        raise ValueError(
            f"available_through ({available_through.isoformat()}) cannot be after the "
            f"decision time ({decision_time.isoformat()})"
        )
    if target_date > available_through.date():
        raise ValueError(
            f"target_date ({target_date.isoformat()}) cannot be after the data cutoff date "
            f"({available_through.date().isoformat()})"
        )


def resolve_audit_targets(
    *,
    listing_ids: Sequence[str | UUID] = (),
    symbols: Sequence[str] = (),
) -> list[AuditTarget]:
    """Resolve immutable listing IDs first, then unambiguous ticker symbols.

    An immutable listing ID is the canonical input. A ticker is accepted only
    as operator convenience and only when exactly one listing matches;
    anything else raises `AmbiguousListingSymbolError` rather than picking a
    winner.
    """
    targets: list[AuditTarget] = []
    for raw_listing_id in dict.fromkeys(str(value) for value in listing_ids):
        try:
            parsed = UUID(raw_listing_id)
        except ValueError:
            targets.append(
                AuditTarget(
                    requested_listing_id=raw_listing_id,
                    requested_symbol=None,
                    listing=None,
                    status="invalid_listing_id",
                    reason="Listing IDs must be UUIDs",
                )
            )
            continue
        listing = Listing.objects.select_related("security__company").filter(pk=parsed).first()
        targets.append(
            AuditTarget(
                requested_listing_id=raw_listing_id,
                requested_symbol=None,
                listing=listing,
                status="resolved" if listing is not None else "unknown_listing",
                reason=("" if listing is not None else "No listing with this ID exists locally"),
            )
        )
    for symbol in dict.fromkeys(symbols):
        matches = list(
            Listing.objects.select_related("security__company")
            .filter(ticker=symbol)
            .order_by("exchange_mic", "valid_from", "pk")
        )
        if len(matches) > 1:
            raise AmbiguousListingSymbolError(
                f"Ticker {symbol!r} matches {len(matches)} listings "
                f"({', '.join(f'{item.exchange_mic}:{item.pk}' for item in matches)}); "
                "audit by immutable listing ID instead"
            )
        listing = matches[0] if matches else None
        targets.append(
            AuditTarget(
                requested_listing_id=None,
                requested_symbol=symbol,
                listing=listing,
                status="resolved" if listing is not None else "unknown_listing",
                reason=(
                    "" if listing is not None else "No listing with this ticker exists locally"
                ),
            )
        )
    return targets


def _listing_report(
    *,
    target: AuditTarget,
    reader: AsOfData,
    target_date: date,
    available_through: datetime,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
    ttm_selection: str,
    alias_candidates: bool,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "requested_listing_id": target.requested_listing_id,
        "requested_symbol": target.requested_symbol,
    }
    if target.listing is None:
        return {
            **request,
            "listing_id": None,
            "symbol": target.requested_symbol,
            "status": target.status,
            "reason": target.reason,
        }
    listing = target.listing
    identity: dict[str, Any] = {
        **request,
        "listing_id": str(listing.pk),
        "symbol": listing.ticker,
        "exchange_mic": listing.exchange_mic,
        "valid_from": listing.valid_from.isoformat() if listing.valid_from else None,
        "valid_to": listing.valid_to.isoformat() if listing.valid_to else None,
    }
    facts = audit_visible_facts(
        reader=reader,
        listing=listing,
        target_date=target_date,
        available_through=available_through,
        config=config,
    )
    if not facts:
        return {
            **identity,
            "status": "no_visible_facts",
            "reason": (
                "No SEC fundamental fact is available under the requested "
                "target-date, data-cutoff, and decision-time boundaries"
            ),
            "visible_fact_count": 0,
            "visible_fact_ids": [],
        }
    try:
        series = build_sec_fundamental_series(
            facts,
            config=sec_config,
            ttm_selection=ttm_selection,
            alias_instant_candidates=alias_candidates,
        )
    except NoncanonicalInstantIdentityError as error:
        # One listing's conflicting instant identities must not silence the
        # rest of the audit, and they must never be resolved by row order.
        return {
            **identity,
            "status": "assessed_withheld",
            "reason": (
                "Balance-sheet evidence is unusable: "
                f"{error}. Same-date alias selection needs one canonical instant "
                "identity per balance-sheet date"
            ),
            "visible_fact_count": len(facts),
            "visible_fact_ids": sorted(str(fact.pk) for fact in facts),
            "noncanonical_instant_facts": list(error.anomalies),
        }
    return {
        **identity,
        "status": "audited",
        "visible_fact_count": len(facts),
        "visible_fact_ids": sorted(str(fact.pk) for fact in facts),
        "ttm_alias_selection": _alias_report(series=series, facts=facts),
        "invested_capital": _invested_capital_report(
            series=series,
            config=config,
            sec_config=sec_config,
            alias_candidates=alias_candidates,
        ),
    }


def audit_visible_facts(
    *,
    reader: AsOfData,
    listing: Listing,
    target_date: date,
    available_through: datetime,
    config: LongForecastConfig,
) -> list[FundamentalFact]:
    """Read exactly the facts `build_long_forecasts` would read.

    The period window is bounded by the forecast ``target_date`` -- not by
    the reader's own clock -- so a quarter that ends between the target date
    and the audit run is excluded here just as it is in the forecast. Fact
    availability is bounded by ``available_through``, while asset visibility
    stays on the reader's decision time.
    """
    period_lookback = timedelta(
        days=(max(family.minimum_annual_periods for family in config.metric_families.values()) + 1)
        * 366
    )
    return list(
        reader.fundamental_facts(
            company_id=listing.security.company_id,
            concepts=list(LONG_FORECAST_CONCEPTS),
            available_through=available_through,
        )
        .filter(provider=config.fundamentals_provider)
        .filter(
            period_end__gte=target_date - period_lookback,
            period_end__lte=target_date,
        )
    )


def _alias_report(
    *,
    series: SecFundamentalSeries,
    facts: list[FundamentalFact],
) -> list[dict[str, Any]]:
    observed_aliases: dict[str, set[str]] = {}
    for fact in facts:
        observed_aliases.setdefault(fact.concept, set()).add(fact.source_concept)
    report: list[dict[str, Any]] = []
    for concept in AUDITED_TTM_CONCEPTS:
        selection = series.ttm_alias_selection.get(concept)
        value = series.ttm.get(concept)
        entry: dict[str, Any] = {
            "concept": concept,
            "observed_source_concepts": sorted(observed_aliases.get(concept, set())),
            "ttm_available": value is not None,
            "ttm_period_start": value.period_start.isoformat() if value is not None else None,
            "ttm_period_end": value.period_end.isoformat() if value is not None else None,
            "selection": selection,
        }
        if selection is not None:
            entry["alias_collision"] = (
                sum(
                    1
                    for candidate in selection.get("alias_candidates", ())
                    if candidate.get("observes_newest_quarter")
                )
                > 1
            )
            entry["stale_complete_alternatives"] = selection.get(
                "stale_complete_alternatives",
                [],
            )
            entry["homogeneous_four_quarter_tail"] = selection.get(
                "homogeneous_four_quarter_tail",
                False,
            )
            entry["controlling_source_fact"] = selection.get("controlling_source_fact")
        report.append(entry)
    return report


def _invested_capital_report(
    *,
    series: SecFundamentalSeries,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
    alias_candidates: bool,
) -> dict[str, Any]:
    if not alias_candidates:
        return {
            "status": "not_assessed",
            "reason": (
                "This configuration does not enable joint compatible "
                "invested-capital pair selection"
            ),
        }
    anchor = series.ttm.get("free_cash_flow") or series.ttm.get("net_income")
    if anchor is None:
        return {
            "status": "unavailable",
            "reason": (
                "No TTM free cash flow or net income window is available to anchor "
                "the invested-capital target dates"
            ),
        }
    report = audit_invested_capital_pairs(
        series=series,
        beginning_target=anchor.period_start - timedelta(days=1),
        ending_target=anchor.period_end,
        tolerance_days=config.eligibility.balance_sheet_date_tolerance_days,
        priority=source_concept_priority(sec_config),
    )
    report["status"] = "audited"
    report["anchor_concept"] = anchor.concept
    return report
