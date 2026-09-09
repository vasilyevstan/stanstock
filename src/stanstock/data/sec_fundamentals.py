from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from stanstock.data.fact_identity import build_observation_hash, build_period_identity
from stanstock.data.models import FundamentalFact
from stanstock.data.sec_config import SecFundamentalsConfig

MIN_QUARTER_DAYS = 70
MAX_QUARTER_DAYS = 110
MIN_ANNUAL_DAYS = 350
MAX_ANNUAL_DAYS = 380

#: Frozen legacy TTM construction: collapse every alias to one selected
#: vintage per (concept, period identity), then take the last four quarters.
#: This is what `us-sec-long-v1`/`us-sec-long-v2` and every generic consumer
#: read, and it must stay byte/behavior compatible.
TTM_SELECTION_LEGACY = "legacy_selected_vintage_tail"

#: `us-sec-long-v3` evidence selection: anchor on the newest eligible quarter
#: end, pick the highest-ranked eligible observation at that quarter under the
#: existing revision/availability/source-priority rules, and build the TTM
#: only from four contiguous compatible quarters supplied by that same source
#: alias. No cross-alias stitching, and no annual current-period fallback.
TTM_SELECTION_NEWEST_QUARTER_ALIAS = "newest_quarter_anchored_homogeneous_alias"

TTM_SELECTION_POLICIES = (TTM_SELECTION_LEGACY, TTM_SELECTION_NEWEST_QUARTER_ALIAS)

#: `FundamentalFact.availability_basis` for a same-accession correction.
#:
#: The SEC can restate a value *under the accession it already filed*. The
#: original acceptance timestamp still describes when the filing was
#: accepted, but it does not describe when the corrected value became
#: knowable: nothing before the retrieval that first carried that value
#: proves it existed. A correction ingested under this basis therefore
#: records the retrieval boundary in ``available_at`` while ``acceptance_at``
#: and ``filed_at`` keep the original, unmodified acceptance.
CORRECTION_AVAILABILITY_BASIS = "same_accession_correction_retrieval"

#: Quality flag carried by every fact ingested under that basis.
CORRECTION_QUALITY_FLAG = "same_accession_correction"

#: Additional flag for a vintage appended because a *fresh* retrieval
#: re-observed content whose newest persisted revision was a legacy
#: correction with no provable timing. The value is unchanged; the new row
#: exists solely to carry an availability boundary that real evidence
#: supports. The unprovable row it follows is never mutated.
REBOUND_QUALITY_FLAG = "reobserved_unproven_correction"


@dataclass(frozen=True, slots=True)
class ResolvedAvailability:
    """Chain-resolution outcome for one persisted observation.

    ``proven_at`` is ``None`` when the row's timing cannot be proven at all.
    Such a row is never admitted at any cutoff, because no persisted evidence
    justifies one.

    ``earliest_possible_at`` is *assessed context*, never an admission
    boundary. It records the earliest moment a chain's ordering allows -- for
    example, that a revision cannot predate the revision it supersedes -- so
    an operator can see how far the uncertainty reaches. A lower bound is not
    proof, so it is deliberately kept out of every admission decision.
    """

    fact: FundamentalFact
    proven_at: datetime | None
    basis: str
    reason: str
    earliest_possible_at: datetime | None = None


#: A correction ingested with its own bound observation event.
RESOLUTION_BOUND_OBSERVATION = "bound_observation_event"
#: An original (revision 1) observation: acceptance is the honest boundary.
RESOLUTION_ORIGINAL_ACCEPTANCE = "original_acceptance"
#: A legacy correction bounded by the retrieval of the asset it came from.
RESOLUTION_LEGACY_ASSET_RETRIEVAL = "legacy_asset_retrieval"
#: A legacy content reversion: its bytes deduplicated onto the asset of an
#: earlier revision, so it has no distinct observation to point at.
RESOLUTION_UNPROVABLE_LEGACY_REVERSION = "unprovable_legacy_reversion"
#: A legacy revision whose own asset was retrieved *before* the boundary of
#: the revision it supersedes. Its asset retrieval cannot be the moment this
#: revision became knowable, and nothing else records it.
RESOLUTION_UNPROVABLE_LEGACY_ORDERING = "unprovable_legacy_ordering"


def _correction_chain_key(fact: FundamentalFact) -> tuple[str, ...]:
    """Identity of the revision chain one observation belongs to.

    Exactly the `unique_fundamental_vintage` key minus ``source_revision``,
    so every revision of one filed observation resolves together.
    """
    return (
        str(fact.company_id),
        fact.provider,
        fact.source_concept,
        fact.period_identity,
        fact.accession,
        fact.unit,
    )


def resolve_availability(
    facts: Iterable[FundamentalFact],
) -> dict[str, ResolvedAvailability]:
    """Resolve every observation's provable availability, chain by chain.

    An original observation is proven by its filing acceptance. A correction
    ingested under `CORRECTION_AVAILABILITY_BASIS` already recorded the
    observation event that established it, so its stored ``available_at`` is
    authoritative.

    A *legacy* correction -- persisted before that basis existed -- carries
    the original acceptance instead. It is admitted only when its own source
    asset's retrieval is a real observation *of that revision*: the asset is
    not shared with an earlier revision, and it was retrieved no earlier than
    the boundary of the revision it supersedes.

    Two legacy shapes fail that test and resolve to ``None``:

    - a *reversion*, whose content repeats an earlier revision's so ingestion
      reused that earlier asset, leaving no distinct retrieval to cite; and
    - a *decreasing ordering*, where the asset was retrieved before the
      superseded revision became knowable, so that retrieval cannot be an
      observation of this revision even though the content differs.

    Neither is given a numeric admission boundary. The chain ordering does
    imply a lower bound, but a lower bound is not proof of when a row was
    seen, so it is recorded as ``earliest_possible_at`` assessed context and
    never used to admit anything.

    Nothing here writes: resolution is derived from immutable rows on every
    read.
    """
    resolved: dict[str, ResolvedAvailability] = {}
    chains: dict[tuple[str, ...], list[FundamentalFact]] = {}
    for fact in facts:
        chains.setdefault(_correction_chain_key(fact), []).append(fact)
    for chain in chains.values():
        running: datetime | None = None
        seen_hashes: set[str] = set()
        for fact in sorted(chain, key=lambda item: item.source_revision):
            entry = _resolve_one(fact, running=running, seen_hashes=seen_hashes)
            resolved[str(fact.pk)] = entry
            if fact.observation_hash:
                seen_hashes.add(fact.observation_hash)
            if entry.proven_at is not None:
                running = entry.proven_at if running is None else max(running, entry.proven_at)
    return resolved


def _resolve_one(
    fact: FundamentalFact,
    *,
    running: datetime | None,
    seen_hashes: set[str],
) -> ResolvedAvailability:
    if fact.source_revision <= 1:
        return ResolvedAvailability(
            fact=fact,
            proven_at=fact.available_at,
            basis=RESOLUTION_ORIGINAL_ACCEPTANCE,
            reason="",
        )
    if fact.availability_basis == CORRECTION_AVAILABILITY_BASIS:
        return ResolvedAvailability(
            fact=fact,
            proven_at=fact.available_at,
            basis=RESOLUTION_BOUND_OBSERVATION,
            reason="",
        )
    if fact.observation_hash and fact.observation_hash in seen_hashes:
        return ResolvedAvailability(
            fact=fact,
            proven_at=None,
            basis=RESOLUTION_UNPROVABLE_LEGACY_REVERSION,
            reason=(
                f"Revision {fact.source_revision} of accession {fact.accession} restates "
                "an earlier revision's exact content, so its raw evidence deduplicated "
                "onto that earlier retrieval and no persisted observation proves when "
                "this revision was actually seen"
            ),
        )
    own = max(fact.available_at, fact.source_asset.retrieved_at)
    if running is not None and running > own:
        # The asset this revision came from was retrieved *before* the
        # revision it supersedes became knowable, so that retrieval is not
        # an observation of this revision at all. A monotonic lower bound
        # ("no earlier than its predecessor") is not proof of when this row
        # was actually seen, and admitting it at that bound would still be a
        # guess. The bound is kept as assessed context only.
        return ResolvedAvailability(
            fact=fact,
            proven_at=None,
            basis=RESOLUTION_UNPROVABLE_LEGACY_ORDERING,
            reason=(
                f"Revision {fact.source_revision} of accession {fact.accession} comes "
                f"from an asset retrieved at {fact.source_asset.retrieved_at.isoformat()}, "
                f"before the revision it supersedes became knowable at "
                f"{running.isoformat()}, so no persisted observation proves when this "
                "revision was actually seen"
            ),
            earliest_possible_at=running,
        )
    return ResolvedAvailability(
        fact=fact,
        proven_at=own,
        basis=RESOLUTION_LEGACY_ASSET_RETRIEVAL,
        reason="",
    )


def proven_availability(fact: FundamentalFact) -> datetime | None:
    """Provable availability of one observation, resolved in isolation.

    Convenience wrapper for a single row. Prefer `resolve_availability` when
    a whole chain is in hand: only chain context can lift a legacy revision
    to the boundary of the revision it supersedes.
    """
    return resolve_availability([fact])[str(fact.pk)].proven_at


def partition_unproven_corrections(
    facts: Iterable[FundamentalFact],
    *,
    available_through: datetime,
) -> tuple[tuple[FundamentalFact, ...], tuple[ResolvedAvailability, ...]]:
    """Split facts into those proven available by ``available_through``, and not.

    The caller has already applied the recorded ``available_at`` cutoff. This
    re-applies the same cutoff against the resolved availability, which
    differs only for a correction. The result is conservative in exactly one
    direction: a correction whose timing is not proven is deferred, and the
    revision it superseded -- which *is* proven at that cutoff -- remains
    available in its place.

    Rows are never mutated; deferral is a read-time decision.
    """
    ordered = list(facts)
    resolved = resolve_availability(ordered)
    admitted: list[FundamentalFact] = []
    deferred: list[ResolvedAvailability] = []
    for fact in ordered:
        entry = resolved[str(fact.pk)]
        if entry.proven_at is not None and entry.proven_at <= available_through:
            admitted.append(fact)
        else:
            deferred.append(entry)
    return tuple(admitted), tuple(deferred)


def deferred_correction_payload(
    entry: ResolvedAvailability,
    *,
    available_through: datetime,
) -> dict[str, Any]:
    """Explicit record of one correction withheld for unproven timing.

    ``fact_id`` is deliberately named so the assessed-evidence closure picks
    the row up automatically: a deferred correction is evidence the run read
    and rejected, never a selected input and never silently dropped.
    """
    fact = entry.fact
    if entry.proven_at is None:
        reason = entry.reason
    else:
        reason = (
            f"Revision {fact.source_revision} restates accession {fact.accession} under "
            "the original acceptance timestamp, but no evidence proves the corrected "
            f"value existed before {entry.proven_at.isoformat()}, which is after the "
            f"data cutoff {available_through.isoformat()}"
        )
    return {
        "fact_id": str(fact.pk),
        "concept": fact.concept,
        "source_concept": fact.source_concept,
        "period_identity": fact.period_identity,
        "period_end": fact.period_end.isoformat(),
        "accession": fact.accession,
        "source_revision": fact.source_revision,
        "availability_basis": fact.availability_basis,
        "resolution_basis": entry.basis,
        "recorded_available_at": fact.available_at.isoformat(),
        "acceptance_at": (fact.acceptance_at.isoformat() if fact.acceptance_at else None),
        "proven_available_at": (
            entry.proven_at.isoformat() if entry.proven_at is not None else None
        ),
        # Assessed context only. A chain-ordering lower bound says this row
        # cannot have been knowable *earlier* than this; it never says it was
        # knowable *by* then, so it admits nothing.
        "earliest_possible_available_at": (
            entry.earliest_possible_at.isoformat()
            if entry.earliest_possible_at is not None
            else None
        ),
        "source_asset_id": str(fact.source_asset_id),
        "source_asset_retrieved_at": fact.source_asset.retrieved_at.isoformat(),
        "available_through": available_through.isoformat(),
        "reason": reason,
    }


ADDITIVE_FLOW_CONCEPTS = frozenset(
    {
        "revenue",
        "operating_income",
        "pretax_income",
        "income_tax_expense",
        "net_income",
        "operating_cash_flow",
        "capital_expenditure",
        "depreciation_amortization",
        "interest_expense",
        "dividends_paid",
        "share_repurchases",
    }
)
WEIGHTED_AVERAGE_CONCEPTS = frozenset({"weighted_average_diluted_shares"})


class NoncanonicalInstantIdentityError(ValueError):
    """An instant SEC fact does not carry the canonical instant identity.

    `us-sec-long-v3` joins balance-sheet aliases on
    ``(canonical concept, source alias, period identity)``. That join is only
    sound while every instant observation for one balance-sheet date agrees on
    exactly one identity -- the one `build_period_identity` produces for
    ``period_type='instant'``, ``period_start=None``, ``period_end=<date>``.

    A conflicting or non-canonical identity would either split one date into
    several pseudo-dates (inflating the same-date combination space) or make
    two genuinely different observations look interchangeable. Neither may be
    resolved silently, and neither may be tie-broken by a generated primary
    key, so the alias boundary refuses the whole listing and the caller
    records an explicit assessed-withheld result.
    """

    def __init__(self, anomalies: tuple[dict[str, Any], ...]) -> None:
        self.anomalies = anomalies
        examples = "; ".join(
            (
                f"{item['concept']} {item['source_concept']} at {item['period_end']} "
                f"reports {item['period_identity']!r}, expected "
                f"{item['expected_period_identity']!r}"
            )
            for item in anomalies[:3]
        )
        super().__init__(
            f"{len(anomalies)} instant SEC fact(s) carry non-canonical period "
            f"identities: {examples}"
        )


def canonical_instant_period_identity(period_end: date) -> str:
    """The one identity an instant observation for ``period_end`` may carry."""
    return build_period_identity(
        period_type=FundamentalFact.PeriodType.INSTANT.value,
        period_start=None,
        period_end=period_end,
    )


def fact_selection_identity(fact: FundamentalFact) -> str:
    """Stable, database-independent tie-break identity for one observation.

    `FundamentalFact.pk` is a generated UUID, so ordering by it makes
    selection depend on row-creation order rather than on evidence. No
    `us-sec-long-v3` path may do that. The persisted `observation_hash`
    already covers the taxonomy, alias, value, unit, currency, period
    identity, fiscal labels, and filing coordinates of one filed
    observation, so two facts sharing it are the same observation and the
    order between them cannot change any result.
    """
    if fact.observation_hash:
        return fact.observation_hash
    return build_observation_hash(
        taxonomy=fact.taxonomy,
        source_concept=fact.source_concept,
        value=fact.value,
        unit=fact.unit,
        currency=fact.currency,
        period_identity=(
            fact.period_identity
            or build_period_identity(
                period_type=fact.period_type,
                period_start=fact.period_start,
                period_end=fact.period_end,
                fiscal_period=fact.fiscal_period,
                frame=fact.frame,
            )
        ),
        fiscal_year=fact.fiscal_year,
        fiscal_period=fact.fiscal_period,
        accession=fact.accession,
        filing_form=fact.filing_form,
        filing_date=fact.filing_date,
        acceptance_at=fact.acceptance_at,
        frame=fact.frame,
    )


def observation_selection_identity(value: FundamentalValue) -> str:
    """Stable non-UUID identity for one direct or derived observation.

    Used only as the final `us-sec-long-v3` tie-break between two quarter
    observations that are otherwise indistinguishable under the full
    controlling-fact rank. It is derived from the observation's own period
    identity, derivation, unit, aliases, accessions, and value -- never from
    a generated primary key.
    """
    payload = "|".join(
        (
            value.concept,
            value.derivation,
            value.unit,
            build_period_identity(
                period_type=FundamentalFact.PeriodType.DURATION.value,
                period_start=value.period_start,
                period_end=value.period_end,
            ),
            ",".join(value.source_concepts),
            ",".join(value.accessions),
            format(value.value, "f"),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FundamentalValue:
    concept: str
    value: Decimal
    unit: str
    period_start: date
    period_end: date
    available_at: datetime
    source_fact_ids: tuple[str, ...]
    accessions: tuple[str, ...]
    source_concepts: tuple[str, ...]
    derivation: str

    @property
    def duration_days(self) -> int:
        return (self.period_end - self.period_start).days + 1


@dataclass(frozen=True, slots=True)
class SecFundamentalSeries:
    selected_facts: tuple[FundamentalFact, ...]
    annual: dict[str, tuple[FundamentalValue, ...]]
    quarters: dict[str, tuple[FundamentalValue, ...]]
    ttm: dict[str, FundamentalValue]
    instants: dict[str, tuple[FundamentalFact, ...]]
    latest_instants: dict[str, FundamentalFact]
    missing: dict[str, str]
    ttm_selection: str = TTM_SELECTION_LEGACY
    ttm_alias_selection: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Whether the caller asked for the additional same-date alias candidate
    #: surface below. ``False`` means `alias_instants` was never built, which
    #: is a different statement than "no alias candidate exists".
    alias_instant_candidates: bool = False
    #: ``canonical concept -> source alias -> latest eligible vintages``.
    #: `instants` collapses every alias to one winner per period identity, so
    #: a same-date alternative alias is invisible there. This surface keeps
    #: those alternatives for the explicit `us-sec-long-v3` joint search and
    #: is empty for every legacy caller.
    alias_instants: dict[str, dict[str, tuple[FundamentalFact, ...]]] = field(default_factory=dict)


def build_sec_fundamental_series(
    facts: Iterable[FundamentalFact],
    *,
    config: SecFundamentalsConfig,
    ttm_selection: str = TTM_SELECTION_LEGACY,
    alias_instant_candidates: bool = False,
) -> SecFundamentalSeries:
    """Normalize SEC facts into annual, quarterly, TTM, and instant evidence.

    ``ttm_selection`` only changes how the trailing-twelve-month window is
    chosen. ``alias_instant_candidates`` only *adds* the `alias_instants`
    surface. Fact vintage selection, the annual series, the quarter series,
    and the instant series stay on the frozen legacy path for every caller.
    """
    if ttm_selection not in TTM_SELECTION_POLICIES:
        raise ValueError(f"Unsupported TTM selection policy: {ttm_selection!r}")
    all_facts = tuple(facts)
    selected = select_latest_fact_vintages(all_facts, config=config)
    annual = _annual_series(selected)
    quarters = _quarter_series(selected)
    _add_free_cash_flow_series(annual)
    _add_free_cash_flow_series(quarters)
    alias_selection: dict[str, dict[str, Any]] = {}
    if ttm_selection == TTM_SELECTION_NEWEST_QUARTER_ALIAS:
        ttm, alias_selection = _newest_quarter_anchored_ttm_series(all_facts, config=config)
    else:
        ttm = _ttm_series(quarters)
    instants = _instant_series(selected)
    latest_instants = {concept: values[-1] for concept, values in instants.items() if values}
    missing: dict[str, str] = {}
    for required in (
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditure",
        "weighted_average_diluted_shares",
    ):
        if required not in annual and required not in ttm:
            missing[required] = "No compatible annual or trailing-twelve-month SEC history"
    if "free_cash_flow" not in annual and "free_cash_flow" not in ttm:
        missing["free_cash_flow"] = (
            "Need compatible operating cash flow and capital expenditure periods"
        )
    return SecFundamentalSeries(
        selected_facts=selected,
        annual={key: tuple(values) for key, values in annual.items()},
        quarters={key: tuple(values) for key, values in quarters.items()},
        ttm=ttm,
        instants={key: tuple(values) for key, values in instants.items()},
        latest_instants=latest_instants,
        missing=missing,
        ttm_selection=ttm_selection,
        ttm_alias_selection=alias_selection,
        alias_instant_candidates=alias_instant_candidates,
        alias_instants=(
            build_alias_instant_candidates(all_facts, config=config)
            if alias_instant_candidates
            else {}
        ),
    )


def select_latest_fact_vintages(
    facts: Iterable[FundamentalFact],
    *,
    config: SecFundamentalsConfig,
) -> tuple[FundamentalFact, ...]:
    priority = source_concept_priority(config)
    selected: dict[tuple[str, str], FundamentalFact] = {}
    for fact in facts:
        if fact.provider != "sec" or (fact.concept, fact.source_concept) not in priority:
            continue
        key = (fact.concept, fact.period_identity)
        existing = selected.get(key)
        if existing is None or _fact_is_newer(
            candidate=fact,
            existing=existing,
            priority=priority,
        ):
            selected[key] = fact
    return tuple(
        sorted(
            selected.values(),
            key=lambda fact: (
                fact.concept,
                fact.period_end,
                fact.period_start or fact.period_end,
                fact.available_at,
            ),
        )
    )


def source_concept_priority(config: SecFundamentalsConfig) -> dict[tuple[str, str], int]:
    """Return the frozen ``(canonical concept, taxonomy:alias) -> rank`` map.

    A lower index is a higher-priority alias, exactly as declared in the
    reviewed SEC fundamentals configuration.
    """
    return {
        (rule.canonical_concept, f"{taxonomy}:{source_concept}"): index
        for taxonomy in config.allowed_taxonomies
        for rule in config.concept_rules
        for index, source_concept in enumerate(rule.source_concepts)
    }


def build_alias_instant_candidates(
    facts: Iterable[FundamentalFact],
    *,
    config: SecFundamentalsConfig,
) -> dict[str, dict[str, tuple[FundamentalFact, ...]]]:
    """Latest eligible instant vintage per ``(concept, source alias, period)``.

    `select_latest_fact_vintages` collapses every alias to a single winner
    per ``(canonical concept, period identity)``, so a balance-sheet
    alternative reported on the *same* date under a different alias is
    simply invisible downstream. This surface keeps exactly one -- the
    latest, never a superseded revision -- vintage per alias so an explicit
    joint search can consider those same-date alternatives. It is additive:
    nothing here changes `instants`, the legacy selection, or any frozen
    caller.
    """
    priority = source_concept_priority(config)
    selected: dict[tuple[str, str, str], FundamentalFact] = {}
    anomalies: list[dict[str, Any]] = []
    for fact in canonical_fact_order(facts):
        if fact.provider != "sec" or (fact.concept, fact.source_concept) not in priority:
            continue
        if fact.period_type != FundamentalFact.PeriodType.INSTANT:
            continue
        expected = canonical_instant_period_identity(fact.period_end)
        if fact.period_identity != expected or fact.period_start is not None:
            anomalies.append(
                {
                    "fact_id": str(fact.pk),
                    "concept": fact.concept,
                    "source_concept": fact.source_concept,
                    "period_start": (
                        fact.period_start.isoformat() if fact.period_start is not None else None
                    ),
                    "period_end": fact.period_end.isoformat(),
                    "period_identity": fact.period_identity,
                    "expected_period_identity": expected,
                    "observation_hash": fact.observation_hash,
                    "accession": fact.accession,
                    "available_at": fact.available_at.isoformat(),
                }
            )
            continue
        key = (fact.concept, fact.source_concept, fact.period_identity)
        existing = selected.get(key)
        if existing is None or _fact_is_newer(
            candidate=fact,
            existing=existing,
            priority=priority,
        ):
            selected[key] = fact
    grouped: dict[str, dict[str, list[FundamentalFact]]] = {}
    for (concept, alias, _identity), fact in selected.items():
        grouped.setdefault(concept, {}).setdefault(alias, []).append(fact)
    if anomalies:
        # Raised only after every in-scope instant fact has been inspected so
        # the caller reports the complete conflict, not just the first one.
        raise NoncanonicalInstantIdentityError(tuple(anomalies))
    return {
        concept: {
            alias: tuple(canonical_fact_order(alias_facts))
            for alias, alias_facts in sorted(aliases.items())
        }
        for concept, aliases in sorted(grouped.items())
    }


def canonical_fact_order(facts: Iterable[FundamentalFact]) -> list[FundamentalFact]:
    """Order facts so downstream selection cannot depend on input order.

    Only `us-sec-long-v3`-scoped code uses this. The frozen legacy path
    deliberately keeps its original input-order iteration.

    The final tie-break is the stable observation identity, never the
    generated primary key: reassigning row UUIDs must not be able to change
    which evidence a forecast selects.
    """
    return sorted(
        facts,
        key=lambda fact: (
            fact.concept,
            fact.source_concept,
            fact.period_end,
            fact.period_start or fact.period_end,
            fact.period_identity,
            fact.available_at,
            fact.source_revision,
            fact.accession,
            fact.unit,
            fact_selection_identity(fact),
        ),
    )


def _fact_is_newer(
    *,
    candidate: FundamentalFact,
    existing: FundamentalFact,
    priority: dict[tuple[str, str], int],
) -> bool:
    same_revision_series = (
        candidate.source_concept == existing.source_concept
        and candidate.period_identity == existing.period_identity
        and candidate.accession == existing.accession
        and candidate.unit == existing.unit
    )
    if same_revision_series and candidate.source_revision != existing.source_revision:
        return candidate.source_revision > existing.source_revision
    return _fact_rank(candidate, priority) > _fact_rank(existing, priority)


def _fact_rank(
    fact: FundamentalFact,
    priority: dict[tuple[str, str], int],
) -> tuple[datetime, int, int, str]:
    source_priority = priority.get((fact.concept, fact.source_concept), 10_000)
    return (
        fact.available_at,
        fact.source_revision,
        -source_priority,
        fact.accession,
    )


def _annual_series(
    selected: tuple[FundamentalFact, ...],
) -> dict[str, list[FundamentalValue]]:
    annual: dict[str, list[FundamentalValue]] = {}
    for fact in selected:
        if fact.period_type != FundamentalFact.PeriodType.DURATION:
            continue
        value = _value_from_fact(fact)
        if value is None or not MIN_ANNUAL_DAYS <= value.duration_days <= MAX_ANNUAL_DAYS:
            continue
        annual.setdefault(fact.concept, []).append(value)
    for values in annual.values():
        values.sort(key=lambda value: (value.period_end, value.available_at))
    return annual


def _quarter_series(
    selected: tuple[FundamentalFact, ...],
) -> dict[str, list[FundamentalValue]]:
    direct: dict[tuple[str, date], FundamentalValue] = {}
    duration_facts: dict[tuple[str, str, str], list[FundamentalFact]] = {}
    for fact in selected:
        if fact.period_type != FundamentalFact.PeriodType.DURATION:
            continue
        value = _value_from_fact(fact)
        if value is None:
            continue
        if MIN_QUARTER_DAYS <= value.duration_days <= MAX_QUARTER_DAYS:
            direct[(fact.concept, fact.period_end)] = value
        duration_facts.setdefault(
            (fact.concept, fact.source_concept, fact.unit),
            [],
        ).append(fact)

    derived: dict[tuple[str, date], FundamentalValue] = {}
    for (concept, _source_concept, _unit), facts in duration_facts.items():
        if concept not in ADDITIVE_FLOW_CONCEPTS | WEIGHTED_AVERAGE_CONCEPTS:
            continue
        by_start: dict[date, list[FundamentalFact]] = {}
        for fact in facts:
            if fact.period_start is not None:
                by_start.setdefault(fact.period_start, []).append(fact)
        for same_start in by_start.values():
            same_start.sort(key=lambda fact: fact.period_end)
            for previous, current in zip(same_start, same_start[1:], strict=False):
                quarter_start = previous.period_end + timedelta(days=1)
                quarter_days = (current.period_end - quarter_start).days + 1
                if not MIN_QUARTER_DAYS <= quarter_days <= MAX_QUARTER_DAYS:
                    continue
                value = _subtract_ytd(
                    concept=concept,
                    previous=previous,
                    current=current,
                    quarter_start=quarter_start,
                )
                if value is not None:
                    derived[(concept, current.period_end)] = value

    combined = dict(direct)
    for key, derived_value in derived.items():
        direct_value = combined.get(key)
        if direct_value is None or derived_value.available_at > direct_value.available_at:
            combined[key] = derived_value
    quarters: dict[str, list[FundamentalValue]] = {}
    for (concept, _period_end), value in combined.items():
        quarters.setdefault(concept, []).append(value)
    for values in quarters.values():
        values.sort(key=lambda value: (value.period_end, value.available_at))
    return quarters


def _subtract_ytd(
    *,
    concept: str,
    previous: FundamentalFact,
    current: FundamentalFact,
    quarter_start: date,
) -> FundamentalValue | None:
    if previous.period_start is None or current.period_start is None:
        return None
    quarter_days = (current.period_end - quarter_start).days + 1
    if concept in ADDITIVE_FLOW_CONCEPTS:
        value = current.value - previous.value
        derivation = "ytd_difference"
    elif concept in WEIGHTED_AVERAGE_CONCEPTS:
        current_days = (current.period_end - current.period_start).days + 1
        previous_days = (previous.period_end - previous.period_start).days + 1
        weighted_total = current.value * current_days - previous.value * previous_days
        value = weighted_total / Decimal(quarter_days)
        if value <= 0:
            return None
        derivation = "weighted_ytd_difference"
    else:
        return None
    return FundamentalValue(
        concept=concept,
        value=value,
        unit=current.unit,
        period_start=quarter_start,
        period_end=current.period_end,
        available_at=max(previous.available_at, current.available_at),
        source_fact_ids=(str(previous.pk), str(current.pk)),
        accessions=tuple(dict.fromkeys((previous.accession, current.accession))),
        source_concepts=tuple(dict.fromkeys((previous.source_concept, current.source_concept))),
        derivation=derivation,
    )


def _ttm_series(
    quarters: dict[str, list[FundamentalValue]],
) -> dict[str, FundamentalValue]:
    result: dict[str, FundamentalValue] = {}
    for concept, values in quarters.items():
        if concept not in ADDITIVE_FLOW_CONCEPTS | WEIGHTED_AVERAGE_CONCEPTS:
            continue
        if len(values) < 4:
            continue
        window = values[-4:]
        if not _quarters_contiguous(window):
            continue
        if len({value.unit for value in window}) != 1:
            continue
        if len({value.source_concepts for value in window}) != 1:
            continue
        span_days = (window[-1].period_end - window[0].period_start).days + 1
        if not MIN_ANNUAL_DAYS <= span_days <= MAX_ANNUAL_DAYS:
            continue
        result[concept] = _ttm_value(concept, window)
    _add_free_cash_flow_value(result)
    return result


def _quarters_contiguous(values: list[FundamentalValue]) -> bool:
    for previous, current in zip(values, values[1:], strict=False):
        gap = (current.period_start - previous.period_end).days - 1
        if gap != 0:
            return False
    return True


def _v3_alias_quarter_candidates(
    selected: tuple[FundamentalFact, ...],
    *,
    concept: str,
) -> dict[date, list[FundamentalValue]]:
    """Retain *every* quarter observation for one alias, keyed by quarter end.

    Legacy `_quarter_series` collapses a directly reported quarter and a
    YTD-derived quarter for the same period end by ``available_at`` alone,
    and it overwrites same-end direct observations that differ only by
    period start. Both decisions happen before any controlling-fact ranking,
    so a direct revision 1 can beat a derived observation controlled by
    revision 5 whenever their availabilities happen to tie.

    This `us-sec-long-v3`-only surface keeps the alternatives instead. The
    period-construction rules -- quarter-length bounds, YTD chaining by
    identical period start, the additive/weighted derivation split -- are the
    frozen ones; only the collapsing is deferred to
    `_v3_select_quarter_observation`. `_quarter_series` itself is untouched.
    """
    candidates: dict[date, list[FundamentalValue]] = {}
    duration_facts: dict[tuple[str, str], list[FundamentalFact]] = {}
    for fact in selected:
        if fact.concept != concept:
            continue
        if fact.period_type != FundamentalFact.PeriodType.DURATION:
            continue
        value = _value_from_fact(fact)
        if value is None:
            continue
        if MIN_QUARTER_DAYS <= value.duration_days <= MAX_QUARTER_DAYS:
            candidates.setdefault(fact.period_end, []).append(value)
        duration_facts.setdefault((fact.source_concept, fact.unit), []).append(fact)
    if concept not in ADDITIVE_FLOW_CONCEPTS | WEIGHTED_AVERAGE_CONCEPTS:
        return candidates
    for facts in duration_facts.values():
        by_start: dict[date, list[FundamentalFact]] = {}
        for fact in facts:
            if fact.period_start is not None:
                by_start.setdefault(fact.period_start, []).append(fact)
        for same_start in by_start.values():
            same_start.sort(key=lambda fact: fact.period_end)
            for previous, current in zip(same_start, same_start[1:], strict=False):
                quarter_start = previous.period_end + timedelta(days=1)
                quarter_days = (current.period_end - quarter_start).days + 1
                if not MIN_QUARTER_DAYS <= quarter_days <= MAX_QUARTER_DAYS:
                    continue
                derived = _subtract_ytd(
                    concept=concept,
                    previous=previous,
                    current=current,
                    quarter_start=quarter_start,
                )
                if derived is not None:
                    candidates.setdefault(current.period_end, []).append(derived)
    return candidates


def _v3_quarter_observation_rank(
    value: FundamentalValue,
    *,
    fact_map: dict[str, FundamentalFact],
    priority: dict[tuple[str, str], int],
) -> tuple[int, tuple[datetime, int, int, str], str]:
    """Rank one quarter observation by its single real controlling filing.

    The leading flag keeps an observation whose source facts cannot be read
    back below every readable one: an unreadable lineage can never be named,
    so it must never win. The controlling filing is then compared under the
    complete frozen `_fact_rank` order -- availability, source revision,
    declared alias priority, accession -- rather than availability alone.

    The final term is the observation's own stable identity. It only ever
    separates two observations that are indistinguishable on every ranked
    evidence field, and it is derived from period/alias/accession/value, so
    reassigning row UUIDs cannot move it.
    """
    controlling = _controlling_source_fact(value, fact_map=fact_map, priority=priority)
    identity = observation_selection_identity(value)
    if controlling is None:
        return (0, (datetime.min.replace(tzinfo=UTC), 0, -10_000, ""), identity)
    return (1, _fact_rank(controlling, priority), identity)


def _v3_select_quarter_observation(
    values: list[FundamentalValue],
    *,
    fact_map: dict[str, FundamentalFact],
    priority: dict[tuple[str, str], int],
) -> FundamentalValue:
    return max(
        values,
        key=lambda value: _v3_quarter_observation_rank(
            value,
            fact_map=fact_map,
            priority=priority,
        ),
    )


def _v3_alias_quarter_series(
    selected: tuple[FundamentalFact, ...],
    *,
    examined: Iterable[FundamentalFact],
    concept: str,
    fact_map: dict[str, FundamentalFact],
    priority: dict[tuple[str, str], int],
) -> _AliasQuarterAssessment:
    """Assess one source alias's quarter series, keeping its full lineage.

    The returned ``values`` are the selected observation per quarter end, as
    before. ``assessed_fact_ids`` is the complete set of facts that
    established the assessment: every fact ``examined`` for this alias, the
    same-quarter candidates that lost the controlling-fact ranking, and every
    quarter of an alias that goes on to lose the newest-quarter anchor or to
    be found incomplete. An alias whose facts never produced a candidate at
    all still names them.

    Without that set, a payload could state that an alias is "stale but
    complete" or that its tail is incomplete while naming none of the
    evidence that decided it. Those facts are assessed, never selected: they
    do not enter any arithmetic, but the manifest must still be able to prove
    every one of them.
    """
    candidates = _v3_alias_quarter_candidates(selected, concept=concept)
    values = [
        _v3_select_quarter_observation(
            candidates[period_end],
            fact_map=fact_map,
            priority=priority,
        )
        for period_end in sorted(candidates)
    ]
    assessed_fact_ids = _dedupe_fact_ids(
        (
            # Every fact this alias's assessment read, including the ones
            # rejected before any candidate could be constructed. Collecting
            # only from constructed candidates left an alias that yields no
            # usable quarter -- an annual-only alternate alias, or a pair
            # whose derivation was refused -- described with empty lineage.
            *(str(fact.pk) for fact in examined),
            *(
                fact_id
                for period_end in sorted(candidates)
                for candidate in candidates[period_end]
                for fact_id in candidate.source_fact_ids
            ),
        )
    )
    return _AliasQuarterAssessment(
        values=values,
        assessed_fact_ids=assessed_fact_ids,
        assessed_quarter_period_ends=tuple(sorted(candidates)),
    )


def _dedupe_fact_ids(fact_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fact_ids))


@dataclass(frozen=True, slots=True)
class _AliasQuarterAssessment:
    """One source alias's quarter series plus the evidence that assessed it.

    `us-sec-long-v3` only. ``values`` drives selection; the other two fields
    exist so a losing, stale-but-complete, or incomplete alias can be
    described together with every fact that established that description.
    """

    values: list[FundamentalValue]
    assessed_fact_ids: tuple[str, ...]
    assessed_quarter_period_ends: tuple[date, ...]


def _newest_quarter_anchored_ttm_series(
    facts: tuple[FundamentalFact, ...],
    *,
    config: SecFundamentalsConfig,
) -> tuple[dict[str, FundamentalValue], dict[str, dict[str, Any]]]:
    """Build TTM values anchored on the newest eligible quarter's alias.

    For each canonical TTM concept this finds the newest quarter end that any
    eligible source alias can supply, ranks the observations at exactly that
    quarter end under the existing deterministic revision/availability/
    source-priority rules, and then requires the *winning* alias to supply
    four contiguous compatible quarters spanning 350-380 days. A stale but
    complete alternative alias never wins over a newer restated newest-quarter
    observation, aliases are never stitched together across quarters, and
    there is no annual current-period fallback.

    Within one alias, a directly reported quarter and a YTD-derived quarter
    for the same period end -- and two direct observations whose period
    identities differ but whose ends coincide -- are both retained as
    candidates and resolved by the full rank of one real controlling source
    fact (`_v3_alias_quarter_series`), not by availability alone.
    """
    priority = source_concept_priority(config)
    fact_map = {str(fact.pk): fact for fact in facts}
    by_concept_alias: dict[str, dict[str, list[FundamentalFact]]] = {}
    for fact in canonical_fact_order(facts):
        if fact.provider != "sec" or (fact.concept, fact.source_concept) not in priority:
            continue
        if fact.period_type != FundamentalFact.PeriodType.DURATION:
            continue
        if fact.concept not in ADDITIVE_FLOW_CONCEPTS | WEIGHTED_AVERAGE_CONCEPTS:
            continue
        by_concept_alias.setdefault(fact.concept, {}).setdefault(fact.source_concept, []).append(
            fact
        )

    result: dict[str, FundamentalValue] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for concept in sorted(by_concept_alias):
        alias_quarters: dict[str, list[FundamentalValue]] = {}
        alias_assessments: dict[str, _AliasQuarterAssessment] = {}
        for alias, alias_facts in sorted(by_concept_alias[concept].items()):
            selected = select_latest_fact_vintages(alias_facts, config=config)
            assessment = _v3_alias_quarter_series(
                selected,
                examined=alias_facts,
                concept=concept,
                fact_map=fact_map,
                priority=priority,
            )
            # Retained even when the alias yields no usable quarter: the
            # facts that proved it unusable are still assessed evidence.
            alias_assessments[alias] = assessment
            if assessment.values:
                alias_quarters[alias] = assessment.values
        entry: dict[str, Any] = {
            "policy": TTM_SELECTION_NEWEST_QUARTER_ALIAS,
            "concept": concept,
        }
        # Every fact any alias-tail assessment for this concept read, whether
        # its alias won, lost, was stale-but-complete, was incomplete, or
        # produced nothing at all.
        # Aliases that produced no usable quarter at all never reach
        # ``alias_candidates`` below, so their facts are recorded here.
        # Aliases that *do* appear carry their own complete lineage on their
        # candidate entry; repeating it at concept level would only duplicate
        # identifiers.
        entry["unusable_alias_source_fact_ids"] = list(
            _dedupe_fact_ids(
                fact_id
                for alias in sorted(alias_assessments)
                if alias not in alias_quarters
                for fact_id in alias_assessments[alias].assessed_fact_ids
            )
        )
        if not alias_quarters:
            entry.update(
                {
                    "status": "withheld",
                    "reason": "No eligible discrete quarter observation for any source alias",
                    "newest_quarter_end": None,
                    "selected_source_concept": None,
                    "controlling_source_fact": None,
                    "alias_candidates": [],
                    "stale_complete_alternatives": [],
                    "selected_quarter_period_ends": [],
                    "selected_quarter_lineage": [],
                    "homogeneous_four_quarter_tail": False,
                }
            )
            provenance[concept] = entry
            continue

        newest_quarter_end = max(values[-1].period_end for values in alias_quarters.values())
        candidates: list[dict[str, Any]] = []
        anchor_alias: str | None = None
        anchor_rank: tuple[datetime, int, int, str] | None = None
        anchor_controlling: dict[str, Any] | None = None
        for alias in sorted(alias_quarters):
            values = alias_quarters[alias]
            tail = _homogeneous_four_quarter_tail(values, through=values[-1].period_end)
            at_newest = [value for value in values if value.period_end == newest_quarter_end]
            assessment = alias_assessments[alias]
            candidate: dict[str, Any] = {
                "source_concept": alias,
                "newest_quarter_end": values[-1].period_end.isoformat(),
                "quarter_count": len(values),
                "has_homogeneous_four_quarter_tail": tail is not None,
                "source_priority_rank": priority.get((concept, alias)),
                "observes_newest_quarter": bool(at_newest),
                "controlling_source_fact": None,
                "newest_quarter_derivation": None,
                "newest_quarter_source_fact_ids": [],
                # Complete lineage of this alias's own tail assessment. It is
                # recorded for every alias, not only the anchor, so a stale
                # complete alternative or an incomplete losing tail can never
                # be described without the facts that established it. The
                # tail is identified by its period ends rather than by a
                # second copy of the same identifiers.
                "assessed_quarter_period_ends": [
                    period_end.isoformat() for period_end in assessment.assessed_quarter_period_ends
                ],
                "assessed_source_fact_ids": list(assessment.assessed_fact_ids),
                "assessed_tail_period_ends": (
                    [value.period_end.isoformat() for value in tail] if tail is not None else []
                ),
            }
            if at_newest:
                observation = at_newest[-1]
                controlling = _controlling_source_fact(
                    observation,
                    fact_map=fact_map,
                    priority=priority,
                )
                candidate["newest_quarter_derivation"] = observation.derivation
                candidate["newest_quarter_source_fact_ids"] = list(observation.source_fact_ids)
                if controlling is None:
                    # The observation cites no readable source fact, so no
                    # real filing can be named as its controlling lineage.
                    # It is never allowed to anchor selection.
                    candidate["rankable"] = False
                    candidates.append(candidate)
                    continue
                controlling_payload = _source_fact_payload(controlling)
                candidate["rankable"] = True
                candidate["controlling_source_fact"] = controlling_payload
                candidate["newest_quarter_available_at"] = controlling.available_at.isoformat()
                candidate["newest_quarter_source_revision"] = controlling.source_revision
                rank = _fact_rank(controlling, priority)
                if anchor_rank is None or rank > anchor_rank:
                    anchor_alias, anchor_rank, anchor_controlling = (
                        alias,
                        rank,
                        controlling_payload,
                    )
            candidates.append(candidate)
        entry["newest_quarter_end"] = newest_quarter_end.isoformat()
        entry["alias_candidates"] = candidates
        entry["stale_complete_alternatives"] = [
            candidate["source_concept"]
            for candidate in candidates
            if candidate["has_homogeneous_four_quarter_tail"]
            and candidate["source_concept"] != anchor_alias
        ]
        entry["selected_source_concept"] = anchor_alias
        entry["controlling_source_fact"] = anchor_controlling
        if anchor_alias is None:
            entry.update(
                {
                    "status": "withheld",
                    "reason": (
                        "No source alias observes the newest eligible quarter "
                        f"{newest_quarter_end.isoformat()} through a readable "
                        "controlling source fact"
                    ),
                    "selected_quarter_period_ends": [],
                    "selected_quarter_lineage": [],
                    "homogeneous_four_quarter_tail": False,
                }
            )
            provenance[concept] = entry
            continue
        window = _homogeneous_four_quarter_tail(
            alias_quarters[anchor_alias],
            through=newest_quarter_end,
        )
        if window is None:
            entry.update(
                {
                    "status": "withheld",
                    "reason": (
                        f"Source alias {anchor_alias!r} does not supply four contiguous "
                        f"compatible quarters through {newest_quarter_end.isoformat()}"
                    ),
                    "selected_quarter_period_ends": [],
                    "selected_quarter_lineage": [],
                    "homogeneous_four_quarter_tail": False,
                }
            )
            provenance[concept] = entry
            continue
        entry.update(
            {
                "status": "ttm_available",
                "reason": "",
                "selected_quarter_period_ends": [value.period_end.isoformat() for value in window],
                "selected_quarter_lineage": [
                    _quarter_lineage_payload(value, fact_map=fact_map, priority=priority)
                    for value in window
                ],
                "homogeneous_four_quarter_tail": True,
                "span_days": (window[-1].period_end - window[0].period_start).days + 1,
            }
        )
        provenance[concept] = entry
        result[concept] = _ttm_value(concept, window)
    _add_free_cash_flow_value(result)
    return result, provenance


def _homogeneous_four_quarter_tail(
    values: list[FundamentalValue],
    *,
    through: date,
) -> list[FundamentalValue] | None:
    """Return the four contiguous compatible quarters ending at ``through``.

    Returns ``None`` -- never a shorter, stitched, or annual substitute --
    whenever the window is incomplete, non-contiguous, mixed-unit,
    mixed-alias, or outside the 350-380 day span.
    """
    eligible = [value for value in values if value.period_end <= through]
    if len(eligible) < 4:
        return None
    window = eligible[-4:]
    if window[-1].period_end != through:
        return None
    if not _quarters_contiguous(window):
        return None
    if len({value.unit for value in window}) != 1:
        return None
    if len({value.source_concepts for value in window}) != 1:
        return None
    span_days = (window[-1].period_end - window[0].period_start).days + 1
    if not MIN_ANNUAL_DAYS <= span_days <= MAX_ANNUAL_DAYS:
        return None
    return window


def _controlling_source_fact(
    value: FundamentalValue,
    *,
    fact_map: dict[str, FundamentalFact],
    priority: dict[tuple[str, str], int],
) -> FundamentalFact | None:
    """Return the one real source fact that controls ``value``'s rank.

    A direct quarter has exactly one dependency. A YTD-derived quarter has
    two and only becomes knowable once *both* are available, so the
    dependency that actually gates it is the one that ranks highest under
    the frozen lexicographic `_fact_rank` order (availability, then
    revision, then source priority, then accession).

    Returning a single filed observation -- rather than independently
    maximizing availability, revision, and accession across dependencies --
    means every reported rank field belongs to one real filing. A synthesized
    rank could otherwise claim a revision from one dependency and an
    availability from another, describing a vintage that was never filed.
    """
    facts = [fact_map[fact_id] for fact_id in value.source_fact_ids if fact_id in fact_map]
    if not facts:
        return None
    return max(facts, key=lambda fact: _fact_rank(fact, priority))


def _source_fact_payload(fact: FundamentalFact) -> dict[str, Any]:
    return {
        "fact_id": str(fact.pk),
        "source_concept": fact.source_concept,
        "accession": fact.accession,
        "source_revision": fact.source_revision,
        "available_at": fact.available_at.isoformat(),
        "period_identity": fact.period_identity,
        "period_start": (fact.period_start.isoformat() if fact.period_start is not None else None),
        "period_end": fact.period_end.isoformat(),
        "unit": fact.unit,
    }


def _quarter_lineage_payload(
    value: FundamentalValue,
    *,
    fact_map: dict[str, FundamentalFact],
    priority: dict[tuple[str, str], int],
) -> dict[str, Any]:
    """Reference the controlling filing behind one selected quarter.

    Only identifiers are recorded here. Every fact cited by a selected TTM
    window is already described in full in the forecast's ``input_facts``,
    and the ranking-decisive newest-quarter filing is described in full under
    ``controlling_source_fact``, so repeating each filing's fields per
    quarter would inflate an immutable payload without adding evidence.
    """
    controlling = _controlling_source_fact(value, fact_map=fact_map, priority=priority)
    return {
        "period_start": value.period_start.isoformat(),
        "period_end": value.period_end.isoformat(),
        "derivation": value.derivation,
        "source_fact_ids": list(value.source_fact_ids),
        "controlling_source_fact_id": (str(controlling.pk) if controlling is not None else None),
    }


def _ttm_value(concept: str, window: list[FundamentalValue]) -> FundamentalValue:
    if concept in ADDITIVE_FLOW_CONCEPTS:
        value = sum((item.value for item in window), Decimal("0"))
        derivation = "sum_four_contiguous_quarters"
    else:
        total_days = sum(item.duration_days for item in window)
        value = sum((item.value * item.duration_days for item in window), Decimal("0")) / Decimal(
            total_days
        )
        derivation = "weighted_four_contiguous_quarters"
    return FundamentalValue(
        concept=concept,
        value=value,
        unit=window[-1].unit,
        period_start=window[0].period_start,
        period_end=window[-1].period_end,
        available_at=max(item.available_at for item in window),
        source_fact_ids=tuple(
            dict.fromkeys(fact_id for item in window for fact_id in item.source_fact_ids)
        ),
        accessions=tuple(
            dict.fromkeys(accession for item in window for accession in item.accessions)
        ),
        source_concepts=window[-1].source_concepts,
        derivation=derivation,
    )


def _add_free_cash_flow_series(
    series: dict[str, list[FundamentalValue]],
) -> None:
    operating = {
        (value.period_start, value.period_end): value
        for value in series.get("operating_cash_flow", [])
    }
    capex = {
        (value.period_start, value.period_end): value
        for value in series.get("capital_expenditure", [])
    }
    derived: list[FundamentalValue] = []
    for period, operating_value in operating.items():
        capex_value = capex.get(period)
        if capex_value is None or capex_value.unit != operating_value.unit:
            continue
        derived.append(_free_cash_flow(operating_value, capex_value))
    if derived:
        series["free_cash_flow"] = sorted(
            derived,
            key=lambda value: (value.period_end, value.available_at),
        )


def _add_free_cash_flow_value(values: dict[str, FundamentalValue]) -> None:
    operating = values.get("operating_cash_flow")
    capex = values.get("capital_expenditure")
    if (
        operating is None
        or capex is None
        or operating.period_start != capex.period_start
        or operating.period_end != capex.period_end
        or operating.unit != capex.unit
    ):
        return
    values["free_cash_flow"] = _free_cash_flow(operating, capex)


def _free_cash_flow(
    operating: FundamentalValue,
    capex: FundamentalValue,
) -> FundamentalValue:
    return FundamentalValue(
        concept="free_cash_flow",
        value=operating.value - abs(capex.value),
        unit=operating.unit,
        period_start=operating.period_start,
        period_end=operating.period_end,
        available_at=max(operating.available_at, capex.available_at),
        source_fact_ids=tuple(dict.fromkeys((*operating.source_fact_ids, *capex.source_fact_ids))),
        accessions=tuple(dict.fromkeys((*operating.accessions, *capex.accessions))),
        source_concepts=tuple(dict.fromkeys((*operating.source_concepts, *capex.source_concepts))),
        derivation="operating_cash_flow_minus_absolute_capex",
    )


def _instant_series(
    selected: tuple[FundamentalFact, ...],
) -> dict[str, list[FundamentalFact]]:
    result: dict[str, list[FundamentalFact]] = {}
    for fact in selected:
        if fact.period_type != FundamentalFact.PeriodType.INSTANT:
            continue
        result.setdefault(fact.concept, []).append(fact)
    for facts in result.values():
        facts.sort(key=lambda fact: (fact.period_end, fact.available_at))
    return result


def _value_from_fact(fact: FundamentalFact) -> FundamentalValue | None:
    if fact.period_start is None:
        return None
    return FundamentalValue(
        concept=fact.concept,
        value=fact.value,
        unit=fact.unit,
        period_start=fact.period_start,
        period_end=fact.period_end,
        available_at=fact.available_at,
        source_fact_ids=(str(fact.pk),),
        accessions=(fact.accession,),
        source_concepts=(fact.source_concept,),
        derivation="reported",
    )
