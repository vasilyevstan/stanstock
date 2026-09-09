"""`us-under10-shadow-v1`: an unactivated Under-$10 diagnostic assessment.

This module is a pure computation. It performs no ORM query, no asset read,
no provider call, and no network access: every input arrives already resolved
through the caller's point-in-time controls. It may read scalar fields of
already-loaded model instances and their already-selected ``source_asset``
relation, and it must never trigger a lazy query.

What the policy is
------------------
A *shadow* record attached to a newly created analysis whose decision-run USD
reference close falls in the Under-$10 band. It changes no score, confidence,
recommendation, gate, scenario, forecast, prediction, outcome, opportunity,
basket, or allocation. `UNDER10_ACTIVATED` is ``False`` and
``activation_eligible`` is ``false`` in every branch, because the mandatory
verified split/reverse-split evidence gate cannot pass for any provider in
this version.

What it deliberately refuses to do
----------------------------------
* A missing debt component is missing, never zero.
* An unusable value is never presented as a usable input, but its immutable
  fact reference stays in the assessed evidence so the refusal is auditable.
* Adverse and missing are different answers: states 2 and 3 require complete
  inputs, and `SOLVENCY_INSUFFICIENT_EVIDENCE` is never read as adverse.
* A non-negative free cash flow produces a *not applicable* runway, never
  zero and never infinity.
* Liquidity carries no threshold and no pass/fail conclusion; its volume
  basis stays explicitly unverified even when a number is computed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Any

import polars as pl

from stanstock.data.models import DataAsset, FundamentalFact
from stanstock.data.provider_policy import (
    CAPABILITY_UNAVAILABLE,
    SEC_PROVIDER,
    SPLIT_EVENT_CAPABILITY,
    normalized_provider_plan,
    split_event_capability,
)
from stanstock.data.sec_config import SecFundamentalsConfig
from stanstock.data.sec_fundamentals import (
    TTM_SELECTION_LEGACY,
    FundamentalValue,
    SecFundamentalSeries,
    build_sec_fundamental_series,
    partition_unproven_corrections,
    select_latest_fact_vintages,
)
from stanstock.research.affordability import (
    DECISION_TARGET_DATE_BASIS,
    PRICE_BAND_CURRENCY,
    PRICE_BAND_POLICY_VERSION,
    UNDER_10_BAND,
)
from stanstock.research.indicators import (
    DOLLAR_VOLUME_COMPUTED,
    DOLLAR_VOLUME_INVALID_SESSION_DATES,
    DollarVolumeResult,
    median_dollar_volume,
)

# ---------------------------------------------------------------------------
# Frozen policy constants
# ---------------------------------------------------------------------------

UNDER10_SHADOW_POLICY_VERSION = "us-under10-shadow-v1"
UNDER10_SCHEMA_VERSION = 1
UNDER10_LIQUIDITY_SESSIONS = 252
UNDER10_MAX_METRIC_AGE_DAYS = 200
UNDER10_MAX_PRICE_STALENESS_DAYS = 7
UNDER10_MIN_RUNWAY_QUARTERS = Decimal("4")
UNDER10_ACTIVATED = False

#: Whether *any* branch of this policy version can make a candidate eligible
#: for activation. Always `False` in `us-under10-shadow-v1`: the mandatory
#: verified split/reverse-split evidence gate cannot pass for any provider.
#: Like `UNDER10_ACTIVATED` and `UNDER10_NEW_ALLOCATION_PERCENT`, a reader
#: must treat this as authoritative policy, never as a value a stored
#: payload's own `activation_eligible`/`activated` keys could override.
UNDER10_ACTIVATION_ELIGIBLE = False

#: The only authoritative Under-$10 new-allocation figure in this version.
#: Never read from a stored payload: a stored assessment's own
#: ``new_allocation_percent`` key is a recomputable mirror of this constant,
#: not an independent source of truth, so no reader may trust it instead.
UNDER10_NEW_ALLOCATION_PERCENT = 0

#: Canonical, ordered fixed gate names for the v1 payload. The values are
#: always the literal boolean ``False``; a reader must not accept numeric
#: zero (or any other falsey value) as an equivalent gate result.
UNDER10_GATE_KEYS = (
    "solvency_obligation",
    "dollar_liquidity_252",
    "verified_split_evidence",
)

#: Exact field set emitted for every immutable asset reference in this
#: payload, shared with the reader so generation and validation cannot drift.
UNDER10_ASSET_REFERENCE_KEYS = ("id", "sha256")

UNDER10_CONCEPTS = (
    "cash_and_equivalents",
    "short_term_debt",
    "current_long_term_debt",
    "current_assets",
    "current_liabilities",
    "operating_cash_flow",
    "capital_expenditure",
)

#: Balance-sheet inputs that must all share exactly one ``period_end``.
UNDER10_INSTANT_CONCEPTS = (
    "cash_and_equivalents",
    "short_term_debt",
    "current_long_term_debt",
    "current_assets",
    "current_liabilities",
)

#: Flow inputs the canonical builder combines into free cash flow.
UNDER10_DURATION_CONCEPTS = (
    "operating_cash_flow",
    "capital_expenditure",
)

#: Every decision operand must be reported in this unit.
UNDER10_REQUIRED_UNIT = "USD"

#: `adjust=splits` is proven for *prices*; no provider proves a matching
#: volume adjustment, so the dollar-volume basis stays explicitly unverified
#: even when the median is computed.
UNDER10_VOLUME_BASIS = "provider_reported_unverified_split_basis"
UNDER10_LIQUIDITY_METRIC = "median_dollar_volume_252_sessions"

UNDER10_REQUIRED_INTERVAL = "1day"
UNDER10_REQUIRED_ADJUSTMENT = "splits"
UNDER10_REQUIRED_RETURN_DEFINITION = "split_adjusted_price_return"

#: Reported ratio/runway precision. State decisions never read these.
REPORTED_PLACES = Decimal("0.0001")
#: `FundamentalFact.value` precision, preserved by every monetary string.
MONETARY_PLACES = Decimal("0.00000001")
#: `StockAnalysis.current_price` precision.
REFERENCE_CLOSE_PLACES = Decimal("0.000001")

# ---------------------------------------------------------------------------
# Solvency states, runway states, and reasons
# ---------------------------------------------------------------------------

SOLVENCY_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION = "adverse_near_term_obligation"
SOLVENCY_ELEVATED_OBLIGATION_RISK = "elevated_obligation_risk"
SOLVENCY_NO_ADVERSE_EVIDENCE = "no_adverse_evidence_observed"

SOLVENCY_STATES = (
    SOLVENCY_INSUFFICIENT_EVIDENCE,
    SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION,
    SOLVENCY_ELEVATED_OBLIGATION_RISK,
    SOLVENCY_NO_ADVERSE_EVIDENCE,
)

RUNWAY_COMPUTED = "computed"
RUNWAY_NOT_APPLICABLE = "not_applicable_positive_fcf"
RUNWAY_WITHHELD = "withheld"

LIQUIDITY_COMPUTED = "computed"
LIQUIDITY_WITHHELD = "withheld"

#: Insufficiency reasons. Missing, incompatible/stale, and adverse evidence
#: stay distinguishable; nothing collapses into a generic failure.
REASON_EVIDENCE_NOT_CUTOFF_SAFE = "evidence_not_cutoff_safe"
REASON_COMPANY_IDENTITY_UNAVAILABLE = "company_identity_unavailable"
REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING = "near_term_debt_components_missing"
REASON_INSTANT_PERIOD_MISMATCH = "instant_period_mismatch"
REASON_STALE_METRIC = "stale_metric"
REASON_FUTURE_PERIOD_END = "future_period_end"
REASON_NONPOSITIVE_CURRENT_LIABILITIES = "nonpositive_current_liabilities"
REASON_NONFINITE_INPUT = "nonfinite_input"
REASON_INCOMPATIBLE_UNIT = "incompatible_unit"
REASON_FREE_CASH_FLOW_MISSING = "free_cash_flow_missing"

#: Adverse predicates, reported so a state is explainable without re-deriving
#: it from the inputs.
REASON_NEAR_TERM_DEBT_EXCEEDS_CASH = "near_term_debt_exceeds_cash"
REASON_CURRENT_ASSETS_BELOW_LIABILITIES = "current_assets_below_current_liabilities"
REASON_NEGATIVE_FREE_CASH_FLOW = "negative_free_cash_flow"
REASON_RUNWAY_BELOW_MINIMUM_QUARTERS = "cash_runway_below_minimum_quarters"

#: Liquidity refusals owned by the assessment builder rather than the
#: indicator primitive.
LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE = "price_provenance_unavailable"
LIQUIDITY_BASIS_INCOMPATIBLE = "basis_incompatible"
LIQUIDITY_STALE_PRICE_EVIDENCE = "stale_price_evidence"
LIQUIDITY_FUTURE_PRICE_SESSION = "future_price_session"

#: Arithmetic context for the state decision. Wide enough for the model's
#: 32-digit values and their sums; half-even so a reported figure and a
#: decision never disagree about a tie.
_DECIMAL_PRECISION = 64


def _missing_reason(concept: str) -> str:
    return f"{concept}_missing"


def _unusable_reason(concept: str) -> str:
    return f"{concept}_not_usable"


def under10_inactive_gates() -> dict[str, bool]:
    """Return a fresh canonical v1 gate document (all gates exactly false)."""
    return dict.fromkeys(UNDER10_GATE_KEYS, False)


def under10_blocking_reasons(split_reason: str) -> list[str]:
    """Return the canonical v1 blocker list for a validated split refusal."""
    return [split_reason]


# ---------------------------------------------------------------------------
# Policy identity
# ---------------------------------------------------------------------------


def under10_policy_document(sec_config: SecFundamentalsConfig) -> dict[str, Any]:
    """The complete canonical description this policy version is hashed from.

    It binds the reviewed SEC fundamentals configuration's own effective
    hash, so a change to the canonical concept taxonomy is a change to this
    policy's identity even though no constant here moved.
    """
    return {
        "policy_version": UNDER10_SHADOW_POLICY_VERSION,
        "schema_version": UNDER10_SCHEMA_VERSION,
        "activated": UNDER10_ACTIVATED,
        "shadow_only": True,
        "activation_eligible": UNDER10_ACTIVATION_ELIGIBLE,
        "new_allocation_percent": UNDER10_NEW_ALLOCATION_PERCENT,
        "band": {
            "price_band": UNDER_10_BAND,
            "price_band_policy_version": PRICE_BAND_POLICY_VERSION,
            "currency": PRICE_BAND_CURRENCY,
            "date_basis": DECISION_TARGET_DATE_BASIS,
        },
        "identity": {
            # The complete candidate-identity contract every payload's
            # `evaluated_for` carries and every reader binds exactly: the
            # permanent `Listing.id`, the decision run's own target date
            # and data cutoff, the persisted decision-run reference
            # close/currency, and
            # (in `liquidity.price_asset`) the immutable price asset's
            # UUID and content checksum. None of these alone -- not even
            # the price asset -- is sufficient without the others; two
            # genuine, unrelated candidates can otherwise share every
            # other field on the same decision date.
            "listing_id_bound": True,
            "target_date_bound": True,
            "data_cutoff_bound_exact": True,
            "reference_close_bound": True,
            "currency_bound": True,
            "price_asset_uuid_and_checksum_bound": True,
        },
        "concepts": {
            "required": list(UNDER10_CONCEPTS),
            "instant": list(UNDER10_INSTANT_CONCEPTS),
            "duration": list(UNDER10_DURATION_CONCEPTS),
        },
        "selection": {
            "sec_evidence_qualification": {
                "all_conditions_required": True,
                "canonical_concepts": list(UNDER10_CONCEPTS),
                "fact_provider": SEC_PROVIDER,
                "source_asset_provider": SEC_PROVIDER,
            },
            "sec_fundamentals_config_version": sec_config.config_version,
            "sec_fundamentals_config_hash": sec_config.config_hash,
            "ttm_selection": TTM_SELECTION_LEGACY,
            "alias_instant_candidates": False,
            "correction_availability": "partition_unproven_corrections_at_data_cutoff",
            "free_cash_flow_selection": "ttm_then_latest_compatible_annual",
            "free_cash_flow_derivation": "operating_cash_flow_minus_absolute_capex",
            "instant_period_rule": "single_shared_period_end",
            "near_term_debt_rule": ("short_term_debt_plus_current_long_term_debt_both_required"),
            "missing_component_is_zero": False,
            "backward_search_for_common_date": False,
            "max_metric_age_days": UNDER10_MAX_METRIC_AGE_DAYS,
            "future_period_end_admitted": False,
            "required_unit": UNDER10_REQUIRED_UNIT,
            "on_time_asset_cutoff_rule": "available_at_then_retrieved_at",
        },
        "states": [
            {
                "order": 1,
                "state": SOLVENCY_INSUFFICIENT_EVIDENCE,
                "rule": (
                    "any required input missing, unusable, incompatible, stale, "
                    "future-dated, or not eligible under the applicable cutoff rules"
                ),
            },
            {
                "order": 2,
                "state": SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION,
                "rule": (
                    "near_term_debt > cash AND (current_assets < current_liabilities "
                    "OR (free_cash_flow < 0 AND 4*cash < "
                    "minimum_runway_quarters*abs(free_cash_flow)))"
                ),
            },
            {
                "order": 3,
                "state": SOLVENCY_ELEVATED_OBLIGATION_RISK,
                "rule": (
                    "free_cash_flow < 0 OR near_term_debt > cash "
                    "OR current_assets < current_liabilities"
                ),
            },
            {
                "order": 4,
                "state": SOLVENCY_NO_ADVERSE_EVIDENCE,
                "rule": "remaining complete-input cases",
            },
        ],
        "runway": {
            "minimum_quarters": str(UNDER10_MIN_RUNWAY_QUARTERS),
            "quarters_formula": "4 * cash / abs(free_cash_flow)",
            "boundary_comparison": ("4 * cash < minimum_quarters * abs(free_cash_flow)"),
            "non_negative_free_cash_flow_status": RUNWAY_NOT_APPLICABLE,
            "non_negative_free_cash_flow_quarters": None,
            "independent_of_other_obligation_inputs": True,
        },
        "liquidity": {
            "metric": UNDER10_LIQUIDITY_METRIC,
            "sessions": UNDER10_LIQUIDITY_SESSIONS,
            "currency": PRICE_BAND_CURRENCY,
            "volume_basis": UNDER10_VOLUME_BASIS,
            "required_interval": UNDER10_REQUIRED_INTERVAL,
            "required_adjustment": UNDER10_REQUIRED_ADJUSTMENT,
            "required_return_definition": UNDER10_REQUIRED_RETURN_DEFINITION,
            "max_price_staleness_days": UNDER10_MAX_PRICE_STALENESS_DAYS,
            "distinct_observed_sessions_only": True,
            "padding_or_replacement_permitted": False,
            "threshold": None,
        },
        "split_verification": {
            "capability": SPLIT_EVENT_CAPABILITY,
            "status": CAPABILITY_UNAVAILABLE,
            "inference_prohibited": True,
            "verified_branch_exists": False,
        },
        "gates": under10_inactive_gates(),
        "serialization": {
            "encoding": "utf-8",
            "sorted_keys": True,
            "separators": ",:",
            "allow_nan": False,
            "date_format": "iso-8601",
            "timestamp_format": "iso-8601",
            "monetary_decimal_places": 8,
            "reported_decimal_places": 4,
            "reference_close_decimal_places": 6,
            "rounding": "ROUND_HALF_EVEN",
            "decimal_precision": _DECIMAL_PRECISION,
            "assessment_hash_excludes": ["assessment_hash"],
        },
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    """UTF-8 canonical JSON: sorted keys, fixed separators, no NaN/Infinity."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def under10_policy_hash(sec_config: SecFundamentalsConfig) -> str:
    return hashlib.sha256(
        canonical_json(under10_policy_document(sec_config)).encode("utf-8")
    ).hexdigest()


def under10_assessment_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 over the complete payload minus only ``assessment_hash``.

    This is a recomputation checksum for a canonical output. It is not tamper
    protection and it does not make `StockAnalysis.data_quality` immutable.
    """
    return hashlib.sha256(
        canonical_json(
            {key: value for key, value in payload.items() if key != "assessment_hash"}
        ).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Resolved evidence
# ---------------------------------------------------------------------------


class _ResolvedInput:
    """One decision operand plus the reason it is unusable, if it is.

    ``value`` is populated only when the operand is genuinely usable, so a
    stale or non-finite observation can never leak into the payload's input
    block or into a comparison.
    """

    __slots__ = ("value", "reason")

    def __init__(self, value: Decimal | None = None, reason: str | None = None) -> None:
        self.value = value
        self.reason = reason

    @property
    def usable(self) -> bool:
        return self.value is not None and self.reason is None


def build_under10_assessment(
    *,
    facts: Sequence[FundamentalFact],
    sec_config: SecFundamentalsConfig,
    price_frame: pl.DataFrame,
    price_asset: DataAsset | None,
    price_source: Mapping[str, Any] | None,
    reference_close: Decimal,
    target_date: date,
    data_cutoff: datetime,
    code_revision_value: str,
    provider: str,
    provider_plan: str | None,
    evidence_cutoff_safe: bool,
    company_identity_present: bool,
    invalid_session_date_rows: int,
    listing_id: str,
) -> dict[str, Any]:
    """Build the complete `us-under10-shadow-v1` payload.

    ``invalid_session_date_rows`` has no default: it is a caller-provided
    fact about the price frame's own read (see
    `AsOfData.price_frame_with_diagnostics`), and a caller that forgot to
    resolve it must fail loudly rather than silently behave as though no
    session date was ever invalid.

    ``listing_id`` (the permanent `Listing.id` UUID, as a string) is the
    complete candidate identity this assessment was evaluated for. A
    matching run target date/cutoff/reference close/currency and even an
    identical immutable price asset are not, on their own, proof this
    payload belongs to *this* listing rather than an unrelated one sharing
    every one of those values on the same decision date; only the
    permanent listing id closes that gap, and the reader binds it exactly.
    """
    qualified = qualify_under10_sec_facts(facts)
    solvency = _build_solvency(
        facts=qualified,
        sec_config=sec_config,
        target_date=target_date,
        data_cutoff=data_cutoff,
        evidence_cutoff_safe=evidence_cutoff_safe,
        company_identity_present=company_identity_present,
    )
    liquidity = _build_liquidity(
        price_frame=price_frame,
        price_asset=price_asset,
        price_source=price_source,
        target_date=target_date,
        invalid_session_date_rows=invalid_session_date_rows,
    )
    split_status, split_reason = split_event_capability(provider, provider_plan)
    payload: dict[str, Any] = {
        "schema_version": UNDER10_SCHEMA_VERSION,
        "policy_version": UNDER10_SHADOW_POLICY_VERSION,
        "policy_hash": under10_policy_hash(sec_config),
        "assessment_hash": "",
        "activated": UNDER10_ACTIVATED,
        "shadow_only": True,
        "activation_eligible": UNDER10_ACTIVATION_ELIGIBLE,
        "code_revision": code_revision_value,
        "evaluated_for": {
            "listing_id": listing_id,
            "target_date": target_date.isoformat(),
            "data_cutoff": data_cutoff.isoformat(),
            "price_band": UNDER_10_BAND,
            "reference_close": _reference_close_text(reference_close),
            "date_basis": DECISION_TARGET_DATE_BASIS,
            "currency": PRICE_BAND_CURRENCY,
        },
        "solvency": solvency,
        "liquidity": liquidity,
        "split_verification": {
            "status": split_status,
            "reason": split_reason,
            "provider": provider,
            "plan_recorded": normalized_provider_plan(provider_plan) is not None,
            "capability": SPLIT_EVENT_CAPABILITY,
            "inference_prohibited": True,
        },
        # Fixed not-passed activation summaries. They are never data-driven,
        # never collapse a solvency state, and never become a decision input.
        "gates": under10_inactive_gates(),
        "blocking_reasons": under10_blocking_reasons(split_reason),
        "new_allocation_percent": UNDER10_NEW_ALLOCATION_PERCENT,
    }
    payload["assessment_hash"] = under10_assessment_hash(payload)
    return payload


def qualify_under10_sec_facts(
    facts: Sequence[FundamentalFact],
) -> list[FundamentalFact]:
    """Provider-qualified SEC facts, deduplicated in deterministic order.

    A fact qualifies only when its canonical concept belongs to this policy
    *and* both the fact row and its immutable source asset identify the
    authoritative SEC provider. A full-analysis run can therefore hand over
    every visible fact and be narrowed in memory without a second query.
    Provider-qualified rows later rejected on availability, compatibility,
    or sufficiency grounds remain in the assessed lineage; foreign or
    provider-mismatched rows were never SEC evidence and do not.

    The same immutable fact can legitimately appear more than once in the
    supplied sequence (e.g. two querysets concatenated by a caller). Keyed
    deduplication by primary key -- rather than trusting the caller never to
    repeat a row -- means repeated or reordered identical evidence always
    collapses to the same assessed set, so the payload and its checksum stay
    identical regardless of how the input was assembled.
    """
    seen: dict[str, FundamentalFact] = {}
    for fact in facts:
        if (
            fact.concept not in UNDER10_CONCEPTS
            or fact.provider != SEC_PROVIDER
            or fact.source_asset.provider != SEC_PROVIDER
        ):
            continue
        seen.setdefault(str(fact.pk), fact)
    return [seen[key] for key in sorted(seen)]


# ---------------------------------------------------------------------------
# Solvency
# ---------------------------------------------------------------------------


def _build_solvency(
    *,
    facts: Sequence[FundamentalFact],
    sec_config: SecFundamentalsConfig,
    target_date: date,
    data_cutoff: datetime,
    evidence_cutoff_safe: bool,
    company_identity_present: bool,
) -> dict[str, Any]:
    assessed_fact_ids = [str(fact.pk) for fact in facts]
    assessed_assets = _asset_references(facts)
    blocking: list[str] = []
    if not company_identity_present:
        blocking.append(REASON_COMPANY_IDENTITY_UNAVAILABLE)
    if not evidence_cutoff_safe:
        blocking.append(REASON_EVIDENCE_NOT_CUTOFF_SAFE)
    if blocking:
        return _insufficient_solvency(
            reasons=blocking,
            assessed_fact_ids=assessed_fact_ids,
            assessed_assets=assessed_assets,
        )

    # `build_sec_fundamental_series` performs unguarded Decimal arithmetic --
    # YTD subtraction, TTM summation, and the free-cash-flow subtraction
    # itself -- under whatever context is ambient when it runs. Entering the
    # policy's own precision-64 context here, rather than trusting the
    # caller's ambient default, is what keeps a near-cancelling subtraction
    # (e.g. two 21-digit operands eight decimal places apart) from silently
    # rounding to zero; `Infinity`/`NaN` operands are preflighted below so
    # they cannot raise inside that same unguarded arithmetic.
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        # A correction restating an already-filed accession is only usable
        # once some observation proves *when* the restated value became
        # knowable. A deferred correction leaves the proven prior vintage in
        # place and stays in the assessed evidence above.
        admitted, _deferred = partition_unproven_corrections(facts, available_through=data_cutoff)
        nonfinite_reason = _nonfinite_flow_operand_reason(admitted, sec_config=sec_config)
        if nonfinite_reason is not None:
            # A selected (latest) OCF/capex vintage that is `Infinity`/`NaN`
            # is withheld before the frozen builder ever combines it: filtering
            # it out of `admitted` instead would let the builder's own vintage
            # selection silently promote an older, more favorable observation.
            return _insufficient_solvency(
                reasons=[nonfinite_reason],
                assessed_fact_ids=assessed_fact_ids,
                assessed_assets=assessed_assets,
            )
        series = build_sec_fundamental_series(
            admitted,
            config=sec_config,
            ttm_selection=TTM_SELECTION_LEGACY,
            alias_instant_candidates=False,
        )

    reasons: list[str] = []
    instants, instant_date = _resolve_instants(series, target_date=target_date, reasons=reasons)
    fcf, periods = _resolve_free_cash_flow(series, target_date=target_date, reasons=reasons)

    cash = instants["cash_and_equivalents"]
    current_assets = instants["current_assets"]
    current_liabilities = instants["current_liabilities"]
    near_term_debt = _near_term_debt(
        instants["short_term_debt"],
        instants["current_long_term_debt"],
        reasons=reasons,
    )
    runway = _build_runway(cash=cash, fcf=fcf)
    current_ratio = _current_ratio(current_assets, current_liabilities)

    inputs = {
        "cash_and_equivalents": _monetary_text(cash),
        "near_term_debt": _monetary_text(near_term_debt),
        "current_assets": _monetary_text(current_assets),
        "current_liabilities": _monetary_text(current_liabilities),
        "current_ratio": current_ratio,
        "free_cash_flow": _monetary_text(fcf),
    }
    complete = all(
        candidate.usable
        for candidate in (cash, near_term_debt, current_assets, current_liabilities, fcf)
    )
    if not complete:
        return {
            "status": SOLVENCY_INSUFFICIENT_EVIDENCE,
            "reasons": _dedupe(reasons),
            "inputs": inputs,
            "periods": _periods_payload(instant_date, periods),
            "runway": runway,
            "assessed_fact_ids": assessed_fact_ids,
            "assessed_assets": assessed_assets,
        }
    status, adverse_reasons = _classify_solvency(
        cash=_require(cash),
        near_term_debt=_require(near_term_debt),
        current_assets=_require(current_assets),
        current_liabilities=_require(current_liabilities),
        free_cash_flow=_require(fcf),
    )
    return {
        "status": status,
        "reasons": adverse_reasons,
        "inputs": inputs,
        "periods": _periods_payload(instant_date, periods),
        "runway": runway,
        "assessed_fact_ids": assessed_fact_ids,
        "assessed_assets": assessed_assets,
    }


def _insufficient_solvency(
    *,
    reasons: Sequence[str],
    assessed_fact_ids: Sequence[str],
    assessed_assets: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """An early insufficient return that still names its rejected evidence."""
    ordered = _dedupe(reasons)
    return {
        "status": SOLVENCY_INSUFFICIENT_EVIDENCE,
        "reasons": ordered,
        "inputs": {
            "cash_and_equivalents": None,
            "near_term_debt": None,
            "current_assets": None,
            "current_liabilities": None,
            "current_ratio": None,
            "free_cash_flow": None,
        },
        "periods": {
            "instant_date": None,
            "duration_start": None,
            "duration_end": None,
            "duration_basis": None,
        },
        "runway": {
            "status": RUNWAY_WITHHELD,
            "quarters": None,
            "reason": ordered[0] if ordered else REASON_EVIDENCE_NOT_CUTOFF_SAFE,
        },
        "assessed_fact_ids": list(assessed_fact_ids),
        "assessed_assets": list(assessed_assets),
    }


def _nonfinite_flow_operand_reason(
    admitted: Sequence[FundamentalFact],
    *,
    sec_config: SecFundamentalsConfig,
) -> str | None:
    """Whether a selected OCF/capex vintage is unusable before it is combined.

    `build_sec_fundamental_series` combines operating cash flow and capital
    expenditure with unguarded Decimal subtraction (directly, and inside its
    YTD/TTM derivation); an `Infinity`/`NaN` operand there raises
    `InvalidOperation` before the builder's own finiteness checks ever run.

    This calls the exact same vintage selector the builder uses internally,
    over the same admitted facts and configuration, so it inspects the
    identical operand the builder would combine -- never a stand-in derived
    by filtering `admitted` first, which would let the frozen selector
    silently promote an older, more favorable vintage in its place.
    """
    selected = select_latest_fact_vintages(admitted, config=sec_config)
    for fact in selected:
        if fact.concept in UNDER10_DURATION_CONCEPTS and not fact.value.is_finite():
            return REASON_NONFINITE_INPUT
    return None


def _resolve_instants(
    series: SecFundamentalSeries,
    *,
    target_date: date,
    reasons: list[str],
) -> tuple[dict[str, _ResolvedInput], date | None]:
    """Resolve the five balance-sheet operands from one shared instant date.

    Each selected observation is validated on its own (unit, finiteness,
    freshness) *before* the shared-date rule, so a present-but-unusable value
    can never be reported as a usable input just because a different concept
    was the one missing.

    The shared-period-end compatibility check runs over whichever instants
    are actually present -- before the missing-concept early return, not
    only when all five happen to be present. A missing concept must never
    let two mutually incompatible *surviving* facts (e.g. current assets and
    current liabilities from different balance-sheet dates) slip through as
    usable together merely because a third, unrelated concept is absent.
    Fewer than two present facts cannot disagree with each other, so an
    absence with otherwise-agreeing survivors is not a mismatch: cash (and
    therefore an independent negative-FCF runway diagnostic) stays usable
    exactly when its own evidence is genuinely compatible.
    """
    resolved: dict[str, _ResolvedInput] = {}
    selected: dict[str, FundamentalFact] = {}
    for concept in UNDER10_INSTANT_CONCEPTS:
        fact = series.latest_instants.get(concept)
        if fact is None:
            resolved[concept] = _ResolvedInput(reason=_missing_reason(concept))
            continue
        selected[concept] = fact
        resolved[concept] = _validate_instant_fact(fact, target_date=target_date, reasons=reasons)

    # Current liabilities must never be presented as a usable obligation
    # denominator when it is zero or negative -- independent of whether a
    # debt component or another instant is missing below. Checking this
    # before the missing-concept return means a validity defect in one
    # concept can never suppress a validity defect already found in another;
    # missing debt and non-positive liabilities both stay recorded.
    liabilities = resolved.get("current_liabilities")
    if liabilities is not None and liabilities.usable and _require(liabilities) <= 0:
        reasons.append(REASON_NONPOSITIVE_CURRENT_LIABILITIES)
        resolved["current_liabilities"] = _ResolvedInput(
            reason=REASON_NONPOSITIVE_CURRENT_LIABILITIES
        )

    # A convenient common date is never searched for backwards; the latest
    # selected vintages either agree or the evidence is incompatible. This
    # must run over the *present* subset before the missing-concept check
    # below returns early, or an absent concept could hide a genuine
    # disagreement among the concepts that did survive.
    period_ends = {fact.period_end for fact in selected.values()}
    if len(period_ends) > 1:
        reasons.append(REASON_INSTANT_PERIOD_MISMATCH)
        for concept in selected:
            resolved[concept] = _ResolvedInput(reason=REASON_INSTANT_PERIOD_MISMATCH)

    missing = [concept for concept in UNDER10_INSTANT_CONCEPTS if concept not in selected]
    debt_components = ("short_term_debt", "current_long_term_debt")
    if any(concept in missing for concept in debt_components):
        reasons.append(REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING)
    for concept in missing:
        if concept not in debt_components:
            reasons.append(_missing_reason(concept))
    if missing:
        return resolved, None

    # All five instants are present here; `period_ends` above already
    # reflects whether they agree.
    if len(period_ends) != 1:
        return resolved, None
    instant_date = period_ends.pop()
    return resolved, instant_date


def _validate_instant_fact(
    fact: FundamentalFact,
    *,
    target_date: date,
    reasons: list[str],
) -> _ResolvedInput:
    freshness = _freshness_reason(fact.period_end, target_date=target_date)
    if freshness is not None:
        reasons.append(freshness)
        return _ResolvedInput(reason=freshness)
    if fact.unit != UNDER10_REQUIRED_UNIT:
        reasons.append(REASON_INCOMPATIBLE_UNIT)
        return _ResolvedInput(reason=REASON_INCOMPATIBLE_UNIT)
    if not fact.value.is_finite():
        reasons.append(REASON_NONFINITE_INPUT)
        return _ResolvedInput(reason=REASON_NONFINITE_INPUT)
    return _ResolvedInput(value=fact.value)


def _near_term_debt(
    short_term: _ResolvedInput,
    current_long_term: _ResolvedInput,
    *,
    reasons: list[str],
) -> _ResolvedInput:
    """``short_term_debt + current_long_term_debt``; a missing part is missing."""
    if not short_term.usable or not current_long_term.usable:
        reason = short_term.reason or current_long_term.reason
        if reason in {
            _missing_reason("short_term_debt"),
            _missing_reason("current_long_term_debt"),
        }:
            reason = REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING
        return _ResolvedInput(reason=reason or REASON_NEAR_TERM_DEBT_COMPONENTS_MISSING)
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        total = _require(short_term) + _require(current_long_term)
    if not total.is_finite():
        reasons.append(REASON_NONFINITE_INPUT)
        return _ResolvedInput(reason=REASON_NONFINITE_INPUT)
    return _ResolvedInput(value=total)


def _resolve_free_cash_flow(
    series: SecFundamentalSeries,
    *,
    target_date: date,
    reasons: list[str],
) -> tuple[_ResolvedInput, dict[str, Any]]:
    """TTM free cash flow, else the latest compatible annual free cash flow.

    Unrelated "latest" operating-cash-flow and capital-expenditure
    observations are never combined here: the canonical builder already
    derived free cash flow only from period- and unit-compatible pairs, and a
    stale or adverse selected metric is never replaced with an older, more
    convenient one.
    """
    periods: dict[str, Any] = {
        "duration_start": None,
        "duration_end": None,
        "duration_basis": None,
    }
    value: FundamentalValue | None = series.ttm.get("free_cash_flow")
    basis = "ttm"
    if value is None:
        annual = series.annual.get("free_cash_flow") or ()
        value = annual[-1] if annual else None
        basis = "annual"
    if value is None:
        reasons.append(REASON_FREE_CASH_FLOW_MISSING)
        return _ResolvedInput(reason=REASON_FREE_CASH_FLOW_MISSING), periods
    periods = {
        "duration_start": value.period_start.isoformat(),
        "duration_end": value.period_end.isoformat(),
        "duration_basis": basis,
    }
    freshness = _freshness_reason(value.period_end, target_date=target_date)
    if freshness is not None:
        reasons.append(freshness)
        return _ResolvedInput(reason=freshness), periods
    if value.unit != UNDER10_REQUIRED_UNIT:
        reasons.append(REASON_INCOMPATIBLE_UNIT)
        return _ResolvedInput(reason=REASON_INCOMPATIBLE_UNIT), periods
    if not value.value.is_finite():
        reasons.append(REASON_NONFINITE_INPUT)
        return _ResolvedInput(reason=REASON_NONFINITE_INPUT), periods
    return _ResolvedInput(value=value.value), periods


def _freshness_reason(period_end: date, *, target_date: date) -> str | None:
    """``0 <= target_date - period_end <= UNDER10_MAX_METRIC_AGE_DAYS``."""
    age = (target_date - period_end).days
    if age < 0:
        return REASON_FUTURE_PERIOD_END
    if age > UNDER10_MAX_METRIC_AGE_DAYS:
        return REASON_STALE_METRIC
    return None


def _build_runway(*, cash: _ResolvedInput, fcf: _ResolvedInput) -> dict[str, Any]:
    """Cash runway, reported independently of the other obligation inputs."""
    if not fcf.usable:
        return {
            "status": RUNWAY_WITHHELD,
            "quarters": None,
            "reason": fcf.reason or REASON_FREE_CASH_FLOW_MISSING,
        }
    free_cash_flow = _require(fcf)
    if free_cash_flow >= 0:
        return {"status": RUNWAY_NOT_APPLICABLE, "quarters": None, "reason": None}
    if not cash.usable:
        return {
            "status": RUNWAY_WITHHELD,
            "quarters": None,
            "reason": cash.reason or _missing_reason("cash_and_equivalents"),
        }
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        quarters = (Decimal(4) * _require(cash)) / abs(free_cash_flow)
    if not quarters.is_finite():
        return {
            "status": RUNWAY_WITHHELD,
            "quarters": None,
            "reason": REASON_NONFINITE_INPUT,
        }
    return {
        "status": RUNWAY_COMPUTED,
        "quarters": _reported_text(quarters),
        "reason": None,
    }


def _current_ratio(
    current_assets: _ResolvedInput,
    current_liabilities: _ResolvedInput,
) -> str | None:
    """Reported diagnostic only. No state decision reads this value."""
    if not current_assets.usable or not current_liabilities.usable:
        return None
    liabilities = _require(current_liabilities)
    if liabilities <= 0:
        return None
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        ratio = _require(current_assets) / liabilities
    if not ratio.is_finite():
        return None
    return _reported_text(ratio)


def _classify_solvency(
    *,
    cash: Decimal,
    near_term_debt: Decimal,
    current_assets: Decimal,
    current_liabilities: Decimal,
    free_cash_flow: Decimal,
) -> tuple[str, list[str]]:
    """Strict first-match state partition over exact Decimal operands.

    The four-quarter runway boundary is applied as a cross-multiplication, so
    a displayed ``4.0000`` never overrides an exact below-four classification
    and equality at four quarters stays on the non-adverse side.
    """
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        debt_exceeds_cash = near_term_debt > cash
        assets_below_liabilities = current_assets < current_liabilities
        negative_fcf = free_cash_flow < 0
        short_runway = negative_fcf and (
            Decimal(4) * cash < UNDER10_MIN_RUNWAY_QUARTERS * abs(free_cash_flow)
        )
    reasons: list[str] = []
    if debt_exceeds_cash:
        reasons.append(REASON_NEAR_TERM_DEBT_EXCEEDS_CASH)
    if assets_below_liabilities:
        reasons.append(REASON_CURRENT_ASSETS_BELOW_LIABILITIES)
    if negative_fcf:
        reasons.append(REASON_NEGATIVE_FREE_CASH_FLOW)
    if short_runway:
        reasons.append(REASON_RUNWAY_BELOW_MINIMUM_QUARTERS)
    if debt_exceeds_cash and (assets_below_liabilities or short_runway):
        return SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION, reasons
    if negative_fcf or debt_exceeds_cash or assets_below_liabilities:
        return SOLVENCY_ELEVATED_OBLIGATION_RISK, reasons
    return SOLVENCY_NO_ADVERSE_EVIDENCE, []


def _periods_payload(instant_date: date | None, periods: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instant_date": instant_date.isoformat() if instant_date is not None else None,
        "duration_start": periods["duration_start"],
        "duration_end": periods["duration_end"],
        "duration_basis": periods["duration_basis"],
    }


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------


def _build_liquidity(
    *,
    price_frame: pl.DataFrame,
    price_asset: DataAsset | None,
    price_source: Mapping[str, Any] | None,
    target_date: date,
    invalid_session_date_rows: int,
) -> dict[str, Any]:
    """The 252-observed-session median dollar-volume diagnostic.

    Provenance and basis are proven first: a frame whose price anchor is
    unknown or mismatched is withheld with empty basis fields (nothing was
    observed for the *correct* asset), while a confirmed anchor with
    incompatible metadata is withheld with that metadata reported as
    observed -- never silently replaced with nulls, which would erase the
    difference between "absent" and "present but wrong".
    """
    empty_basis: dict[str, Any] = {
        "interval": None,
        "adjustment": None,
        "return_definition": None,
        "volume_basis": UNDER10_VOLUME_BASIS,
    }
    if price_asset is None:
        return _withheld_liquidity(
            reason=LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE,
            basis=empty_basis,
            price_asset=None,
        )
    anchor = price_source.get("asset_id") if isinstance(price_source, Mapping) else None
    if not isinstance(anchor, str) or anchor != str(price_asset.id):
        return _withheld_liquidity(
            reason=LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE,
            basis=empty_basis,
            price_asset=None,
        )
    reference = _asset_reference(price_asset)
    metadata = price_asset.metadata if isinstance(price_asset.metadata, Mapping) else {}
    required = (
        ("interval", UNDER10_REQUIRED_INTERVAL),
        ("adjustment", UNDER10_REQUIRED_ADJUSTMENT),
        ("return_definition", UNDER10_REQUIRED_RETURN_DEFINITION),
        ("currency", PRICE_BAND_CURRENCY),
    )
    if any(metadata.get(key) != expected for key, expected in required):
        # The asset anchor is confirmed, so this metadata was genuinely
        # observed on the correct asset; report it, incompatible or not,
        # rather than nulling it out as though nothing was observed.
        observed_basis: dict[str, Any] = {
            "interval": metadata.get("interval"),
            "adjustment": metadata.get("adjustment"),
            "return_definition": metadata.get("return_definition"),
            "volume_basis": UNDER10_VOLUME_BASIS,
        }
        return _withheld_liquidity(
            reason=LIQUIDITY_BASIS_INCOMPATIBLE,
            basis=observed_basis,
            price_asset=reference,
        )
    basis: dict[str, Any] = {
        "interval": UNDER10_REQUIRED_INTERVAL,
        "adjustment": UNDER10_REQUIRED_ADJUSTMENT,
        "return_definition": UNDER10_REQUIRED_RETURN_DEFINITION,
        "volume_basis": UNDER10_VOLUME_BASIS,
    }
    if invalid_session_date_rows:
        # `AsOfData.price_frame_with_diagnostics` clips future rows without
        # counting them, but a row whose session date could not be
        # established at all is a different problem: it never enters the
        # frame this helper receives, so `median_dollar_volume` would report
        # a shorter-but-plausible history instead of the missing identity.
        return _withheld_liquidity(
            reason=DOLLAR_VOLUME_INVALID_SESSION_DATES,
            basis=basis,
            price_asset=reference,
        )
    result = median_dollar_volume(price_frame, sessions=UNDER10_LIQUIDITY_SESSIONS)
    staleness = _price_staleness_reason(result, target_date=target_date)
    if staleness is not None:
        return _withheld_liquidity(
            reason=staleness,
            basis=basis,
            price_asset=reference,
            observed=result,
        )
    if result.status != DOLLAR_VOLUME_COMPUTED or result.value is None:
        return _withheld_liquidity(
            reason=result.reason or LIQUIDITY_BASIS_INCOMPATIBLE,
            basis=basis,
            price_asset=reference,
            observed=result,
        )
    return {
        "status": LIQUIDITY_COMPUTED,
        "metric": UNDER10_LIQUIDITY_METRIC,
        "value": result.value,
        "currency": PRICE_BAND_CURRENCY,
        "sessions_used": result.sessions_used,
        "first_session": _date_text(result.first_session),
        "last_session": _date_text(result.last_session),
        "basis": basis,
        "price_asset": reference,
        "reason": None,
    }


def _price_staleness_reason(
    result: DollarVolumeResult,
    *,
    target_date: date,
) -> str | None:
    last_session = result.last_session
    if last_session is None:
        return None
    if last_session > target_date:
        return LIQUIDITY_FUTURE_PRICE_SESSION
    if (target_date - last_session).days > UNDER10_MAX_PRICE_STALENESS_DAYS:
        return LIQUIDITY_STALE_PRICE_EVIDENCE
    return None


def _withheld_liquidity(
    *,
    reason: str,
    basis: Mapping[str, Any],
    price_asset: dict[str, str] | None,
    observed: DollarVolumeResult | None = None,
) -> dict[str, Any]:
    return {
        "status": LIQUIDITY_WITHHELD,
        "metric": UNDER10_LIQUIDITY_METRIC,
        "value": None,
        "currency": PRICE_BAND_CURRENCY,
        "sessions_used": observed.sessions_used if observed is not None else None,
        "first_session": _date_text(observed.first_session) if observed is not None else None,
        "last_session": _date_text(observed.last_session) if observed is not None else None,
        "basis": dict(basis),
        "price_asset": price_asset,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _asset_references(facts: Sequence[FundamentalFact]) -> list[dict[str, str]]:
    """Immutable asset identity and complete content checksum, deduplicated.

    Reads the already-selected ``source_asset`` relation only; no fact
    content, source path, or credential-bearing metadata is copied.
    """
    references: dict[str, dict[str, str]] = {}
    for fact in facts:
        reference = _asset_reference(fact.source_asset)
        references[reference["id"]] = reference
    return [references[key] for key in sorted(references)]


def _asset_reference(asset: DataAsset) -> dict[str, str]:
    id_key, checksum_key = UNDER10_ASSET_REFERENCE_KEYS
    return {id_key: str(asset.id), checksum_key: asset.sha256}


def _require(resolved: _ResolvedInput) -> Decimal:
    value = resolved.value
    if value is None:
        raise ValueError("Under-$10 shadow assessment read an unresolved operand")
    return value


def _monetary_text(resolved: _ResolvedInput) -> str | None:
    if not resolved.usable:
        return None
    return _quantized_text(_require(resolved), MONETARY_PLACES)


def _reported_text(value: Decimal) -> str:
    return _quantized_text(value, REPORTED_PLACES)


def _reference_close_text(value: Decimal) -> str:
    return _quantized_text(value, REFERENCE_CLOSE_PLACES)


def _quantized_text(value: Decimal, places: Decimal) -> str:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        try:
            quantized = value.quantize(places, rounding=ROUND_HALF_EVEN)
        except InvalidOperation as error:
            raise ValueError(
                f"Under-$10 shadow assessment cannot serialize {value!r} at {places!r}"
            ) from error
    # Plain fixed-point notation: ``str(Decimal("0E-8"))`` would emit an
    # exponent form for zero, which is the same number but a different
    # canonical byte string.
    return f"{quantized:f}"


def _date_text(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))
