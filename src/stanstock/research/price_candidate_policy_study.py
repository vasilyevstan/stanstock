"""Six prescribed synthetic histories and six DISCONNECTED payoff examples.

The driver demonstrates correctness, not forecasting skill, cohort support,
tradable execution or calibrated outcomes. No source adapter, pooled selector,
file writer or environment discovery is provided. The parent independently
binds execution identity and owns any private serialization/persistence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from uuid import UUID

from stanstock.research.price_candidate_policy import (
    _CONFIG,
    _CONFIG_SHA256,
    _DECISION,
    _TARGET,
    SyntheticCandidateAssessment,
    SyntheticCandidateInput,
    _calendar_sessions,
    _canonical_bytes,
    _fixed12,
    _series_sha256,
    _sha256,
    assess_synthetic_candidate,
)
from stanstock.research.price_product import (
    PriceInputIdentity,
    PriceProductInput,
    PriceSeries,
    SourceExecutionBinding,
)
from stanstock.research.price_product_config import PriceProductConfig

# The parent hashes the path-sorted manifest of actual file digests, including
# the lock bytes once, reusing their digest for the manifest entry and the
# separate dependency-lock identity (not the manifest's own digest). This
# allow-list declares scope; this module never reads these files.
SOURCE_MANIFEST_PATHS = (
    "config/scoring/research-product-v1.yml",
    "pyproject.toml",
    "src/stanstock/research/price_candidate_policy.py",
    "src/stanstock/research/price_candidate_policy_study.py",
    "src/stanstock/research/price_product.py",
    "src/stanstock/research/price_product_config.py",
    "uv.lock",
)

_CONTINUATION = ((0, 10000), (504, 10000), (630, 12000), (735, 13000), (756, 13500))
_PULLBACK = (
    (0, 10000),
    (504, 10000),
    (630, 12000),
    (714, 15000),
    (735, 14000),
    (746, 12000),
    (756, 13500),
)
_DEEP = (
    (0, 10000),
    (504, 15000),
    (630, 12000),
    (714, 12000),
    (735, 9500),
    (746, 7800),
    (756, 8800),
)
_DETERIORATION = (
    (0, 10000),
    (504, 14000),
    (630, 13000),
    (735, 11000),
    (755, 10000),
    (756, 9800),
)
_CASES = (
    ("continuation", _CONTINUATION),
    ("positive_pullback", _PULLBACK),
    ("deep_reversal", _DEEP),
    ("deterioration", _DETERIORATION),
    ("flat", ()),
    ("deep_missing_volume", _DEEP),
)


@dataclass(frozen=True, slots=True)
class SyntheticExitPayoff:
    horizon_sessions: int
    stock_terminal_return: str
    cash_return: str
    benchmark_terminal_return: str
    differential_cost: str
    cash_minus_hold: str
    benchmark_minus_hold: str
    loss_avoided: str
    foregone_upside: str


@dataclass(frozen=True, slots=True)
class SyntheticCandidateStudy:
    cases: tuple[SyntheticCandidateAssessment, ...]
    exit_payoff_examples: tuple[SyntheticExitPayoff, ...]


@dataclass(frozen=True, slots=True)
class SyntheticExecutionIdentity:
    code_revision: str | None
    source_fingerprint_sha256: str
    dependency_lock_sha256: str
    python_version: str
    numpy_version: str
    exchange_calendars_version: str
    platform_architecture: str


def _closes(anchors: tuple[tuple[int, int], ...]) -> tuple[float, ...]:
    if not anchors:
        return (100.0,) * 757
    values = []
    segment = 0
    for i in range(757):
        while segment + 1 < len(anchors) - 1 and i > anchors[segment + 1][0]:
            segment += 1
        start, first = anchors[segment]
        end, last = anchors[segment + 1]
        cents = first + (last - first) * (i - start) // (end - start) + i % 2
        with localcontext() as context:
            context.prec = 80
            context.rounding = ROUND_HALF_EVEN
            values.append(float(Decimal(cents) / Decimal(100)))
    return tuple(values)


def _synthetic_candidates() -> tuple[SyntheticCandidateInput, ...]:
    dates = _calendar_sessions()
    admitted = datetime(2026, 9, 11, 20, 30, tzinfo=UTC)
    benchmark = PriceSeries(
        PriceInputIdentity(UUID(int=200), "synthetic_demo", "SPY", "", admitted, admitted),
        "USD",
        dates,
        tuple(100.0 if i % 2 == 0 else 102.0 for i in range(757)),
    )
    benchmark = replace(
        benchmark, identity=replace(benchmark.identity, sha256=_series_sha256(benchmark))
    )
    cases = []
    for n, (case_id, anchors) in enumerate(_CASES, start=1):
        stock = PriceSeries(
            PriceInputIdentity(
                UUID(int=100 + n),
                "synthetic_demo",
                "SYN-" + case_id.upper(),
                "",
                admitted,
                admitted,
            ),
            "USD",
            dates,
            _closes(anchors),
            None if case_id == "deep_missing_volume" else (1_000_000.0,) * 757,
            case_id != "deep_missing_volume",
        )
        stock = replace(stock, identity=replace(stock.identity, sha256=_series_sha256(stock)))
        product_input = PriceProductInput(
            UUID(int=n),
            _TARGET,
            _DECISION,
            dates,
            stock,
            benchmark,
            SourceExecutionBinding("synthetic_demo", "research"),
        )
        cases.append(SyntheticCandidateInput(case_id, "common_stock", "us", product_input))
    return tuple(cases)


def _payoffs() -> tuple[SyntheticExitPayoff, ...]:
    # Static hypothetical terminal inputs; never derived from a case's pattern.
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        return tuple(
            SyntheticExitPayoff(
                horizon,
                _fixed12(r),
                _fixed12(Decimal(0)),
                _fixed12(Decimal("0.05")),
                _fixed12(Decimal(0)),
                _fixed12(-r),
                _fixed12(Decimal("0.05") - r),
                _fixed12(max(-r, Decimal(0))),
                _fixed12(max(r, Decimal(0))),
            )
            for horizon in (126, 252)
            for r in (Decimal("-0.20"), Decimal("0"), Decimal("0.20"))
        )


def run_synthetic_candidate_study(*, base_config: PriceProductConfig) -> SyntheticCandidateStudy:
    return SyntheticCandidateStudy(
        tuple(
            assess_synthetic_candidate(candidate, base_config=base_config)
            for candidate in _synthetic_candidates()
        ),
        _payoffs(),
    )


def serialize_synthetic_candidate_study(
    study: SyntheticCandidateStudy, *, execution_identity: SyntheticExecutionIdentity
) -> bytes:
    """Pure canonical bytes. Supplied identity is not self-authenticating proof."""
    if tuple(case.case_id for case in study.cases) != tuple(case_id for case_id, _ in _CASES):
        raise ValueError("Complete fixed synthetic case order required")
    if study.exit_payoff_examples != _payoffs():
        raise ValueError("Prescribed disconnected synthetic payoff examples required")
    identity = execution_identity
    if (
        identity.code_revision is not None
        and re.fullmatch(r"[0-9a-f]{40}", identity.code_revision) is None
    ):
        raise ValueError("Execution revision must be an exact committed SHA or null")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (identity.source_fingerprint_sha256, identity.dependency_lock_sha256)
    ):
        raise ValueError("Execution fingerprints must be SHA256 values")
    if any(
        re.fullmatch(r"[A-Za-z0-9_.+ -]+", value) is None
        for value in (
            identity.python_version,
            identity.numpy_version,
            identity.exchange_calendars_version,
            identity.platform_architecture,
        )
    ):
        raise ValueError("Execution environment labels must be nonempty and path-free")
    document = {
        "schema": "opportunities-candidates-synthetic-study@1",
        "contract_revision": _CONFIG.contract_revision,
        "policy_version": _CONFIG.policy_version,
        "config_sha256": _CONFIG_SHA256,
        "base_config_hash": _CONFIG.base_config_hash,
        "execution_identity": asdict(identity),
        "claim_status": "synthetic_correctness_only",
        "selection": {
            "status": "deferred",
            "future_limit": _CONFIG.future_maximum_per_list,
            "buy": None,
            "sell_review": None,
        },
        "real_evaluation": {
            "status": "blocked",
            "reasons": (
                "real_data_adapter_not_authorized",
                "untouched_confirmation_not_established",
                "confirmation_protocol_not_frozen",
            ),
        },
        **asdict(study),
    }
    document["report_sha256"] = _sha256(document)
    return _canonical_bytes(document)
