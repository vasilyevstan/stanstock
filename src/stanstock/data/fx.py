"""Point-in-time FX conversion built on immutable `FxRate` vintages.

Converting a EUR price into a USD portfolio balance is a point-in-time read
like any other, so it goes through the same gate as prices and filing facts:
`AsOfData` supplies only rates whose ``available_at`` (and whose source
asset's ``available_at``/``retrieved_at``) precede the run's decision
boundary. On top of that availability gate this module adds the rules a
currency conversion needs and a single price read does not.

**Every valued date carries its own cutoff.** A run-wide decision boundary
alone is not enough: it would let a correction published in February silently
change how an execution in January was priced. Each valued date ``D`` is
therefore resolved against its own cutoff -- the end of ``D`` -- so a value
dated ``D`` can only ever be converted with information that existed by the
end of ``D``. A rate observed after ``D``, and a vintage published after
``D``, are both excluded no matter how long before the decision boundary they
appeared.

**Late publication and later retrieval are different failures.** A rate
*published* after the valued date describes information nobody had then, and
is refused for every run. A source asset *retrieved* after the valued date is
a different matter: an explicitly research-grade reconstruction is allowed to
read a file StanStock only fetched later, exactly as it may for filing facts
and price frames, while an observed-grade run must have held the asset at the
time. `FxEvidenceGrade` makes the caller state which it is; there is no
default, because guessing would silently upgrade a reconstruction into a
claim about what was observed.

**Explicit carry, never silent staleness.** FX series have no observations on
weekends, holidays, or -- for the synthetic demo bundles -- any day that is
not a Friday. The most recent eligible observation is carried forward, and
the carry distance is recorded per converted date. A carry longer than
``max_carry_days`` fails explicitly instead of quietly pricing a portfolio
off a rate from an arbitrarily distant past.

**Deterministic, named derivation paths.** A provider publishes a subset of
the pairs a portfolio needs. ECB publishes EUR-based quotes only, so
``USD -> EUR`` must be inverted and ``GBP -> USD`` must be crossed through
EUR. Candidate derivations are ranked ``identity`` > ``direct`` > ``inverse``
> ``cross:<pivot>``; the best available rank wins, and every conversion
records which path produced it. Two derivations *of the same rank* that
disagree (for example two pivot currencies implying different cross rates)
are genuinely ambiguous and raise rather than silently picking one.

The whole audit trail -- the observation date used, the carry distance, the
derivation path, the vintage's publication and availability times, the cutoff
that admitted it, and the evidence grade under which it was admitted --
travels with each conversion so a persisted simulation can be replayed and
checked without re-deriving anything.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum

import polars as pl

from stanstock.data.asof import AsOfData

#: Longest gap, in calendar days, between an FX observation and the date it
#: is carried forward to. Seven days covers a weekend plus a surrounding
#: holiday run for a daily series, and covers the synthetic demo bundles'
#: Friday-only observations (whose worst case is a Thursday priced from the
#: previous Friday, six days back). A longer gap is a data outage, not a
#: normal market closure, so it fails instead of being carried. This is the
#: reviewed maximum: callers may tighten it, never widen it.
DEFAULT_MAX_CARRY_DAYS = 7

#: Relative disagreement above which two same-rank derivations of one rate
#: are treated as ambiguous rather than as rounding noise.
PATH_AGREEMENT_TOLERANCE = 1e-6

IDENTITY_PATH = "identity"
DIRECT_PATH = "direct"
INVERSE_PATH = "inverse"
CROSS_PATH_PREFIX = "cross:"

FX_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "value_date": pl.Date(),
    "from_currency": pl.Utf8(),
    "to_currency": pl.Utf8(),
    "rate": pl.Float64(),
    "observation_date": pl.Date(),
    "carry_days": pl.Int32(),
    "path": pl.Utf8(),
    "rate_published_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "rate_available_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    "availability_cutoff": pl.Datetime(time_unit="us", time_zone="UTC"),
    "evidence_grade": pl.Utf8(),
    "source_asset_ids": pl.Utf8(),
}


class FxEvidenceGrade(StrEnum):
    """Whether a run claims observed evidence or a labeled reconstruction.

    The values match `UniverseSnapshot.Grade` so a caller can hand the
    snapshot's own grade straight through rather than re-deciding it.
    """

    OBSERVED = "observed"
    RESEARCH = "research"


class FxConversionError(ValueError):
    """Base class for every explicit point-in-time FX failure."""


class MissingFxRateError(FxConversionError):
    """Raised when no eligible rate path exists for a required conversion."""


class StaleFxRateError(FxConversionError):
    """Raised when the newest usable observation is older than the carry limit."""


class AmbiguousFxRateError(FxConversionError):
    """Raised when equally-ranked derivations of one rate disagree materially."""


@dataclass(frozen=True, slots=True)
class FxConversion:
    """One dated conversion factor plus the provenance that produced it."""

    from_currency: str
    to_currency: str
    value_date: date
    observation_date: date
    rate: float
    path: str
    carry_days: int
    availability_cutoff: datetime
    evidence_grade: FxEvidenceGrade
    published_at: datetime | None = None
    available_at: datetime | None = None
    source_asset_ids: tuple[str, ...] = ()

    def convert(self, amount: float) -> float:
        return amount * self.rate


@dataclass(frozen=True, slots=True)
class _Vintage:
    value: float
    published_at: datetime
    available_at: datetime
    source_available_at: datetime
    source_retrieved_at: datetime
    asset_id: str


@dataclass(frozen=True, slots=True)
class _Leg:
    rate: float
    published_at: datetime
    available_at: datetime
    asset_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    rank: int
    path: str
    rate: float
    published_at: datetime
    available_at: datetime
    asset_ids: tuple[str, ...]


def availability_cutoff_for(value_date: date) -> datetime:
    """The latest instant whose information a value dated ``value_date`` may use.

    Executions are not modeled intraday, so the end of the valued date is the
    most permissive defensible cutoff: anything published later belongs to a
    day that had not happened yet when the value was struck.
    """
    return datetime.combine(value_date, time.max, tzinfo=UTC)


class FxConverter:
    """Derive dated conversion factors from eligible `FxRate` vintages.

    The converter reads every rate the decision boundary permits once, then
    answers conversions from that in-memory snapshot, so one simulation can
    resolve thousands of (currency, date) pairs without re-querying and
    without any chance of two reads seeing different eligibility. Each
    individual conversion is then re-gated against its own valued date, so a
    later vintage inside that snapshot cannot reach back into an earlier date.
    """

    def __init__(
        self,
        asof: AsOfData,
        *,
        evidence_grade: FxEvidenceGrade,
        max_carry_days: int = DEFAULT_MAX_CARRY_DAYS,
        observation_end: date | None = None,
    ) -> None:
        if max_carry_days < 0:
            raise ValueError(f"max_carry_days must be non-negative, got {max_carry_days}")
        self.asof = asof
        self.evidence_grade = FxEvidenceGrade(evidence_grade)
        self.max_carry_days = max_carry_days
        # An observation dated after the last date this converter will ever be
        # asked about is provably unusable, so it is excluded at the query
        # rather than loaded and then skipped. No lower bound is applied: an
        # observation too old to carry must still be *seen*, so the failure
        # can say "stale by N days" instead of "missing".
        self._observation_end = observation_end
        self._loaded = False
        self._pairs: dict[tuple[str, str], dict[date, list[_Vintage]]] = {}
        self._currencies: set[str] = set()
        self._observation_dates: list[date] = []

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        observation_dates: set[date] = set()
        for rate in self.asof.fx_rates(observation_end=self._observation_end):
            base = rate.base_currency.upper()
            quote = rate.quote_currency.upper()
            if base == quote:
                continue
            value = float(rate.value)
            if value <= 0:
                continue
            asset = rate.source_asset
            slot = self._pairs.setdefault((base, quote), {})
            slot.setdefault(rate.observation_date, []).append(
                _Vintage(
                    value=value,
                    published_at=rate.published_at,
                    available_at=rate.available_at,
                    source_available_at=asset.available_at,
                    source_retrieved_at=asset.retrieved_at,
                    asset_id=str(rate.source_asset_id),
                )
            )
            self._currencies.update((base, quote))
            observation_dates.add(rate.observation_date)
        for pair_slot in self._pairs.values():
            for vintages in pair_slot.values():
                # Newest-knowable last, so a per-date scan can walk backwards
                # from the most recent vintage that date's cutoff still admits.
                vintages.sort(key=lambda v: (v.available_at, v.published_at, v.asset_id))
        self._observation_dates = sorted(observation_dates)
        self._loaded = True

    @property
    def known_currencies(self) -> frozenset[str]:
        self._load()
        return frozenset(self._currencies)

    # -- eligibility ------------------------------------------------------

    def _is_eligible(self, vintage: _Vintage, cutoff: datetime) -> bool:
        if vintage.published_at > cutoff:
            # Late publication is never permissible: the value did not exist
            # as public information on the date being priced.
            return False
        if self.evidence_grade is FxEvidenceGrade.RESEARCH:
            return True
        # Observed evidence must additionally have been held at the time; a
        # file fetched later is a reconstruction, however early it was
        # published.
        return (
            vintage.available_at <= cutoff
            and vintage.source_available_at <= cutoff
            and vintage.source_retrieved_at <= cutoff
        )

    def _vintage(
        self, from_currency: str, to_currency: str, observation: date, cutoff: datetime
    ) -> _Vintage | None:
        vintages = self._pairs.get((from_currency, to_currency), {}).get(observation)
        if not vintages:
            return None
        for vintage in reversed(vintages):
            if self._is_eligible(vintage, cutoff):
                return vintage
        return None

    # -- derivation -------------------------------------------------------

    def _simple_leg(
        self, from_currency: str, to_currency: str, observation: date, cutoff: datetime
    ) -> _Leg | None:
        direct = self._vintage(from_currency, to_currency, observation, cutoff)
        if direct is not None:
            return _Leg(direct.value, direct.published_at, direct.available_at, (direct.asset_id,))
        inverse = self._vintage(to_currency, from_currency, observation, cutoff)
        if inverse is not None:
            return _Leg(
                1.0 / inverse.value,
                inverse.published_at,
                inverse.available_at,
                (inverse.asset_id,),
            )
        return None

    def _candidates(
        self, from_currency: str, to_currency: str, observation: date, cutoff: datetime
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        direct = self._vintage(from_currency, to_currency, observation, cutoff)
        if direct is not None:
            candidates.append(
                _Candidate(
                    0,
                    DIRECT_PATH,
                    direct.value,
                    direct.published_at,
                    direct.available_at,
                    (direct.asset_id,),
                )
            )
        inverse = self._vintage(to_currency, from_currency, observation, cutoff)
        if inverse is not None:
            candidates.append(
                _Candidate(
                    1,
                    INVERSE_PATH,
                    1.0 / inverse.value,
                    inverse.published_at,
                    inverse.available_at,
                    (inverse.asset_id,),
                )
            )
        for pivot in sorted(self._currencies):
            if pivot in (from_currency, to_currency):
                continue
            first = self._simple_leg(from_currency, pivot, observation, cutoff)
            if first is None:
                continue
            second = self._simple_leg(pivot, to_currency, observation, cutoff)
            if second is None:
                continue
            candidates.append(
                _Candidate(
                    2,
                    f"{CROSS_PATH_PREFIX}{pivot}",
                    first.rate * second.rate,
                    max(first.published_at, second.published_at),
                    max(first.available_at, second.available_at),
                    tuple(sorted(set(first.asset_ids + second.asset_ids))),
                )
            )
        return candidates

    def _derive(
        self, from_currency: str, to_currency: str, observation: date, cutoff: datetime
    ) -> _Candidate | None:
        candidates = self._candidates(from_currency, to_currency, observation, cutoff)
        if not candidates:
            return None
        best_rank = min(candidate.rank for candidate in candidates)
        same_rank = [candidate for candidate in candidates if candidate.rank == best_rank]
        chosen = min(same_rank, key=lambda candidate: candidate.path)
        for other in same_rank:
            if other.path == chosen.path:
                continue
            if _relative_difference(chosen.rate, other.rate) > PATH_AGREEMENT_TOLERANCE:
                raise AmbiguousFxRateError(
                    f"Ambiguous {from_currency}/{to_currency} rate observed on "
                    f"{observation.isoformat()}: path {chosen.path!r} implies "
                    f"{chosen.rate!r} while path {other.path!r} implies {other.rate!r}. "
                    "Equally-ranked derivations must agree; refusing to pick one silently."
                )
        return chosen

    # -- public conversion API -------------------------------------------

    def conversion(
        self,
        *,
        from_currency: str,
        to_currency: str,
        value_date: date,
    ) -> FxConversion:
        """Return the conversion factor applicable to ``value_date``.

        Fails explicitly -- never returns an approximation -- when no path
        exists under that date's own cutoff, when the newest usable
        observation is staler than the carry limit, or when equally-ranked
        derivations disagree.
        """
        source = from_currency.upper()
        target = to_currency.upper()
        cutoff = availability_cutoff_for(value_date)
        if source == target:
            return FxConversion(
                from_currency=source,
                to_currency=target,
                value_date=value_date,
                observation_date=value_date,
                rate=1.0,
                path=IDENTITY_PATH,
                carry_days=0,
                availability_cutoff=cutoff,
                evidence_grade=self.evidence_grade,
            )

        self._load()
        dated = self._observation_dates[: bisect_right(self._observation_dates, value_date)]

        for observation in reversed(dated):
            carry_days = (value_date - observation).days
            if carry_days > self.max_carry_days:
                break
            chosen = self._derive(source, target, observation, cutoff)
            if chosen is None:
                continue
            return FxConversion(
                from_currency=source,
                to_currency=target,
                value_date=value_date,
                observation_date=observation,
                rate=chosen.rate,
                path=chosen.path,
                carry_days=carry_days,
                availability_cutoff=cutoff,
                evidence_grade=self.evidence_grade,
                published_at=chosen.published_at,
                available_at=chosen.available_at,
                source_asset_ids=chosen.asset_ids,
            )

        # Nothing usable inside the carry window. Look further back only to
        # tell the caller whether the data is stale or simply absent.
        for observation in reversed(dated):
            carry_days = (value_date - observation).days
            if carry_days <= self.max_carry_days:
                continue
            if self._candidates(source, target, observation, cutoff):
                raise StaleFxRateError(
                    f"The newest eligible {source}/{target} observation on or before "
                    f"{value_date.isoformat()} is dated {observation.isoformat()} "
                    f"({carry_days} calendar days earlier), which exceeds the "
                    f"{self.max_carry_days}-day carry limit. Refusing to value "
                    f"{value_date.isoformat()} from a stale rate."
                )

        raise MissingFxRateError(
            f"No {source}/{target} rate is derivable from any observation on or before "
            f"{value_date.isoformat()} that was published by the end of that date "
            f"(evidence grade {self.evidence_grade.value}) and available by the decision "
            f"boundary {self.asof.decision_time.isoformat()}."
        )

    def conversion_series(
        self,
        *,
        from_currency: str,
        to_currency: str,
        value_dates: Sequence[date],
    ) -> list[FxConversion]:
        return [
            self.conversion(
                from_currency=from_currency,
                to_currency=to_currency,
                value_date=value_date,
            )
            for value_date in sorted(set(value_dates))
        ]

    def conversion_frame(
        self,
        *,
        from_currencies: Iterable[str],
        to_currency: str,
        value_dates: Sequence[date],
    ) -> pl.DataFrame:
        """Build the replayable FX input frame for one simulation.

        Every (currency, date) pair the run can possibly need is resolved up
        front so a missing or stale rate fails before any accounting starts,
        and so the persisted frame is a complete, self-contained record of
        the FX inputs rather than a sample of the ones that happened to be
        touched.
        """
        conversions: list[FxConversion] = []
        for currency in sorted({value.upper() for value in from_currencies}):
            conversions.extend(
                self.conversion_series(
                    from_currency=currency,
                    to_currency=to_currency,
                    value_dates=value_dates,
                )
            )
        return build_fx_frame(conversions)


def build_fx_frame(conversions: Iterable[FxConversion]) -> pl.DataFrame:
    rows = [
        {
            "value_date": conversion.value_date,
            "from_currency": conversion.from_currency,
            "to_currency": conversion.to_currency,
            "rate": conversion.rate,
            "observation_date": conversion.observation_date,
            "carry_days": conversion.carry_days,
            "path": conversion.path,
            "rate_published_at": conversion.published_at,
            "rate_available_at": conversion.available_at,
            "availability_cutoff": conversion.availability_cutoff,
            "evidence_grade": conversion.evidence_grade.value,
            "source_asset_ids": ",".join(conversion.source_asset_ids),
        }
        for conversion in conversions
    ]
    if not rows:
        return pl.DataFrame(schema=FX_FRAME_SCHEMA)
    return pl.DataFrame(rows, schema=FX_FRAME_SCHEMA).sort(
        ["value_date", "from_currency", "to_currency"]
    )


def _relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    return abs(left - right) / scale
