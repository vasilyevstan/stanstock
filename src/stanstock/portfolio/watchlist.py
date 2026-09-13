from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal
from uuid import UUID

from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Case, IntegerField, OuterRef, Q, Subquery, Value, When
from django.utils import timezone

from stanstock.core.models import JobRun
from stanstock.core.refresh_verification import (
    ReplayedScheduledRefresh,
    replay_recorded_scheduled_refresh,
)
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, read_checksummed_bytes
from stanstock.data.live_us import resolve_us_target_date
from stanstock.data.models import (
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    Security,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.provider_policy import provider_plan_allows
from stanstock.data.providers.contracts import StockReference
from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.providers.twelve_data import (
    PROVIDER as TWELVE_DATA_PROVIDER,
)
from stanstock.data.providers.twelve_data import parse_stock_catalog_references
from stanstock.research.models import AnalysisRun, StockAnalysis
from stanstock.research.provenance import DATA_MODE_PROVIDER, source_data_mode

from .models import TrackedSymbol

SUPPORTED_CATALOG_EXCHANGES = ("NASDAQ", "NYSE")
SUPPORTED_CATALOG_MICS = {
    # XNAS is Nasdaq's operating MIC; the other three are its reviewed
    # US common-stock market segments. NYSE common stocks use XNYS.
    "NASDAQ": frozenset({"XNAS", "XNGS", "XNCM", "XNMS"}),
    "NYSE": frozenset({"XNYS"}),
}
SUPPORTED_LISTING_TYPES = (
    Security.SecurityType.COMMON_STOCK,
    Security.SecurityType.ADR,
)
SUPPORTED_CATALOG_INSTRUMENT_TYPES: Final[dict[str, str]] = {
    "Common Stock": Security.SecurityType.COMMON_STOCK,
    "ADR": Security.SecurityType.ADR,
    "American Depositary Receipt": Security.SecurityType.ADR,
    "Depositary Receipt": Security.SecurityType.ADR,
}
# Seven calendar days covers a long holiday weekend plus bounded scheduler
# recovery while still refusing an old catalog for interactive support
# admission. This is a support-currentness gate only, never an investment
# signal or a market-price freshness claim.
CATALOG_MAX_AGE_DAYS = 7
VERIFIED_PARENT_SCAN_LIMIT = 8
SCHEDULED_REFRESH_JOB = "scheduled_refresh"
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,31}$")
TRACKED_SYMBOL_FILTER_ALL = "all"
TRACKED_SYMBOL_FILTER_UNDER_10 = "under_10_provider"
TRACKED_SYMBOL_FILTER_NO_LIVE_PRICE = "no_live_price"
TRACKED_SYMBOL_FILTERS = frozenset(
    {
        TRACKED_SYMBOL_FILTER_ALL,
        TRACKED_SYMBOL_FILTER_UNDER_10,
        TRACKED_SYMBOL_FILTER_NO_LIVE_PRICE,
    }
)
TRACKED_SYMBOL_UNDER_10_THRESHOLD = Decimal("10")


class TrackedSymbolValidationError(ValueError):
    """A path-free validation error suitable for an owner-facing form."""


ListingResolutionStatus = Literal[
    "unique_listing",
    "no_local_listing",
    "ambiguous_listing",
]
UNIQUE_LISTING: Final[ListingResolutionStatus] = "unique_listing"
NO_LOCAL_LISTING: Final[ListingResolutionStatus] = "no_local_listing"
AMBIGUOUS_LISTING: Final[ListingResolutionStatus] = "ambiguous_listing"


@dataclass(frozen=True, slots=True)
class TrackedSymbolState:
    preference: TrackedSymbol
    resolution_status: ListingResolutionStatus
    listing: Listing | None
    market_data: LatestMarketData | None
    analysis: StockAnalysis | None
    in_selected_analysis_universe: bool
    has_live_provider_price: bool
    live_price_under_10: bool
    live_price_status: str


@dataclass(frozen=True, slots=True)
class _ListingResolution:
    status: ListingResolutionStatus
    listing: Listing | None


def normalize_tracked_symbol(raw_symbol: str) -> str:
    normalized = raw_symbol.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(normalized):
        raise TrackedSymbolValidationError(
            "Enter a symbol using up to 32 letters, numbers, periods, or hyphens."
        )
    return normalized


def normalize_tracked_symbol_filter(raw_filter: object) -> str:
    candidate = str(raw_filter or "").strip()
    return candidate if candidate in TRACKED_SYMBOL_FILTERS else TRACKED_SYMBOL_FILTER_ALL


def tracked_symbol_filter_choices(*, current: str) -> tuple[dict[str, object], ...]:
    active = normalize_tracked_symbol_filter(current)
    return (
        {
            "value": TRACKED_SYMBOL_FILTER_ALL,
            "label": "All tracked",
            "active": active == TRACKED_SYMBOL_FILTER_ALL,
        },
        {
            "value": TRACKED_SYMBOL_FILTER_UNDER_10,
            "label": "Under $10 provider-backed",
            "active": active == TRACKED_SYMBOL_FILTER_UNDER_10,
        },
        {
            "value": TRACKED_SYMBOL_FILTER_NO_LIVE_PRICE,
            "label": "No live persisted price",
            "active": active == TRACKED_SYMBOL_FILTER_NO_LIVE_PRICE,
        },
    )


def add_tracked_symbol(
    *,
    owner: User,
    raw_symbol: str,
    store: AssetStore | None = None,
) -> tuple[TrackedSymbol, bool]:
    symbol = normalize_tracked_symbol(raw_symbol)
    existing = TrackedSymbol.objects.filter(owner=owner, symbol=symbol).first()
    if existing is not None:
        return existing, False

    candidates = list(
        Listing.objects.filter(is_active=True)
        .filter(Q(provider_symbol=symbol) | Q(ticker=symbol))
        .select_related("security")
        .order_by("pk")[:2]
    )
    resolution = _resolve_listing_candidates(candidates)
    if resolution.status == AMBIGUOUS_LISTING:
        raise TrackedSymbolValidationError(f"{symbol} matches more than one active listing.")
    if resolution.status == NO_LOCAL_LISTING and candidates:
        raise TrackedSymbolValidationError(
            "Only active United States USD common-stock or ADR listings can be tracked."
        )
    if resolution.status == NO_LOCAL_LISTING:
        _validate_catalog_identity(symbol, store=store)
    return TrackedSymbol.objects.get_or_create(owner=owner, symbol=symbol)


def selected_watchlist_analysis_run() -> AnalysisRun | None:
    """Select provider analysis by target first, with an honest demo fallback."""
    first_analysis_quality = (
        StockAnalysis.objects.filter(run_id=OuterRef("pk"))
        .order_by("pk")
        .values("data_quality")[:1]
    )
    grade_preference = Case(
        When(
            universe_snapshot__grade=UniverseSnapshot.Grade.OBSERVED,
            then=Value(0),
        ),
        When(
            universe_snapshot__grade=UniverseSnapshot.Grade.RESEARCH,
            then=Value(1),
        ),
        default=Value(2),
        output_field=IntegerField(),
    )
    runs = (
        AnalysisRun.objects.filter(status="complete")
        .select_related("universe_snapshot__universe")
        .annotate(
            _watchlist_grade_preference=grade_preference,
            _watchlist_first_analysis_quality=Subquery(first_analysis_quality),
        )
        .order_by(
            "-target_date",
            "_watchlist_grade_preference",
            "-generated_at",
            "pk",
        )
    )
    demo_fallback: AnalysisRun | None = None
    for run in runs.iterator():
        if demo_fallback is None:
            demo_fallback = run
        if (
            source_data_mode(getattr(run, "_watchlist_first_analysis_quality", None))
            == DATA_MODE_PROVIDER
        ):
            return run
    return demo_fallback


def tracked_symbol_states(
    *,
    owner: User,
    selected_run: AnalysisRun | None,
    price_filter: str = TRACKED_SYMBOL_FILTER_ALL,
    live_session_date: date | None = None,
    decision_time: datetime | None = None,
) -> tuple[TrackedSymbolState, ...]:
    active_filter = normalize_tracked_symbol_filter(price_filter)
    effective_decision_time = decision_time or timezone.now()
    effective_live_session_date = live_session_date
    if effective_live_session_date is None:
        effective_live_session_date, _grade = resolve_us_target_date(
            decision_time=effective_decision_time
        )
    preferences = tuple(TrackedSymbol.objects.filter(owner=owner))
    symbols = {preference.symbol for preference in preferences}
    if not symbols:
        return ()

    listings_by_symbol: dict[str, list[Listing]] = {symbol: [] for symbol in symbols}
    candidates = (
        Listing.objects.filter(is_active=True)
        .filter(Q(provider_symbol__in=symbols) | Q(ticker__in=symbols))
        .select_related(
            "security__company",
            "latest_market_data",
            "latest_market_data__source_asset",
        )
    )
    for candidate in candidates:
        aliases = {candidate.ticker, candidate.provider_symbol}
        for symbol in aliases & symbols:
            listings_by_symbol[symbol].append(candidate)

    resolutions = {
        symbol: _resolve_listing_candidates(matches)
        for symbol, matches in listings_by_symbol.items()
    }
    unique_listings = {
        symbol: resolution.listing
        for symbol, resolution in resolutions.items()
        if resolution.listing is not None
    }
    listing_ids = {listing.pk for listing in unique_listings.values()}
    analyses_by_listing: dict[UUID, StockAnalysis] = {}
    selected_listing_ids: set[UUID] = set()
    if selected_run is not None and listing_ids:
        analyses_by_listing = {
            analysis.listing_id: analysis
            for analysis in StockAnalysis.objects.filter(
                run=selected_run,
                listing_id__in=listing_ids,
            )
        }
        selected_listing_ids = set(
            UniverseMembership.objects.filter(
                snapshot=selected_run.universe_snapshot,
                eligible=True,
                listing_id__in=listing_ids,
            ).values_list("listing_id", flat=True)
        )

    states: list[TrackedSymbolState] = []
    for preference in preferences:
        resolution = resolutions[preference.symbol]
        resolved_listing = resolution.listing
        market_data = (
            getattr(resolved_listing, "latest_market_data", None)
            if resolved_listing is not None
            else None
        )
        live_status = _live_provider_price_status(
            preference=preference,
            listing=resolved_listing,
            market_data=market_data,
            live_session_date=effective_live_session_date,
            decision_time=effective_decision_time,
        )
        has_live_provider_price = live_status == "eligible"
        live_price_under_10 = bool(
            has_live_provider_price
            and market_data is not None
            and market_data.close < TRACKED_SYMBOL_UNDER_10_THRESHOLD
        )
        states.append(
            TrackedSymbolState(
                preference=preference,
                resolution_status=resolution.status,
                listing=resolved_listing,
                market_data=market_data,
                analysis=(
                    analyses_by_listing.get(resolved_listing.pk)
                    if resolved_listing is not None
                    else None
                ),
                in_selected_analysis_universe=(
                    resolved_listing is not None and resolved_listing.pk in selected_listing_ids
                ),
                has_live_provider_price=has_live_provider_price,
                live_price_under_10=live_price_under_10,
                live_price_status=live_status,
            )
        )
    return _filter_tracked_symbol_states(tuple(states), active_filter)


def verified_catalog_references_for_symbols(
    *,
    symbols: Collection[str],
    store: AssetStore | None = None,
) -> dict[str, StockReference]:
    normalized_symbols = tuple(
        dict.fromkeys(normalize_tracked_symbol(symbol) for symbol in symbols)
    )
    if not normalized_symbols:
        return {}
    try:
        references = _verified_catalog_references(symbols=normalized_symbols, store=store)
    except TrackedSymbolValidationError:
        raise
    except (
        OSError,
        ProviderError,
        RefreshVerificationError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        raise TrackedSymbolValidationError(
            "Verified symbol catalog evidence is unavailable or invalid. "
            "Run a local market refresh before trying again."
        ) from None

    resolved: dict[str, StockReference] = {}
    for symbol in normalized_symbols:
        matches = [reference for reference in references if reference.symbol == symbol]
        if not matches:
            raise TrackedSymbolValidationError(
                f"{symbol} was not found in the latest verified NASDAQ/NYSE catalog bundle."
            )
        if len(matches) != 1:
            raise TrackedSymbolValidationError(
                f"{symbol} is ambiguous in the latest verified NASDAQ/NYSE catalog bundle."
            )
        reference = matches[0]
        _validate_supported_catalog_reference(reference)
        resolved[symbol] = reference
    return resolved


def _resolve_listing_candidates(candidates: list[Listing]) -> _ListingResolution:
    if len(candidates) > 1:
        return _ListingResolution(AMBIGUOUS_LISTING, None)
    if not candidates:
        return _ListingResolution(NO_LOCAL_LISTING, None)
    listing = candidates[0]
    if (
        listing.region != Region.US
        or listing.currency != "USD"
        or listing.security.security_type not in SUPPORTED_LISTING_TYPES
    ):
        return _ListingResolution(NO_LOCAL_LISTING, None)
    return _ListingResolution(UNIQUE_LISTING, listing)


def _validate_catalog_identity(symbol: str, *, store: AssetStore | None) -> None:
    verified_catalog_references_for_symbols(symbols=(symbol,), store=store)


def _validate_supported_catalog_reference(reference: StockReference) -> None:
    if (
        reference.country != "United States"
        or reference.currency != "USD"
        or reference.instrument_type not in SUPPORTED_CATALOG_INSTRUMENT_TYPES
        or reference.exchange not in SUPPORTED_CATALOG_EXCHANGES
        or reference.mic_code not in SUPPORTED_CATALOG_MICS[reference.exchange]
    ):
        raise TrackedSymbolValidationError(
            "Only United States USD common stocks or ADRs on reviewed NASDAQ or NYSE venues "
            "can be tracked."
        )
    installed_plan = _installed_provider_plan()
    if not provider_plan_allows(installed_plan, reference.access_plan):
        raise TrackedSymbolValidationError(
            "The installed Twelve Data plan does not provide access to this catalog symbol."
        )


def _verified_catalog_references(
    *,
    symbols: Collection[str],
    store: AssetStore | None,
) -> list[StockReference]:
    asset_store = store or _open_existing_asset_store()
    bundle = _latest_verified_refresh_bundle()
    age_days = (timezone.localdate() - bundle.parent.target_date).days
    if age_days < 0:
        raise RefreshVerificationError(
            "catalog_bundle_future",
            "The newest verified catalog bundle targets a future calendar date",
        )
    if age_days > CATALOG_MAX_AGE_DAYS:
        raise TrackedSymbolValidationError(
            "Verified symbol catalog evidence is stale (older than seven calendar days). "
            "Run a local market refresh before trying again."
        )

    references: list[StockReference] = []
    required_symbols = frozenset(symbols)
    for asset in bundle.catalog_assets:
        payload = read_checksummed_bytes(asset_store, asset)
        parsed, _count = parse_stock_catalog_references(
            payload,
            exchange=asset.subject,
            required_symbols=required_symbols,
            require_complete=True,
        )
        for reference in parsed:
            if reference.exchange != asset.subject:
                raise RefreshVerificationError(
                    "catalog_row_exchange_mismatch",
                    "A catalog row exchange does not match its catalog asset subject",
                )
        references.extend(parsed)
    return references


def _filter_tracked_symbol_states(
    states: tuple[TrackedSymbolState, ...],
    active_filter: str,
) -> tuple[TrackedSymbolState, ...]:
    if active_filter == TRACKED_SYMBOL_FILTER_UNDER_10:
        return tuple(state for state in states if state.live_price_under_10)
    if active_filter == TRACKED_SYMBOL_FILTER_NO_LIVE_PRICE:
        return tuple(state for state in states if not state.has_live_provider_price)
    return states


def _live_provider_price_status(
    *,
    preference: TrackedSymbol,
    listing: Listing | None,
    market_data: LatestMarketData | None,
    live_session_date: date,
    decision_time: datetime,
) -> str:
    if listing is None or market_data is None:
        return "missing"
    if listing.currency != "USD":
        return "non_usd"
    if market_data.session_date < live_session_date:
        return "stale"
    if market_data.session_date > live_session_date:
        return "future"
    if market_data.close <= 0:
        return "invalid"
    source_asset = getattr(market_data, "source_asset", None)
    if source_asset is None:
        return "missing"
    if (
        source_asset.provider != TWELVE_DATA_PROVIDER
        or source_asset.kind != "price_history"
        or source_asset.subject
        not in {
            preference.symbol,
            listing.provider_symbol,
        }
    ):
        return "not_live_provider"
    if source_asset.available_at > decision_time or source_asset.retrieved_at > decision_time:
        return "future"
    if source_asset.period_end != live_session_date:
        if source_asset.period_end and source_asset.period_end < live_session_date:
            return "stale"
        return "future"
    metadata = source_asset.metadata if isinstance(source_asset.metadata, dict) else {}
    if metadata.get("currency") != "USD":
        return "non_usd"
    if metadata.get("interval") != "1day" or metadata.get("adjustment") != "splits":
        return "bad_source"
    return "eligible"


def _latest_verified_refresh_bundle() -> ReplayedScheduledRefresh:
    """Select the newest independently replayed, recorded-verification parent.

    Successful parents without a ``verified`` claim are ignored, allowing a
    newer failed/unverified retry record to leave the last genuine bundle
    usable. Once a parent claims ``verified``, any replay or equality failure
    is treated as corruption and fails closed instead of silently falling
    back to an older parent.
    """
    parents = JobRun.objects.filter(
        job_name=SCHEDULED_REFRESH_JOB,
        region="us",
        status=JobRun.Status.SUCCESS,
    ).order_by("-target_date", "-finished_at", "-started_at", "-pk")[:VERIFIED_PARENT_SCAN_LIMIT]
    for parent in parents:
        details = parent.details
        verification = details.get("verification") if isinstance(details, dict) else None
        if isinstance(verification, dict) and verification.get("status") == "verified":
            return replay_recorded_scheduled_refresh(parent)
    raise TrackedSymbolValidationError(
        "Verified symbol catalog evidence is unavailable. "
        "Run a local market refresh before trying again."
    )


def _open_existing_asset_store() -> AssetStore:
    root = Path(settings.DATA_DIR)
    try:
        if not root.is_dir():
            raise RefreshVerificationError(
                "catalog_asset_root_unavailable",
                "The configured asset root does not exist",
            )
        return AssetStore(root)
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "catalog_asset_root_unavailable",
            "The configured asset root could not be opened",
        ) from None


def _installed_provider_plan() -> str:
    recorded = (
        ProviderRecord.objects.filter(provider=TWELVE_DATA_PROVIDER)
        .values_list("metadata__plan", flat=True)
        .first()
    )
    return recorded if isinstance(recorded, str) else ""
