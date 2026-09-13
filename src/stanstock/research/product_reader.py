"""Fail-closed, owner-bound read model for the active research product.

The writer's calculation artifact is the complete result.  This reader first
selects the expected owner cohort, invokes the public product verifier exactly
once for that run, and only then projects already-recorded values for web
rendering.  It never re-runs FHS, selects a newer price asset, or falls back to
legacy score-based analyses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol
from uuid import UUID

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.utils import timezone

from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.assets import (
    AssetStore,
    open_asset_store,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.live_us import resolve_us_target_date
from stanstock.data.models import DataAsset, Listing, ProviderRecord
from stanstock.data.provider_policy import (
    TWELVE_DATA_PROVIDER,
    validate_provider_usage,
)
from stanstock.data.providers.exceptions import ProviderConfigurationError
from stanstock.data.research_product import PRODUCT_INTAKE_KIND, product_membership_payload
from stanstock.data.research_product_demo import DEMO_OWNER_ID
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    MOMENTUM_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
)
from stanstock.research.product_pipeline import verify_price_product_output

ReaderStatus = Literal[
    "available",
    "disabled",
    "absent",
    "stale",
    "unauthorized",
    "integrity_failed",
]


class ProductUser(Protocol):
    @property
    def is_authenticated(self) -> bool: ...

    @property
    def pk(self) -> object: ...


HORIZON_LABELS = {
    "6m": "6 months",
    "12m": "12 months",
    "3y": "3 years",
    "5y": "5 years",
}


@dataclass(frozen=True, slots=True)
class ProductProjection:
    horizon: str
    label: str
    sessions: int
    lower_return: Decimal | None
    median_return: Decimal | None
    upper_return: Decimal | None
    lower_price: Decimal | None
    median_price: Decimal | None
    upper_price: Decimal | None
    zero_drift_lower_return: Decimal | None
    zero_drift_median_return: Decimal | None
    zero_drift_upper_return: Decimal | None
    insufficiency_reason: str

    @property
    def available(self) -> bool:
        return all(
            value is not None
            for value in (
                self.lower_return,
                self.median_return,
                self.upper_return,
                self.lower_price,
                self.median_price,
                self.upper_price,
            )
        )


@dataclass(frozen=True, slots=True)
class ProductCard:
    analysis: StockAnalysis
    decision_prediction: Prediction
    advisory_predictions: tuple[Prediction, ...]
    direction: str
    suggestion: str | None
    decision_horizon_sessions: int
    blocking_reasons: tuple[str, ...]
    allocation_restriction: str
    stock_price_return: Decimal | None
    benchmark_log_momentum: Decimal | None
    relative_log_momentum: Decimal | None
    annualized_volatility: Decimal | None
    benchmark_annualized_volatility: Decimal | None
    relative_volatility: Decimal | None
    relative_volatility_label: str
    maximum_drawdown: Decimal | None
    average_dollar_turnover_20d: Decimal | None
    risk_insufficiency_reasons: tuple[str, ...]
    projections: tuple[ProductProjection, ...]
    target_under_10: bool
    captured_role: str
    current_promotion_eligible: bool
    current_price: Decimal | None
    current_price_date: date | None
    source_period_start: date | None
    source_period_end: date | None
    source_retrieved_at: datetime
    source_available_at: datetime

    @property
    def listing(self) -> Listing:
        return self.analysis.listing


@dataclass(frozen=True, slots=True)
class ProductAdmission:
    symbol: str
    status: str
    reason_code: str
    captured_role: str
    required_closes: int | None
    available_closes: int | None
    missing_closes: int | None
    bootstrap_attempted: bool | None


@dataclass(frozen=True, slots=True)
class ProductRead:
    status: ReaderStatus
    message: str
    run: AnalysisRun | None = None
    cards: tuple[ProductCard, ...] = ()
    admissions: tuple[ProductAdmission, ...] = ()
    provider: str = ""
    evidence_grade: str = ""
    owner_id: str = ""
    verification_code: str = ""

    @property
    def available(self) -> bool:
        return self.status == "available"

    @property
    def admitted_count(self) -> int:
        return sum(item.status == "admitted" for item in self.admissions)

    @property
    def rejected_count(self) -> int:
        return len(self.admissions) - self.admitted_count


@dataclass(frozen=True, slots=True)
class _AuthorizedCandidate:
    run: AnalysisRun
    membership: dict[str, object]
    intake: dict[str, object]


def read_research_product(
    *,
    user: ProductUser,
    store: AssetStore | None = None,
) -> ProductRead:
    """Return the newest expected authorized cohort, or an explicit failure.

    Authorization is established from the registered captured intake before
    the complete run verification.  Once a current cohort is selected, an
    integrity failure suppresses it; the reader does not search backward for
    a more convenient product run and never substitutes a legacy run.
    """
    if not settings.RESEARCH_PRODUCT_ENABLED:
        return ProductRead(
            status="disabled",
            message=(
                "The primary research product is disabled; archived evidence remains available."
            ),
        )
    if not user.is_authenticated:
        return ProductRead(
            status="unauthorized",
            message="Sign in to view an owner-authorized research cohort.",
        )
    try:
        asset_store = store or open_asset_store()
    except RefreshVerificationError as exc:
        return ProductRead(
            status="integrity_failed",
            message="Active research output was suppressed because storage is unavailable.",
            verification_code=exc.reason_code,
        )
    try:
        candidate = _select_authorized_candidate(user=user, store=asset_store)
    except (RefreshVerificationError, KeyError, TypeError, ValueError) as exc:
        code = (
            exc.reason_code
            if isinstance(exc, RefreshVerificationError)
            else "product_selection_integrity_invalid"
        )
        return ProductRead(
            status="integrity_failed",
            message=(
                "Active research output was suppressed because its owner-bound "
                "selection evidence is invalid."
            ),
            verification_code=code,
        )
    if candidate is None:
        return ProductRead(
            status="absent",
            message="No verified active research-product cohort is available for this account.",
        )
    expected_target, _expected_grade = resolve_us_target_date(decision_time=timezone.now())
    if candidate.run.target_date != expected_target:
        return ProductRead(
            status="stale",
            message=(
                "The owner-authorized research cohort does not match the expected "
                "market session, so active output is withheld."
            ),
            run=candidate.run,
            verification_code="product_target_stale",
        )
    try:
        _require_current_display_authorization(candidate.intake, user=user)
        # This is intentionally the only complete verifier call in this read.
        verify_price_product_output(run=candidate.run, store=asset_store, replay=False)
        cards = _build_cards(candidate, store=asset_store)
        admissions = _build_admissions(candidate)
    except RefreshVerificationError as exc:
        return ProductRead(
            status="integrity_failed",
            message=(
                "Active research output was suppressed because its registered evidence "
                "could not be verified."
            ),
            run=candidate.run,
            verification_code=exc.reason_code,
        )
    except ProviderConfigurationError:
        return ProductRead(
            status="unauthorized",
            message="Current provider display authorization does not permit this cohort.",
            run=candidate.run,
        )
    except (InvalidOperation, KeyError, ObjectDoesNotExist, TypeError, ValueError):
        return ProductRead(
            status="integrity_failed",
            message=(
                "Active research output was suppressed because its recorded product "
                "shape or identity is invalid."
            ),
            run=candidate.run,
            verification_code="product_reader_shape_invalid",
        )
    return ProductRead(
        status="available",
        message="Registered research-product output verified.",
        run=candidate.run,
        cards=cards,
        admissions=admissions,
        provider=str(candidate.intake["source_provider"]),
        evidence_grade=candidate.run.universe_snapshot.grade,
        owner_id=str(candidate.intake["owner_id"]),
        verification_code="verified",
    )


def _select_authorized_candidate(
    *, user: ProductUser, store: AssetStore
) -> _AuthorizedCandidate | None:
    expected_owner = DEMO_OWNER_ID if settings.DEMO_MODE else str(user.pk)
    expected_provider = "synthetic_demo" if settings.DEMO_MODE else TWELVE_DATA_PROVIDER
    intake_by_universe: dict[str, DataAsset] = {}
    intake_assets = DataAsset.objects.filter(
        provider="stanstock",
        kind=PRODUCT_INTAKE_KIND,
        subject__contains=f":{expected_owner}:",
    ).order_by("-available_at", "-retrieved_at", "-id")
    for asset in intake_assets:
        subject_parts = asset.subject.split(":")
        if (
            len(subject_parts) != 4
            or subject_parts[0] != PRODUCT_VERSION
            or subject_parts[2] != expected_owner
        ):
            continue
        universe_slug = f"{PRODUCT_VERSION}-{asset.sha256[:16]}"
        if universe_slug in intake_by_universe:
            raise ValueError("Authorized product intake identity is ambiguous")
        intake_by_universe[universe_slug] = asset
    if not intake_by_universe:
        return None
    run = (
        AnalysisRun.objects.select_related("universe_snapshot__universe")
        .filter(
            config_version=PRODUCT_VERSION,
            config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
            status="complete",
            universe_snapshot__universe_id__in=intake_by_universe,
        )
        .order_by("-target_date", "-generated_at", "-id")
        .first()
    )
    if run is None:
        return None
    intake_asset = intake_by_universe[str(run.universe_snapshot.universe_id)]
    intake = json.loads(read_checksummed_bytes(store, intake_asset))
    if (
        not isinstance(intake, dict)
        or intake.get("owner_id") != expected_owner
        or intake.get("source_provider") != expected_provider
        or intake.get("product_version") != PRODUCT_VERSION
    ):
        raise ValueError("Authorized product intake identity is invalid")
    membership = product_membership_payload(run.universe_snapshot, store=store)
    bound_intake = _intake_payload(membership, store=store)
    if bound_intake != intake:
        raise ValueError("Product run membership substituted its authorized intake")
    return _AuthorizedCandidate(run=run, membership=membership, intake=intake)


def _intake_payload(membership: Mapping[str, object], *, store: AssetStore) -> dict[str, object]:
    raw_ref = membership.get("intake")
    if not isinstance(raw_ref, Mapping):
        raise ValueError("Product membership has no intake reference")
    decision_time = datetime.fromisoformat(str(membership["decision_time"]))
    if decision_time.tzinfo is None:
        raise ValueError("Product membership decision time is not timezone-aware")
    asset = resolve_asset_ref(AssetRef.from_json(raw_ref), cutoff=decision_time)
    if asset.provider != "stanstock" or asset.kind != "research_product_intake":
        raise ValueError("Product membership intake identity is invalid")
    payload = json.loads(read_checksummed_bytes(store, asset))
    if not isinstance(payload, dict):
        raise ValueError("Product intake is not an object")
    return payload


def _require_current_display_authorization(
    intake: Mapping[str, object], *, user: ProductUser
) -> None:
    owner_id = str(intake.get("owner_id") or "")
    provider = str(intake.get("source_provider") or "")
    if settings.DEMO_MODE:
        if owner_id != DEMO_OWNER_ID or provider != "synthetic_demo":
            raise ProviderConfigurationError("Demo cannot display a private owner cohort")
        return
    if owner_id != str(user.pk) or provider != TWELVE_DATA_PROVIDER:
        raise ProviderConfigurationError("Research cohort owner does not match")
    record = ProviderRecord.objects.filter(provider=TWELVE_DATA_PROVIDER).first()
    if record is None:
        raise ProviderConfigurationError("Provider display authorization is absent")
    validate_provider_usage(record, owner_id=owner_id)


def _build_cards(candidate: _AuthorizedCandidate, *, store: AssetStore) -> tuple[ProductCard, ...]:
    analyses = list(
        StockAnalysis.objects.select_related(
            "listing__security__company",
            "listing__latest_market_data__source_asset",
        )
        .filter(run=candidate.run)
        .order_by("listing__ticker", "listing_id")
    )
    predictions = list(
        Prediction.objects.select_related("outcome")
        .filter(analysis__run=candidate.run)
        .order_by("listing_id", "evidence_role", "horizon", "id")
    )
    by_analysis: dict[int, list[Prediction]] = {}
    for prediction in predictions:
        by_analysis.setdefault(prediction.analysis_id, []).append(prediction)
    intake = candidate.intake
    core_ids = {UUID(str(value)) for value in _list_value(intake, "core_listing_ids")}
    saved_ids = {UUID(str(value)) for value in _list_value(intake, "saved_listing_ids")}
    cards = [
        _card(
            analysis=analysis,
            rows=by_analysis.get(analysis.id, []),
            captured_role=(
                "saved"
                if analysis.listing_id in saved_ids
                else "core"
                if analysis.listing_id in core_ids
                else "captured"
            ),
            store=store,
        )
        for analysis in analyses
    ]
    return tuple(cards)


def _card(
    *,
    analysis: StockAnalysis,
    rows: list[Prediction],
    captured_role: str,
    store: AssetStore,
) -> ProductCard:
    decision = next(
        row
        for row in rows
        if row.method_version == MOMENTUM_METHOD_VERSION
        and row.evidence_role == Prediction.EvidenceRole.DECISION
    )
    advisory = tuple(
        row
        for row in rows
        if row.method_version == FHS_METHOD_VERSION
        and row.evidence_role == Prediction.EvidenceRole.ADVISORY
    )
    calculation = _mapping(decision.calculation)
    momentum = _optional_mapping(calculation.get("momentum"))
    risk = _mapping(calculation.get("risk"))
    recommendation = _mapping(calculation.get("recommendation"))
    forecast = _mapping(calculation.get("forecast"))
    projections_raw = forecast.get("projections")
    if not isinstance(projections_raw, list):
        raise ValueError("Product forecast projections are missing")
    projections = tuple(_projection(_mapping(raw)) for raw in projections_raw)
    if tuple(item.horizon for item in projections) != ("6m", "12m", "3y", "5y"):
        raise ValueError("Product forecast horizons are invalid")
    source_refs = decision.source_assets
    if not isinstance(source_refs, list) or len(source_refs) != 4:
        raise ValueError("Product source closure is invalid")
    stock_asset = resolve_asset_ref(
        AssetRef.from_json(source_refs[0]), cutoff=analysis.run.generated_at
    )
    # The complete verifier has already checksum-read this exact source.  Keep
    # the dependency explicit so a future reader refactor cannot substitute a
    # different "latest" asset merely to populate dates.
    if not isinstance(stock_asset, DataAsset):
        raise ValueError("Product source asset is invalid")
    target_under_10 = analysis.current_price < Decimal("10")
    current_price: Decimal | None = None
    current_date: date | None = None
    current_valid = False
    try:
        market = analysis.listing.latest_market_data
    except ObjectDoesNotExist:
        market = None
    if market is not None:
        current_price = market.close
        current_date = market.session_date
        current_valid = (
            market.session_date == analysis.run.target_date
            and market.source_asset_id == stock_asset.id
            and market.close == analysis.current_price
            and analysis.listing.currency == "USD"
        )
    suggestion = _optional_string(recommendation.get("suggestion"))
    promotion = (
        suggestion == "buy"
        and not target_under_10
        and current_valid
        and current_price is not None
        and current_price >= Decimal("10")
    )
    return ProductCard(
        analysis=analysis,
        decision_prediction=decision,
        advisory_predictions=advisory,
        direction=_optional_string(recommendation.get("raw_direction")) or "unavailable",
        suggestion=suggestion,
        decision_horizon_sessions=_int_value(recommendation, "decision_horizon_sessions"),
        blocking_reasons=_string_tuple(recommendation.get("blocking_reasons")),
        allocation_restriction=(
            _optional_string(recommendation.get("allocation_restriction")) or ""
        ),
        stock_price_return=_optional_decimal(
            None if momentum is None else momentum.get("stock_price_return")
        ),
        benchmark_log_momentum=_optional_decimal(
            None if momentum is None else momentum.get("benchmark_log_momentum")
        ),
        relative_log_momentum=_optional_decimal(
            None if momentum is None else momentum.get("relative_log_momentum")
        ),
        annualized_volatility=_optional_decimal(risk.get("annualized_volatility")),
        benchmark_annualized_volatility=_optional_decimal(
            risk.get("benchmark_annualized_volatility")
        ),
        relative_volatility=_optional_decimal(risk.get("relative_volatility")),
        relative_volatility_label=str(risk["relative_volatility_label"]),
        maximum_drawdown=_optional_decimal(risk.get("maximum_drawdown")),
        average_dollar_turnover_20d=_optional_decimal(risk.get("average_dollar_turnover_20d")),
        risk_insufficiency_reasons=_string_tuple(risk.get("insufficiency_reasons")),
        projections=projections,
        target_under_10=target_under_10,
        captured_role=captured_role,
        current_promotion_eligible=promotion,
        current_price=current_price,
        current_price_date=current_date,
        source_period_start=stock_asset.period_start,
        source_period_end=stock_asset.period_end,
        source_retrieved_at=stock_asset.retrieved_at,
        source_available_at=stock_asset.available_at,
    )


def _projection(raw: Mapping[str, object]) -> ProductProjection:
    horizon = str(raw["horizon"])
    returns = _optional_mapping(raw.get("ledger_returns"))
    prices = _optional_mapping(raw.get("ledger_prices"))
    sensitivity = _optional_mapping(raw.get("zero_drift_ledger_returns"))
    return ProductProjection(
        horizon=horizon,
        label=HORIZON_LABELS[horizon],
        sessions=_int_value(raw, "sessions"),
        lower_return=_triplet_value(returns, "lower"),
        median_return=_triplet_value(returns, "median"),
        upper_return=_triplet_value(returns, "upper"),
        lower_price=_triplet_value(prices, "lower"),
        median_price=_triplet_value(prices, "median"),
        upper_price=_triplet_value(prices, "upper"),
        zero_drift_lower_return=_triplet_value(sensitivity, "lower"),
        zero_drift_median_return=_triplet_value(sensitivity, "median"),
        zero_drift_upper_return=_triplet_value(sensitivity, "upper"),
        insufficiency_reason=_optional_string(raw.get("insufficiency_reason")) or "",
    )


def _build_admissions(candidate: _AuthorizedCandidate) -> tuple[ProductAdmission, ...]:
    raw_admissions = candidate.membership.get("admissions")
    if not isinstance(raw_admissions, Mapping):
        raise ValueError("Product admission evidence is missing")
    core_symbols = set(_string_list(candidate.intake, "core_symbols"))
    saved_symbols = set(_string_list(candidate.intake, "saved_symbols"))
    admissions: list[ProductAdmission] = []
    for symbol in sorted(raw_admissions):
        raw = _mapping(raw_admissions[symbol])
        history = _optional_mapping(raw.get("history_qualification"))
        bootstrap_value = raw.get("bootstrap_attempted")
        admissions.append(
            ProductAdmission(
                symbol=str(symbol),
                status=str(raw["status"]),
                reason_code=str(raw.get("reason_code") or ""),
                captured_role=(
                    "saved"
                    if symbol in saved_symbols
                    else "core"
                    if symbol in core_symbols
                    else "captured"
                ),
                required_closes=_optional_int(
                    None if history is None else history.get("required_closes")
                ),
                available_closes=_optional_int(
                    None
                    if history is None
                    else history.get(
                        "available_closes",
                        history.get("available_required_closes"),
                    )
                ),
                missing_closes=_optional_int(
                    None
                    if history is None
                    else history.get(
                        "missing_closes",
                        history.get("missing_required_closes"),
                    )
                ),
                bootstrap_attempted=bootstrap_value if isinstance(bootstrap_value, bool) else None,
            )
        )
    return tuple(admissions)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a product mapping")
    return value


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _mapping(value)


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Expected a product string list")
    return tuple(value)


def _list_value(value: Mapping[str, object], key: str) -> list[object]:
    result = value.get(key)
    if not isinstance(result, list):
        raise ValueError(f"Expected product list {key}")
    return result


def _string_list(value: Mapping[str, object], key: str) -> list[str]:
    result = _list_value(value, key)
    if any(not isinstance(item, str) for item in result):
        raise ValueError(f"Expected product string list {key}")
    return [str(item) for item in result]


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a product string")
    return value


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Expected a product decimal")
    return Decimal(str(value))


def _triplet_value(value: Mapping[str, object] | None, key: str) -> Decimal | None:
    return None if value is None else _optional_decimal(value.get(key))


def _int_value(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"Expected product integer {key}")
    return result


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected a product integer")
    return value
