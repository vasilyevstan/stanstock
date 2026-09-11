from __future__ import annotations

import uuid
from typing import Any

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models.functions import Trim, Upper

from stanstock.data.models import DataAsset, ImmutableEvidenceModel, Listing
from stanstock.research.models import AnalysisRun


class TrackedSymbol(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="tracked_symbols",
    )
    symbol = models.CharField(max_length=32)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["symbol", "created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["owner", "symbol"],
                name="unique_owner_tracked_symbol",
            ),
            models.CheckConstraint(
                condition=~models.Q(symbol="") & models.Q(symbol=Upper(Trim(models.F("symbol")))),
                name="tracked_symbol_normalized",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.owner_id}:{self.symbol}"

    def save(self, *args: Any, **kwargs: Any) -> None:
        from stanstock.portfolio.watchlist import normalize_tracked_symbol

        self.symbol = normalize_tracked_symbol(self.symbol)
        super().save(*args, **kwargs)


class Portfolio(models.Model):
    class Currency(models.TextChoices):
        USD = "USD", "USD"
        EUR = "EUR", "EUR"
        GBP = "GBP", "GBP"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="tracked_portfolios",
    )
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True)
    base_currency = models.CharField(max_length=3, choices=Currency.choices)
    cash_balance = models.DecimalField(max_digits=24, decimal_places=6, default=0)
    source_analysis_run = models.ForeignKey(
        AnalysisRun,
        on_delete=models.PROTECT,
        related_name="sample_portfolios",
        null=True,
        blank=True,
    )
    construction_policy = models.CharField(max_length=80, blank=True)
    construction_metadata = models.JSONField(default=dict, blank=True)
    starting_capital = models.DecimalField(
        max_digits=24,
        decimal_places=6,
        null=True,
        blank=True,
    )
    monthly_contribution = models.DecimalField(
        max_digits=24,
        decimal_places=6,
        default=600,
    )
    allow_fractional_shares = models.BooleanField(default=True)
    archived_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name", "created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["owner", "name"],
                name="unique_owner_portfolio_name",
            ),
            models.CheckConstraint(
                condition=models.Q(cash_balance__gte=0),
                name="portfolio_cash_nonnegative",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(starting_capital__isnull=True) | models.Q(starting_capital__gt=0)
                ),
                name="portfolio_starting_capital_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(monthly_contribution__gt=0),
                name="portfolio_monthly_contribution_positive",
            ),
            models.UniqueConstraint(
                fields=["owner", "source_analysis_run"],
                condition=models.Q(
                    source_analysis_run__isnull=False,
                    archived_at__isnull=True,
                ),
                name="unique_active_sample_portfolio_run",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.owner_id}:{self.name}"

    def save(self, *args: Any, **kwargs: Any) -> None:
        self.clean()
        if self.pk:
            original = (
                Portfolio.objects.filter(pk=self.pk)
                .values(
                    "owner_id",
                    "source_analysis_run_id",
                    "name",
                    "description",
                    "base_currency",
                    "cash_balance",
                    "construction_policy",
                    "construction_metadata",
                    "starting_capital",
                    "monthly_contribution",
                    "allow_fractional_shares",
                )
                .first()
            )
            if original and original["source_analysis_run_id"] is not None:
                frozen_values = (
                    ("owner_id", original["owner_id"], self.owner_id),
                    (
                        "source_analysis_run_id",
                        original["source_analysis_run_id"],
                        self.source_analysis_run_id,
                    ),
                    ("name", original["name"], self.name),
                    ("description", original["description"], self.description),
                    ("base_currency", original["base_currency"], self.base_currency),
                    ("cash_balance", original["cash_balance"], self.cash_balance),
                    (
                        "construction_policy",
                        original["construction_policy"],
                        self.construction_policy,
                    ),
                    (
                        "construction_metadata",
                        original["construction_metadata"],
                        self.construction_metadata,
                    ),
                    (
                        "starting_capital",
                        original["starting_capital"],
                        self.starting_capital,
                    ),
                    (
                        "monthly_contribution",
                        original["monthly_contribution"],
                        self.monthly_contribution,
                    ),
                    (
                        "allow_fractional_shares",
                        original["allow_fractional_shares"],
                        self.allow_fractional_shares,
                    ),
                )
                changed = [
                    field
                    for field, original_value, current_value in frozen_values
                    if original_value != current_value
                ]
                if changed:
                    raise ValidationError(
                        "Model portfolio construction is frozen; only archive status may change."
                    )
        super().save(*args, **kwargs)

    def clean(self) -> None:
        super().clean()
        if self.is_model_portfolio:
            if not self.construction_policy:
                raise ValidationError(
                    {"construction_policy": "Model portfolios require a construction policy."}
                )
            if self.starting_capital is None or self.starting_capital <= 0:
                raise ValidationError(
                    {"starting_capital": "Model portfolios require positive starting capital."}
                )
        elif (
            self.construction_policy
            or self.construction_metadata
            or self.starting_capital is not None
        ):
            raise ValidationError("Manual portfolios cannot contain model-portfolio provenance.")

    @property
    def is_model_portfolio(self) -> bool:
        return self.source_analysis_run_id is not None


class PortfolioHolding(models.Model):
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.CASCADE,
        related_name="holdings",
    )
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    quantity = models.DecimalField(max_digits=24, decimal_places=8)
    average_cost = models.DecimalField(max_digits=20, decimal_places=6)
    acquired_on = models.DateField(null=True, blank=True)
    notes = models.CharField(max_length=240, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["listing__ticker"]
        constraints = [
            models.UniqueConstraint(
                fields=["portfolio", "listing"],
                name="unique_portfolio_holding",
            ),
            models.CheckConstraint(
                condition=models.Q(quantity__gt=0),
                name="portfolio_holding_quantity_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(average_cost__gt=0),
                name="portfolio_holding_average_cost_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.portfolio_id}:{self.listing_id}"

    def save(self, *args: Any, **kwargs: Any) -> None:
        if (
            self.portfolio_id
            and Portfolio.objects.filter(
                pk=self.portfolio_id,
                source_analysis_run__isnull=False,
            ).exists()
        ):
            raise ValidationError("Model portfolio holdings are frozen.")
        super().save(*args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> tuple[int, dict[str, int]]:
        if self.portfolio.source_analysis_run_id is not None:
            raise ValidationError("Model portfolio holdings are frozen.")
        return super().delete(*args, **kwargs)


class PortfolioDeposit(ImmutableEvidenceModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.PROTECT,
        related_name="deposits",
    )
    amount = models.DecimalField(max_digits=24, decimal_places=6)
    currency = models.CharField(max_length=3, choices=Portfolio.Currency.choices)
    occurred_at = models.DateTimeField()
    recorded_at = models.DateTimeField(auto_now_add=True)
    idempotency_key = models.UUIDField()
    boundary_snapshot = models.ForeignKey(
        "PortfolioSnapshot",
        on_delete=models.PROTECT,
        related_name="deposit_boundaries",
        null=True,
        blank=True,
    )
    boundary_issue = models.CharField(max_length=240, blank=True)
    cash_balance_after = models.DecimalField(max_digits=24, decimal_places=6)
    note = models.CharField(max_length=240, blank=True)

    class Meta:
        ordering = ["occurred_at", "recorded_at", "id"]
        constraints = [
            models.UniqueConstraint(
                fields=["portfolio", "idempotency_key"],
                name="unique_portfolio_deposit_request",
            ),
            models.CheckConstraint(
                condition=models.Q(amount__gt=0),
                name="portfolio_deposit_amount_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(cash_balance_after__gte=0),
                name="portfolio_deposit_balance_nonnegative",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(boundary_snapshot__isnull=False, boundary_issue="")
                    | (models.Q(boundary_snapshot__isnull=True) & ~models.Q(boundary_issue=""))
                ),
                name="portfolio_deposit_boundary_state",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.portfolio_id}:{self.occurred_at}:{self.amount}"


class PortfolioPlanExecution(ImmutableEvidenceModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.PROTECT,
        related_name="plan_executions",
    )
    idempotency_key = models.UUIDField()
    plan_hash = models.CharField(max_length=64)
    policy_version = models.CharField(max_length=80)
    executed_at = models.DateTimeField()
    recorded_at = models.DateTimeField(auto_now_add=True)
    starting_nav = models.DecimalField(max_digits=24, decimal_places=6)
    starting_cash = models.DecimalField(max_digits=24, decimal_places=6)
    ending_cash = models.DecimalField(max_digits=24, decimal_places=6)
    fractional_shares = models.BooleanField()
    metadata = models.JSONField(default=dict)

    class Meta:
        ordering = ["-executed_at", "-recorded_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["portfolio", "idempotency_key"],
                name="unique_portfolio_plan_execution_request",
            ),
            models.CheckConstraint(
                condition=models.Q(starting_nav__gte=0),
                name="portfolio_execution_nav_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(starting_cash__gte=0),
                name="portfolio_execution_start_cash_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(ending_cash__gte=0),
                name="portfolio_execution_end_cash_nonnegative",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.portfolio_id}:{self.executed_at}:{self.plan_hash[:12]}"


class PortfolioPurchase(ImmutableEvidenceModel):
    class Role(models.TextChoices):
        CORE = "core", "Core ETF"
        SATELLITE = "satellite", "Stock satellite"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    execution = models.ForeignKey(
        PortfolioPlanExecution,
        on_delete=models.PROTECT,
        related_name="purchases",
    )
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    source_asset = models.ForeignKey(DataAsset, on_delete=models.PROTECT)
    source_session_date = models.DateField()
    role = models.CharField(max_length=16, choices=Role.choices)
    quantity = models.DecimalField(max_digits=24, decimal_places=8)
    price = models.DecimalField(max_digits=20, decimal_places=6)
    amount = models.DecimalField(max_digits=24, decimal_places=6)
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["role", "listing__ticker"]
        constraints = [
            models.UniqueConstraint(
                fields=["execution", "listing"],
                name="unique_execution_purchase_listing",
            ),
            models.CheckConstraint(
                condition=models.Q(quantity__gt=0),
                name="portfolio_purchase_quantity_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(price__gt=0),
                name="portfolio_purchase_price_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(amount__gt=0),
                name="portfolio_purchase_amount_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.execution_id}:{self.listing_id}:{self.quantity}"


class PortfolioPerformanceBaseline(ImmutableEvidenceModel):
    class Reason(models.TextChoices):
        MANUAL_HOLDING_CHANGE = "manual_holding_change", "Manual holding change"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.PROTECT,
        related_name="performance_baselines",
    )
    snapshot = models.ForeignKey(
        "PortfolioSnapshot",
        on_delete=models.PROTECT,
        related_name="performance_baselines",
        null=True,
        blank=True,
    )
    reason = models.CharField(max_length=40, choices=Reason)
    boundary_issue = models.CharField(max_length=240, blank=True)
    note = models.CharField(max_length=240, blank=True)
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-recorded_at", "-id"]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(snapshot__isnull=False, boundary_issue="")
                    | (models.Q(snapshot__isnull=True) & ~models.Q(boundary_issue=""))
                ),
                name="portfolio_performance_baseline_state",
            )
        ]

    def __str__(self) -> str:
        return f"{self.portfolio_id}:{self.reason}:{self.recorded_at}"


class PortfolioSnapshot(ImmutableEvidenceModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.PROTECT,
        related_name="snapshots",
    )
    as_of_date = models.DateField()
    recorded_at = models.DateTimeField(auto_now_add=True)
    oldest_price_date = models.DateField(null=True, blank=True)
    newest_price_date = models.DateField(null=True, blank=True)
    base_currency = models.CharField(max_length=3, choices=Portfolio.Currency.choices)
    cash_balance = models.DecimalField(max_digits=24, decimal_places=6)
    securities_value = models.DecimalField(max_digits=24, decimal_places=6)
    total_value = models.DecimalField(max_digits=24, decimal_places=6)
    cost_basis = models.DecimalField(max_digits=24, decimal_places=6)
    unrealized_gain = models.DecimalField(max_digits=24, decimal_places=6)
    return_pct = models.DecimalField(max_digits=16, decimal_places=8, null=True)
    input_hash = models.CharField(max_length=64)
    code_revision = models.CharField(max_length=64)
    return_definition = models.CharField(
        max_length=48,
        default="split_adjusted_price_return",
    )
    dividends_included = models.BooleanField(default=False)
    corporate_action_warnings = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-as_of_date", "-recorded_at", "-id"]
        constraints = [
            models.UniqueConstraint(
                fields=["portfolio", "as_of_date", "input_hash"],
                name="unique_portfolio_snapshot_input",
            ),
            models.CheckConstraint(
                condition=models.Q(cash_balance__gte=0),
                name="portfolio_snapshot_cash_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(securities_value__gte=0),
                name="portfolio_snapshot_securities_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(total_value__gte=0),
                name="portfolio_snapshot_total_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(cost_basis__gte=0),
                name="portfolio_snapshot_cost_basis_nonnegative",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.portfolio_id}:{self.as_of_date}:{self.input_hash[:12]}"


class PortfolioSnapshotHolding(ImmutableEvidenceModel):
    snapshot = models.ForeignKey(
        PortfolioSnapshot,
        on_delete=models.CASCADE,
        related_name="positions",
    )
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    source_asset = models.ForeignKey(DataAsset, on_delete=models.PROTECT)
    source_session_date = models.DateField()
    quantity = models.DecimalField(max_digits=24, decimal_places=8)
    average_cost = models.DecimalField(max_digits=20, decimal_places=6)
    price = models.DecimalField(max_digits=20, decimal_places=6)
    cost_basis = models.DecimalField(max_digits=24, decimal_places=6)
    market_value = models.DecimalField(max_digits=24, decimal_places=6)
    unrealized_gain = models.DecimalField(max_digits=24, decimal_places=6)
    corporate_action_suspected = models.BooleanField(default=False)

    class Meta:
        ordering = ["listing__ticker"]
        constraints = [
            models.UniqueConstraint(
                fields=["snapshot", "listing"],
                name="unique_portfolio_snapshot_holding",
            ),
            models.CheckConstraint(
                condition=models.Q(quantity__gt=0),
                name="portfolio_snapshot_quantity_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(average_cost__gt=0),
                name="portfolio_snapshot_average_cost_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(price__gt=0),
                name="portfolio_snapshot_price_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(cost_basis__gt=0),
                name="portfolio_snapshot_holding_cost_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(market_value__gt=0),
                name="portfolio_snapshot_holding_value_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.snapshot_id}:{self.listing_id}"
