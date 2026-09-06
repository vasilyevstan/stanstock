from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models

from stanstock.data.models import DataAsset, ImmutableEvidenceModel, Listing


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
        ]

    def __str__(self) -> str:
        return f"{self.owner_id}:{self.name}"


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
