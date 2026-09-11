from __future__ import annotations

import uuid
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from django import forms
from django.contrib.auth.models import User
from django.db.models import Q

from stanstock.data.etfs import INVESTABLE_US_ETF_MIC, INVESTABLE_US_ETF_SYMBOL
from stanstock.data.fx import DEFAULT_MAX_CARRY_DAYS
from stanstock.data.models import (
    Listing,
    Region,
    Security,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.portfolio.models import Portfolio
from stanstock.portfolio.planner import DEFAULT_MONTHLY_CONTRIBUTION
from stanstock.portfolio.service import (
    SAMPLE_PORTFOLIO_DEFAULT_CAPITAL,
    SAMPLE_PORTFOLIO_DEFAULT_TOP_N,
    SAMPLE_PORTFOLIO_MAX_TOP_N,
)
from stanstock.portfolio.watchlist import (
    TrackedSymbolValidationError,
    normalize_tracked_symbol,
)
from stanstock.research.affordability import price_band_choices
from stanstock.research.models import Recommendation, RiskClass
from stanstock.simulation.models import SimulationDefinition

#: Currencies the demo universe and the ECB reference series can actually
#: reconcile. A code outside this set has no eligible conversion path and
#: would fail at build time, so it is not offered.
CURRENCY_CHOICES: list[tuple[str, str]] = [
    ("", "Not applicable"),
    ("USD", "USD"),
    ("EUR", "EUR"),
    ("GBP", "GBP"),
]


class OpportunityFilterForm(forms.Form):
    q = forms.CharField(required=False, label="Search")
    region = forms.ChoiceField(
        required=False,
        choices=[("", "All regions"), *Region.choices],
    )
    recommendation = forms.ChoiceField(
        required=False,
        choices=[("", "All"), *Recommendation.choices],
    )
    risk = forms.ChoiceField(
        required=False,
        choices=[("", "All"), *RiskClass.choices],
    )
    price_band = forms.ChoiceField(
        required=False,
        choices=[("", "All price bands"), *price_band_choices()],
        label="Price band",
    )
    country = forms.ChoiceField(required=False)
    exchange = forms.ChoiceField(required=False)
    sector = forms.ChoiceField(required=False)
    min_score = forms.DecimalField(
        required=False,
        min_value=Decimal(0),
        max_value=Decimal(100),
        decimal_places=2,
    )
    min_confidence = forms.DecimalField(
        required=False,
        min_value=Decimal(0),
        max_value=Decimal(100),
        decimal_places=2,
    )

    def configure_choices(
        self,
        *,
        countries: Iterable[str] = (),
        exchanges: Iterable[str] = (),
        sectors: Iterable[str] = (),
    ) -> None:
        choice_sets = {
            "country": ("All countries", countries),
            "exchange": ("All exchanges", exchanges),
            "sector": ("All sectors", sectors),
        }
        for field_name, (empty_label, values) in choice_sets.items():
            field = self.fields[field_name]
            if not isinstance(field, forms.ChoiceField):
                raise TypeError(f"{field_name} must be a ChoiceField")
            field.choices = [
                ("", empty_label),
                *((value, value) for value in values),
            ]


class TrackedSymbolForm(forms.Form):
    symbol = forms.CharField(
        max_length=32,
        label="Symbol",
        help_text="US common stock or ADR symbol, for example TEST.",
        widget=forms.TextInput(
            attrs={
                "autocomplete": "off",
                "autocapitalize": "characters",
                "placeholder": "TEST",
            }
        ),
    )

    def clean_symbol(self) -> str:
        try:
            return normalize_tracked_symbol(str(self.cleaned_data["symbol"]))
        except TrackedSymbolValidationError as exc:
            raise forms.ValidationError(str(exc)) from exc


class PortfolioForm(forms.ModelForm):  # type: ignore[type-arg]
    class Meta:
        model = Portfolio
        fields = (
            "name",
            "description",
            "base_currency",
            "cash_balance",
            "monthly_contribution",
            "allow_fractional_shares",
        )
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
        }
        labels = {
            "cash_balance": "Starting cash",
            "monthly_contribution": "Monthly contribution",
            "allow_fractional_shares": "Allow fractional shares",
        }
        help_texts = {
            "cash_balance": (
                "Initial uninvested cash. After creation, add cash through immutable "
                "deposit events."
            ),
            "monthly_contribution": (
                "Default amount for the contribution form; the planner also carries "
                "unused cash forward."
            ),
            "allow_fractional_shares": (
                "Enabled by default. Disable to preview and record whole-share "
                "purchases with residual cash carried forward."
            ),
            "base_currency": (
                "Tracked holdings must currently trade in this currency; portfolio FX "
                "conversion is intentionally not implicit."
            ),
        }

    def __init__(
        self,
        *args: Any,
        owner: User,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.owner = owner
        self.instance.owner_id = owner.pk
        if self.instance._state.adding:
            self.fields["monthly_contribution"].initial = DEFAULT_MONTHLY_CONTRIBUTION
        else:
            cash_field = self.fields["cash_balance"]
            cash_field.disabled = True
            cash_field.label = "Current cash"
            cash_field.help_text = "Managed by immutable deposits and confirmed planner purchases."

    def clean_name(self) -> str:
        name = str(self.cleaned_data["name"]).strip()
        duplicate = Portfolio.objects.filter(owner_id=self.owner.pk, name__iexact=name)
        if not self.instance._state.adding:
            duplicate = duplicate.exclude(pk=self.instance.pk)
        if duplicate.exists():
            raise forms.ValidationError("You already have a portfolio with this name.")
        return name

    def clean_base_currency(self) -> str:
        currency = str(self.cleaned_data["base_currency"])
        if (
            not self.instance._state.adding
            and self.instance.holdings.exclude(listing__currency=currency).exists()
        ):
            raise forms.ValidationError(
                "Remove holdings in other currencies before changing the base currency."
            )
        if not self.instance._state.adding:
            original_currency = (
                Portfolio.objects.filter(pk=self.instance.pk)
                .values_list(
                    "base_currency",
                    flat=True,
                )
                .first()
            )
            if (
                original_currency is not None
                and currency != original_currency
                and self.instance.deposits.exists()
            ):
                raise forms.ValidationError(
                    "The base currency cannot change after deposit tracking begins."
                )
        return currency


class PortfolioHoldingForm(forms.Form):
    listing = forms.ModelChoiceField(
        queryset=Listing.objects.none(),
        label="Security",
    )
    quantity = forms.DecimalField(
        max_digits=24,
        decimal_places=8,
        min_value=Decimal("0.00000001"),
    )
    average_cost = forms.DecimalField(
        max_digits=20,
        decimal_places=6,
        min_value=Decimal("0.000001"),
        label="Average cost per share",
    )
    acquired_on = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={"type": "date"}),
        label="Acquired on",
    )
    notes = forms.CharField(
        required=False,
        max_length=240,
        widget=forms.Textarea(attrs={"rows": 2}),
    )

    def __init__(self, *args: Any, portfolio: Portfolio, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        listing_field = self.fields["listing"]
        if not isinstance(listing_field, forms.ModelChoiceField):
            raise TypeError("listing must be a ModelChoiceField")
        listing_field.queryset = (
            Listing.objects.filter(
                is_active=True,
                currency=portfolio.base_currency,
                latest_market_data__isnull=False,
            )
            .filter(
                ~Q(security__security_type=Security.SecurityType.ETF)
                | Q(
                    security__security_type=Security.SecurityType.ETF,
                    ticker=INVESTABLE_US_ETF_SYMBOL,
                    provider_symbol=INVESTABLE_US_ETF_SYMBOL,
                    exchange_mic=INVESTABLE_US_ETF_MIC,
                    region=Region.US,
                    is_primary=True,
                )
            )
            .select_related("security__company")
            .order_by("ticker")
        )


class PortfolioDepositForm(forms.Form):
    amount = forms.DecimalField(
        max_digits=24,
        decimal_places=6,
        min_value=Decimal("0.000001"),
        label="External deposit",
    )
    note = forms.CharField(
        required=False,
        max_length=240,
        widget=forms.TextInput(attrs={"placeholder": "Optional note"}),
    )
    idempotency_key = forms.UUIDField(widget=forms.HiddenInput())

    def __init__(self, *args: Any, portfolio: Portfolio, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.is_bound:
            self.fields["amount"].initial = portfolio.monthly_contribution
            self.fields["idempotency_key"].initial = uuid.uuid4()


class PortfolioPlanConfirmationForm(forms.Form):
    plan_hash = forms.CharField(max_length=64, widget=forms.HiddenInput())
    idempotency_key = forms.UUIDField(widget=forms.HiddenInput())

    def __init__(
        self,
        *args: Any,
        plan_hash: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.is_bound:
            self.fields["plan_hash"].initial = plan_hash
            self.fields["idempotency_key"].initial = uuid.uuid4()


class SamplePortfolioForm(forms.Form):
    starting_capital = forms.DecimalField(
        max_digits=24,
        decimal_places=2,
        min_value=Decimal("100.00"),
        initial=SAMPLE_PORTFOLIO_DEFAULT_CAPITAL,
        label="Starting capital",
        help_text=(
            "Reference capital for the frozen model basket. This is research tracking, "
            "not a record of an executed trade."
        ),
    )
    top_n = forms.IntegerField(
        min_value=1,
        max_value=SAMPLE_PORTFOLIO_MAX_TOP_N,
        initial=SAMPLE_PORTFOLIO_DEFAULT_TOP_N,
        label="Number of opportunities",
        help_text=(
            "Select the highest-ranked eligible opportunities and equal-weight the available set."
        ),
    )


class SimulationForm(forms.Form):
    name = forms.CharField(
        max_length=160,
        initial="Simulation",
        label="Simulation Name",
    )
    mode = forms.ChoiceField(
        choices=SimulationDefinition.Mode.choices,
        initial=SimulationDefinition.Mode.BACKTEST,
        label="Mode",
    )
    snapshot = forms.ModelChoiceField(
        queryset=UniverseSnapshot.objects.select_related("universe").order_by("-as_of_date"),
        label="Universe Snapshot",
    )
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={"type": "date"}),
        label="Start Date",
    )
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={"type": "date"}),
        label="End Date",
    )
    starting_capital = forms.DecimalField(
        max_digits=14,
        decimal_places=2,
        initial=Decimal("100000.00"),
        min_value=Decimal("1.00"),
        label="Starting Capital",
    )
    transaction_cost_bps = forms.DecimalField(
        max_digits=6,
        decimal_places=2,
        initial=Decimal("10.00"),
        min_value=Decimal("0.00"),
        label="Transaction Cost (bps)",
    )
    slippage_bps = forms.DecimalField(
        max_digits=6,
        decimal_places=2,
        initial=Decimal("5.00"),
        min_value=Decimal("0.00"),
        label="Slippage (bps)",
    )
    top_n = forms.IntegerField(
        required=False,
        min_value=1,
        initial=10,
        label="Top N (for Backtest)",
    )
    selected_listings = forms.ModelMultipleChoiceField(
        queryset=Listing.objects.select_related("security__company").order_by("ticker"),
        required=False,
        label="Selected Listings (for Portfolio)",
        to_field_name="id",
    )
    benchmark_subject = forms.CharField(
        required=False,
        label="Benchmark Subject (optional)",
    )
    benchmark_currency = forms.ChoiceField(
        required=False,
        choices=CURRENCY_CHOICES,
        label="Benchmark Currency",
        help_text=(
            "Required when the run converts currencies; a price series carries no "
            "denomination of its own."
        ),
    )
    base_currency = forms.ChoiceField(
        required=False,
        choices=[("", "Infer when unambiguous"), *CURRENCY_CHOICES[1:]],
        label="Base Currency",
        help_text=(
            "Every value is reported in this currency. Required when the selection spans "
            "several native currencies; conversion uses dated rates available by the "
            "decision boundary."
        ),
    )
    restrict_native_currency = forms.ChoiceField(
        required=False,
        choices=[("", "Use every eligible listing"), *CURRENCY_CHOICES[1:]],
        label="Restrict To Native Currency",
        help_text="Optionally exclude listings that are not natively in this currency.",
    )
    fx_max_carry_days = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=DEFAULT_MAX_CARRY_DAYS,
        initial=DEFAULT_MAX_CARRY_DAYS,
        label="FX Carry Limit (days)",
        help_text=(
            "Longest gap between an FX observation and the date it is carried forward to "
            f"(0 to {DEFAULT_MAX_CARRY_DAYS}). A longer gap fails instead of pricing from a "
            "stale rate."
        ),
    )

    def clean(self) -> dict[str, Any]:
        cleaned_data = super().clean()
        if cleaned_data is None:
            return {}
        mode = cleaned_data.get("mode")
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")
        top_n = cleaned_data.get("top_n")
        selected_listings = cleaned_data.get("selected_listings")
        base_currency = cleaned_data.get("base_currency")
        restrict_currency = cleaned_data.get("restrict_native_currency")

        if start_date and end_date and start_date > end_date:
            self.add_error("start_date", "Start date cannot be after end date.")

        if cleaned_data.get("benchmark_currency") and not cleaned_data.get("benchmark_subject"):
            self.add_error(
                "benchmark_currency",
                "Choose a benchmark subject before naming its currency; there is nothing to "
                "denominate otherwise.",
            )

        if mode == SimulationDefinition.Mode.BACKTEST:
            if not top_n or top_n <= 0:
                self.add_error("top_n", "Top N is required for backtest mode.")
        elif mode == SimulationDefinition.Mode.PORTFOLIO:
            if not selected_listings:
                self.add_error(
                    "selected_listings",
                    "Selected listings are required for portfolio mode.",
                )
            else:
                selected_uuids = [listing.id for listing in selected_listings]
                if len(selected_uuids) != len(set(selected_uuids)):
                    self.add_error(
                        "selected_listings",
                        "Duplicate listing UUIDs detected in selection.",
                    )
                selected_currencies = sorted(
                    {listing.currency.upper() for listing in selected_listings}
                )
                if len(selected_currencies) > 1 and not base_currency:
                    self.add_error(
                        "base_currency",
                        "Choose the base currency to convert this multi-currency selection "
                        f"into. Selected listings use: {', '.join(selected_currencies)}.",
                    )
                if restrict_currency:
                    excluded = sorted(
                        f"{listing.ticker} ({listing.currency.upper()})"
                        for listing in selected_listings
                        if listing.currency.upper() != restrict_currency
                    )
                    if excluded:
                        self.add_error(
                            "restrict_native_currency",
                            f"This restriction would exclude selected listings: "
                            f"{', '.join(excluded)}. Drop the restriction or remove them "
                            "from the selection.",
                        )
                snapshot = cleaned_data.get("snapshot")
                if snapshot:
                    eligible_uuids = set(
                        UniverseMembership.objects.filter(
                            snapshot=snapshot, eligible=True
                        ).values_list("listing_id", flat=True)
                    )
                    ineligible = [str(lid) for lid in selected_uuids if lid not in eligible_uuids]
                    if ineligible:
                        self.add_error(
                            "selected_listings",
                            f"Selected listings must be eligible members of the chosen snapshot. "
                            f"Ineligible: {ineligible}",
                        )
                cleaned_data["parsed_listing_ids"] = selected_uuids

        return cleaned_data
