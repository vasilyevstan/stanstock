from __future__ import annotations

import uuid
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from django import forms
from django.contrib.auth.models import User
from django.db.models import Q

from stanstock.data.etfs import INVESTABLE_US_ETF_MIC, INVESTABLE_US_ETF_SYMBOL
from stanstock.data.models import (
    Listing,
    Region,
    Security,
)
from stanstock.portfolio.models import Portfolio
from stanstock.portfolio.planner import DEFAULT_MONTHLY_CONTRIBUTION
from stanstock.portfolio.watchlist import (
    TrackedSymbolValidationError,
    normalize_tracked_symbol,
)
from stanstock.research.affordability import PRICE_BANDS, UNDER_10_BAND, price_band_choices
from stanstock.research.models import Recommendation, RiskClass

#: Currencies the demo universe and the ECB reference series can actually
#: reconcile. A code outside this set has no eligible conversion path and
#: would fail at build time, so it is not offered.
CURRENCY_CHOICES: list[tuple[str, str]] = [
    ("", "Not applicable"),
    ("USD", "USD"),
    ("EUR", "EUR"),
    ("GBP", "GBP"),
]

PRODUCT_HORIZON_CHOICES: tuple[tuple[str, str], ...] = (
    ("6m", "6 months"),
    ("12m", "12 months"),
    ("3y", "3 years"),
    ("5y", "5 years"),
)

_NON_UNDER_10_PRICE_BANDS = tuple(
    definition for definition in PRICE_BANDS if definition.slug != UNDER_10_BAND
)
_UNDER_10_PRICE_BAND_LABEL = next(
    label for slug, label in price_band_choices() if slug == UNDER_10_BAND
)
RESEARCH_PRODUCT_PRICE_BAND_CHOICES: tuple[tuple[str, str], ...] = (
    ("", "All reference prices"),
    (UNDER_10_BAND, _UNDER_10_PRICE_BAND_LABEL),
    (
        "at_least_10",
        f"${_NON_UNDER_10_PRICE_BANDS[0].minimum} and above",
    ),
    *((definition.slug, definition.label) for definition in _NON_UNDER_10_PRICE_BANDS),
)


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


class ResearchProductFilterForm(forms.Form):
    q = forms.CharField(required=False, label="Search")
    horizon = forms.ChoiceField(
        required=False,
        choices=PRODUCT_HORIZON_CHOICES,
        initial="6m",
        widget=forms.HiddenInput(),
    )
    direction = forms.ChoiceField(
        required=False,
        choices=[
            ("", "All directions"),
            ("positive", "Positive"),
            ("negative", "Negative"),
            ("mixed", "Mixed"),
            ("unavailable", "Unavailable"),
        ],
        label="Research direction",
    )
    suggestion = forms.ChoiceField(
        required=False,
        choices=[("", "All suggestions"), *Recommendation.choices],
        label="Research suggestion",
    )
    risk = forms.ChoiceField(
        required=False,
        choices=[("", "All relative-volatility bands"), *RiskClass.choices],
        label="Relative volatility",
    )
    price_band = forms.ChoiceField(
        required=False,
        choices=RESEARCH_PRODUCT_PRICE_BAND_CHOICES,
        label="Decision-date reference price",
        widget=forms.HiddenInput(attrs={"class": "bandHiddenInput"}),
    )


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
