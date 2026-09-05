from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from django import forms

from stanstock.data.fx import DEFAULT_MAX_CARRY_DAYS
from stanstock.data.models import Listing, Region, UniverseMembership, UniverseSnapshot
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
