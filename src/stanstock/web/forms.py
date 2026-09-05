from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from django import forms

from stanstock.data.models import Listing, Region, UniverseMembership, UniverseSnapshot
from stanstock.research.models import Recommendation, RiskClass
from stanstock.simulation.models import SimulationDefinition


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
    base_currency = forms.ChoiceField(
        required=False,
        choices=[("", "Infer when unambiguous"), ("USD", "USD"), ("EUR", "EUR"), ("GBP", "GBP")],
        label="Native Currency",
        help_text=(
            "Required for mixed-currency universe backtests. Portfolio selections must all "
            "use this currency."
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

        if start_date and end_date and start_date > end_date:
            self.add_error("start_date", "Start date cannot be after end date.")

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
