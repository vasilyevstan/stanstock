from __future__ import annotations

from decimal import Decimal

from stanstock.web.templatetags.stanstock import display_label, label_list, price


def test_price_hides_database_precision_without_losing_sub_dollar_detail() -> None:
    assert price(Decimal("328.209990")) == "328.21"
    assert price(Decimal("0.125000")) == "0.125"
    assert price(float("nan")) == "Unavailable"
    assert price(None) == "Unavailable"


def test_display_label_hides_internal_provider_and_evidence_slugs() -> None:
    assert display_label("twelve_data") == "Twelve Data"
    assert display_label("price_history") == "Price history"
    assert display_label("us") == "US"
    assert display_label("empirical_calibrated") == "Probability gate passed — not calibrated"
    assert display_label("empirical_range_only") == "Analog range only — probability withheld"
    assert display_label("custom_status") == "Custom Status"


def test_label_list_formats_provider_collections() -> None:
    assert label_list(["twelve_data", "sec"]) == "Twelve Data, SEC EDGAR"
