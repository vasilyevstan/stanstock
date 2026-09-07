from __future__ import annotations

from pathlib import Path

import pytest

from stanstock.data.sec_config import (
    load_sec_cik_config,
    load_sec_fundamentals_config,
)


def test_released_sec_configs_are_strict_and_complete() -> None:
    fundamentals = load_sec_fundamentals_config()
    mapping = load_sec_cik_config()

    assert fundamentals.config_version == "us-sec-fundamentals-v1"
    assert fundamentals.requests_per_second == 5
    assert fundamentals.companyfacts_lag_retry_days == 7
    assert len(fundamentals.concept_rules) >= 20
    source_rules = fundamentals.source_concept_rules
    assert source_rules[("us-gaap", "ShortTermBorrowings")].canonical_concept == ("short_term_debt")
    assert (
        source_rules[("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent")].canonical_concept
        == "current_long_term_debt"
    )
    assert source_rules[("us-gaap", "LongTermDebt")].canonical_concept == (
        "reported_long_term_debt"
    )
    assert mapping.config_version == "us-sec-cik-v1"
    assert len(mapping.mappings) == 100
    assert mapping.excluded == {}
    assert all(len(item.cik) == 10 and item.cik.isdigit() for item in mapping.mappings.values())


def test_sec_config_rejects_duplicate_source_concepts(tmp_path: Path) -> None:
    config = tmp_path / "sec.yml"
    config.write_text(
        """
schema_version: 1
config_version: duplicate-test
provider: sec
mapping_endpoint: https://example.test/mapping.json
submissions_reconciliation_days: 30
companyfacts_lag_retry_days: 7
requests_per_second: 5
allowed_taxonomies: [us-gaap]
allowed_forms: [10-K]
concepts:
  revenue:
    period_type: duration
    units: [USD]
    source_concepts: [Revenues]
  other_revenue:
    period_type: duration
    units: [USD]
    source_concepts: [Revenues]
""".strip()
    )

    with pytest.raises(ValueError, match="mapped more than once"):
        load_sec_fundamentals_config(config)
