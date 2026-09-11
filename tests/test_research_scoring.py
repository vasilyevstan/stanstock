from __future__ import annotations

import hashlib
import inspect
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

import stanstock.research.config as config_module
from short_frozen_base import base_short_modules, read_base_bytes
from stanstock.data.management.config_loader import default_us_scoring_config_path
from stanstock.research.config import (
    HORIZONS,
    V3_EFFECTIVE_CONFIG_HASH,
    ScoringConfig,
    config_hash,
    load_scoring_config,
)
from stanstock.research.scoring import (
    aggregate_score,
    assess_risk,
    decide_recommendation,
    score_components,
)
from stanstock.research.types import (
    ComponentScores,
    IndicatorResult,
    ResearchValues,
    RiskAssessment,
    Scenario,
)


def _rich_indicators() -> IndicatorResult:
    values = {
        "return_20d": 0.08,
        "return_63d": 0.18,
        "return_126d": 0.24,
        "close_vs_sma_50": 0.06,
        "close_vs_sma_200": 0.12,
        "rsi_14": 58.0,
        "macd_histogram": 1.0,
        "macd_histogram_pct": 0.01,
        "52w_position": 0.85,
        "annualized_volatility": 0.22,
        "downside_volatility": 0.16,
        "max_drawdown": -0.12,
        "beta": 1.05,
        "abnormal_volume": 1.1,
        "abnormal_volume_strict": 1.1,
        "avg_volume_20d": 1_500_000,
        "avg_dollar_volume_20d": 150_000_000,
        "relative_return_20d": 0.02,
        "relative_return_63d": 0.07,
        "relative_return_252d": 0.11,
    }
    return IndicatorResult(values=values, observation_count=260, last_date=date(2026, 9, 4))


def _rich_fundamentals() -> ResearchValues:
    return ResearchValues(
        values={
            "net_margin": 0.18,
            "operating_margin": 0.22,
            "free_cash_flow_margin": 0.16,
            "cash_to_debt": 1.8,
            "debt_to_equity": 0.35,
            "interest_coverage": 9.0,
            "free_cash_flow_consistency": 1.0,
            "revenue_growth": 0.18,
            "net_income_growth": 0.22,
            "free_cash_flow_growth": 0.16,
            "pe_ratio": 14.0,
            "ps_ratio": 2.0,
            "pb_ratio": 2.0,
            "ev_to_sales": 2.2,
            "ev_to_ebitda": 8.0,
            "free_cash_flow_yield": 0.08,
        }
    )


def _evidenced_scenarios() -> dict[str, Scenario]:
    return {
        "short": Scenario(
            bear=-0.02,
            base=0.03,
            bull=0.08,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
        "medium": Scenario(
            bear=-0.12,
            base=0.10,
            bull=0.25,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
        "long": Scenario(
            bear=-0.20,
            base=0.35,
            bull=0.80,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
    }


def test_config_horizon_weights_are_versioned_and_sum_to_one() -> None:
    config = load_scoring_config()

    assert config.version == "default-v1"
    assert config.horizon_weights == {
        "short": {
            "quality": 0.05,
            "growth": 0.05,
            "valuation": 0.05,
            "momentum_technical": 0.40,
            "risk_liquidity": 0.25,
            "market_sector": 0.20,
        },
        "medium": {
            "quality": 0.20,
            "growth": 0.20,
            "valuation": 0.20,
            "momentum_technical": 0.15,
            "risk_liquidity": 0.15,
            "market_sector": 0.10,
        },
        "long": {
            "quality": 0.30,
            "growth": 0.25,
            "valuation": 0.20,
            "momentum_technical": 0.05,
            "risk_liquidity": 0.15,
            "market_sector": 0.05,
        },
    }
    assert config.recommendation.buy_min_avg_volume_20d == 100_000
    assert config.recommendation.buy_max_bear_downside == {
        "short": -0.08,
        "medium": -0.30,
        "long": -0.50,
    }
    for horizon in HORIZONS:
        assert sum(config.horizon_weights[horizon].values()) == 1.0
    assert (
        config.horizon_weights["short"]["momentum_technical"]
        > config.horizon_weights["long"]["momentum_technical"]
    )
    assert config.horizon_weights["long"]["quality"] > config.horizon_weights["short"]["quality"]


def test_us_price_baseline_uses_short_price_only_score() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v1.yml"
    )
    components = ComponentScores(
        components={
            "quality": 100,
            "growth": 100,
            "valuation": 100,
            "momentum_technical": 80,
            "risk_liquidity": 60,
            "market_sector": 40,
        },
        factor_scores={},
        missing={},
        coverage=1.0,
    )

    aggregate = aggregate_score(components, config)

    assert config.analysis_mode == "price_only_baseline"
    assert config.overall_horizon == "short"
    assert config.supported_horizons == ("short",)
    assert aggregate.horizon_scores["short"] == 66
    assert aggregate.overall == aggregate.horizon_scores["short"]


def test_v1_config_hash_and_legacy_factor_policy_remain_unchanged() -> None:
    path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v1.yml"
    config = load_scoring_config(path)

    assert config_hash(config) == (
        "8bd3adebc56bd70cc0b924b8d069c28f971b9d22000ca0811c1eaeb1d9420d83"
    )
    assert config.factor_policy.macd_indicator == "macd_histogram"
    assert config.factor_policy.abnormal_volume_indicator == "abnormal_volume"
    assert config.factor_policy.liquidity_indicator == "avg_volume_20d"
    assert config.factor_policy.strict_finite_inputs is False
    assert config.recommendation.buy_min_liquidity_20d == 100_000


def test_v2_config_uses_normalized_macd_and_dollar_liquidity() -> None:
    path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    config = load_scoring_config(path)

    assert config.version == "us-price-baseline-v2"
    assert config.factor_policy.macd_indicator == "macd_histogram_pct"
    assert config.factor_policy.macd_score_low == -0.02
    assert config.factor_policy.macd_score_high == 0.02
    assert config.factor_policy.abnormal_volume_indicator == "abnormal_volume_strict"
    assert config.factor_policy.liquidity_indicator == "avg_dollar_volume_20d"
    assert config.factor_policy.liquidity_score_low == 1_000_000
    assert config.factor_policy.liquidity_score_high == 50_000_000
    assert config.factor_policy.strict_finite_inputs is True
    assert config.recommendation.buy_min_liquidity_20d == 5_000_000


def test_v2_config_hash_and_production_default_path_remain_pinned() -> None:
    """`us-price-baseline-v2` is the production default resolved by
    `default_us_scoring_config_path()` (see `live_us.py`); pin its literal
    effective hash so an untracked or silently edited default cannot change
    frozen live/backfill behavior. Comparing two loads of the same current
    file is not a pin -- the hash below is the exact expected value."""
    path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    config = load_scoring_config(path)
    expected_hash = "43cc0ee0e29f79dec4ad8d8a91e43e7b3df1d368dbc385a116fcc6ff05f18e9b"

    assert config_hash(config) == expected_hash

    default_path = default_us_scoring_config_path()
    assert default_path == path
    default_config = load_scoring_config(default_path)
    assert config_hash(default_config) == expected_hash


@pytest.mark.parametrize(
    ("repository_path", "expected_hash"),
    (
        (
            "config/scoring/us-price-baseline-v1.yml",
            "8bd3adebc56bd70cc0b924b8d069c28f971b9d22000ca0811c1eaeb1d9420d83",
        ),
        (
            "config/scoring/us-price-baseline-v2.yml",
            "43cc0ee0e29f79dec4ad8d8a91e43e7b3df1d368dbc385a116fcc6ff05f18e9b",
        ),
    ),
    ids=("v1", "v2"),
)
def test_legacy_root_self_merge_matches_exact_base_mapping_payload_and_hash(
    tmp_path: Path,
    repository_path: str,
    expected_hash: str,
) -> None:
    path = tmp_path / Path(repository_path).name
    source = read_base_bytes(repository_path).decode("utf-8")
    path.write_text(f"&root\n{source}\n<<: *root\n", encoding="utf-8")

    with base_short_modules() as base:
        base_config = base.config.load_scoring_config(path)
        base_payload = json.dumps(
            base_config.raw,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        base_hash = base.config.config_hash(base_config)

    head_config = load_scoring_config(path)
    head_payload = json.dumps(
        head_config.raw,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    assert head_config.raw == base_config.raw
    assert head_payload == base_payload
    assert config_hash(head_config) == base_hash == expected_hash


@pytest.mark.parametrize(
    "document",
    (
        "version: us-price-baseline-v2\n<<: malformed\n",
        "version: us-price-baseline-v2\n<<: [{legacy: true}, malformed]\n",
    ),
    ids=("scalar", "mixed-sequence"),
)
def test_legacy_malformed_merge_exception_matches_exact_base(
    tmp_path: Path,
    document: str,
) -> None:
    path = tmp_path / "legacy-malformed-merge.yml"
    path.write_text(document, encoding="utf-8")

    with base_short_modules() as base:
        with pytest.raises(yaml.constructor.ConstructorError) as base_error:
            base.config.load_scoring_config(path)

    with pytest.raises(yaml.constructor.ConstructorError) as head_error:
        load_scoring_config(path)

    assert type(head_error.value) is type(base_error.value)
    assert str(head_error.value) == str(base_error.value)


def test_v3_config_bytes_effective_hash_and_typed_maps_are_exact() -> None:
    path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    config = load_scoring_config(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "392dce367ea8772a09a2f4a13c2a45f1d147b2053da2646719661845d408ae4c"
    )
    assert V3_EFFECTIVE_CONFIG_HASH == (
        "aee8ae47cc82092f2167d778381e55bccc7c393913a997afd1ca391ca182585c"
    )
    assert config_hash(config) == V3_EFFECTIVE_CONFIG_HASH
    assert config.version == "us-price-baseline-v3"
    assert config.short_scoring is not None
    assert len(config.short_scoring.factor_maps) == 16
    assert len(config.short_scoring.risk_penalty_maps) == 4
    assert config.short_scoring.rsi.convention == "cutler_sma"
    assert config.short_scoring.risk_window.sessions == 252
    assert config.short_scoring.risk_window.annualization_sessions == 252
    assert config.short_scoring.beta_roles.factor_score == "excluded"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update({"unknown": 1}),
        lambda raw: raw["coverage"].pop("confidence_cap"),
        lambda raw: raw["short_scoring"]["rsi"].update({"unknown": 1}),
        lambda raw: raw["short_scoring"]["factor_maps"]["momentum.rsi"].pop("high"),
        lambda raw: raw["short_scoring"]["risk_penalty_maps"].pop("absolute_beta"),
        lambda raw: raw["recommendation"].update({"buy_min_score": True}),
        lambda raw: raw["freshness"].update({"fresh_days": 1.5}),
        lambda raw: raw["short_scoring"]["factor_maps"]["momentum.rsi"].update(
            {"low": float("nan")}
        ),
        lambda raw: raw["short_scoring"]["factor_maps"]["momentum.rsi"].update(
            {"low": 70.0, "high": 30.0}
        ),
        lambda raw: raw["short_scoring"]["factor_maps"]["momentum.rsi"].update(
            {"kind": "linear_lower"}
        ),
        lambda raw: raw["short_scoring"]["rsi"].update({"convention": "wilder"}),
        lambda raw: raw.update({"version": "us-price-baseline-v4"}),
    ],
    ids=[
        "unknown-top",
        "missing-nested",
        "unknown-recursive",
        "missing-map-field",
        "missing-risk-map",
        "boolean-number",
        "noninteger",
        "nonfinite",
        "invalid-bounds",
        "key-inconsistent-kind",
        "invalid-literal",
        "invalid-version",
    ],
)
def test_v3_config_rejects_malformed_recursive_schema(mutate) -> None:
    path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw)

    with pytest.raises(ValueError):
        ScoringConfig.from_mapping(raw)


@pytest.mark.parametrize(
    "duplicate",
    [
        "version: us-price-baseline-v2\nversion: us-price-baseline-v3\n",
        "version: us-price-baseline-v3\nversion: us-price-baseline-v2\n",
        'version: us-price-baseline-v2\nversion: "us-price-baseline-v\\x33"\n',
        'version: "us-price-baseline-v\\u0033"\nversion: us-price-baseline-v2\n',
    ],
)
def test_v3_config_rejects_duplicate_version_regardless_of_order(
    tmp_path: Path,
    duplicate: str,
) -> None:
    path = tmp_path / "duplicate.yml"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_scoring_config(path)


@pytest.mark.parametrize("escaped_suffix", ("\\x33", "\\u0033"))
def test_escaped_v3_config_rejects_nested_duplicate_keys(
    tmp_path: Path,
    escaped_suffix: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    ).read_text(encoding="utf-8")
    escaped = source.replace(
        "version: us-price-baseline-v3",
        f'version: "us-price-baseline-v{escaped_suffix}"',
        1,
    )
    duplicate = escaped.replace(
        "      low: 30.0\n      high: 70.0",
        "      low: 30.0\n      low: 31.0\n      high: 70.0",
        1,
    )
    path = tmp_path / "duplicate-nested.yml"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_scoring_config(path)


@pytest.mark.parametrize("escaped_suffix", ("\\x33", "\\u0033"))
@pytest.mark.parametrize(
    "duplicate",
    (
        "      low: 30.0\n      low: 31.0\n      high: 70.0",
        "      low: 31.0\n      low: 30.0\n      high: 70.0",
    ),
    ids=("canonical-first", "duplicate-first"),
)
def test_inline_merged_v3_rejects_nested_duplicate_keys_in_both_orders(
    tmp_path: Path,
    escaped_suffix: str,
    duplicate: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    ).read_text(encoding="utf-8")
    merged = source.replace(
        "version: us-price-baseline-v3",
        f'<<: &v3 {{version: "us-price-baseline-v{escaped_suffix}"}}',
        1,
    )
    malformed = merged.replace(
        "      low: 30.0\n      high: 70.0",
        duplicate,
        1,
    )
    path = tmp_path / "inline-merged-duplicate.yml"
    path.write_text(malformed, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


@pytest.mark.parametrize(
    "merge",
    (
        '<<: [{version: "us-price-baseline-v\\x33"}, {}]',
        '<<: [{}, {version: "us-price-baseline-v\\u0033"}]',
    ),
)
def test_merge_sequence_v3_rejects_nested_duplicate_keys(
    tmp_path: Path,
    merge: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    ).read_text(encoding="utf-8")
    malformed = source.replace("version: us-price-baseline-v3", merge, 1).replace(
        "      low: 30.0\n      high: 70.0",
        "      low: 30.0\n      low: 31.0\n      high: 70.0",
        1,
    )
    path = tmp_path / "sequence-merged-duplicate.yml"
    path.write_text(malformed, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


@pytest.mark.parametrize(
    "merge",
    (
        '<<: [{version: "us-price-baseline-v\\x33"}, {version: us-price-baseline-v2}]',
        '<<: [{version: us-price-baseline-v2}, {version: "us-price-baseline-v\\u0033"}]',
    ),
)
def test_merge_sequence_v3_rejects_conflicting_versions(
    tmp_path: Path,
    merge: str,
) -> None:
    path = tmp_path / "sequence-merged-conflict.yml"
    path.write_text(f"{merge}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


@pytest.mark.parametrize(
    "document",
    (
        '<<: {version: "us-price-baseline-v\\x33"}\nversion: us-price-baseline-v2\n',
        'version: us-price-baseline-v2\n<<: {version: "us-price-baseline-v\\u0033"}\n',
        '<<: {version: "us-price-baseline-v\\u0033"}\nversion: us-price-baseline-v3\n',
        'version: us-price-baseline-v3\n<<: {version: "us-price-baseline-v\\x33"}\n',
    ),
)
def test_merged_v3_rejects_explicit_version_duplicates_and_conflicts(
    tmp_path: Path,
    document: str,
) -> None:
    path = tmp_path / "merged-explicit-version.yml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


def test_aliased_root_merge_v3_selects_strict_loading(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    ).read_text(encoding="utf-8")
    malformed = source.replace(
        "version: us-price-baseline-v3",
        'merge_source: &v3 {version: "us-price-baseline-v\\x33"}\n<<: *v3',
        1,
    ).replace(
        "      low: 30.0\n      high: 70.0",
        "      low: 30.0\n      low: 31.0\n      high: 70.0",
        1,
    )
    path = tmp_path / "aliased-merge.yml"
    path.write_text(malformed, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


@pytest.mark.parametrize(
    "document",
    (
        '&root\nversion: "us-price-baseline-v\\x33"\n<<: *root\n',
        ('version: "us-price-baseline-v\\u0033"\nnested: &nested\n  <<: *nested\n'),
    ),
)
def test_v3_yaml_merge_cycles_fail_closed(tmp_path: Path, document: str) -> None:
    path = tmp_path / "merge-cycle.yml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


def test_merge_cycle_before_reachable_v3_claim_routes_strict_and_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cycle-before-v3.yml"
    path.write_text(
        '&root\n<<: [*root, {version: "us-price-baseline-v\\x33"}]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


def test_root_v3_rejects_tagged_sequence_merge_key_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tagged-sequence-merge-root-v3.yml"
    path.write_text(
        "version: us-price-baseline-v3\n"
        "? !<tag:yaml.org,2002:merge> [not-a-scalar-merge-key]\n"
        ": {legacy: true}\n",
        encoding="utf-8",
    )
    constructor_calls = 0

    def fail_if_constructed(self, stream) -> None:
        nonlocal constructor_calls
        constructor_calls += 1
        pytest.fail("_UniqueKeyLoader must not be constructed for merged v3 YAML")

    monkeypatch.setattr(config_module._UniqueKeyLoader, "__init__", fail_if_constructed)

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)

    assert constructor_calls == 0


def test_tagged_sequence_merge_carries_v3_claim_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "tagged-sequence-merge-carried-v3.yml"
    path.write_text(
        "? !<tag:yaml.org,2002:merge> [not-a-scalar-merge-key]\n"
        ': {version: "us-price-baseline-v\\x33"}\n',
        encoding="utf-8",
    )
    constructor_calls = 0

    def fail_if_constructed(self, stream) -> None:
        nonlocal constructor_calls
        constructor_calls += 1
        pytest.fail("_UniqueKeyLoader must not be constructed for merged v3 YAML")

    monkeypatch.setattr(config_module._UniqueKeyLoader, "__init__", fail_if_constructed)

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)

    assert constructor_calls == 0


def test_tagged_sequence_merge_depth_26_alias_graph_rejects_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = ['level_0: &level_0 {version: "us-price-baseline-v\\x33"}']
    for level in range(1, 27):
        lines.extend(
            (
                f"level_{level}: &level_{level}",
                "  ? !<tag:yaml.org,2002:merge> [not-a-scalar-merge-key]",
                f"  : [*level_{level - 1}, *level_{level - 1}]",
            )
        )
    lines.extend(
        (
            "? !<tag:yaml.org,2002:merge> [root-non-scalar-merge-key]",
            ": *level_26",
        )
    )
    path = tmp_path / "us-price-baseline-v3.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    constructor_calls = 0

    def fail_if_constructed(self, stream) -> None:
        nonlocal constructor_calls
        constructor_calls += 1
        pytest.fail("_UniqueKeyLoader must not be constructed for merged v3 YAML")

    monkeypatch.setattr(config_module._UniqueKeyLoader, "__init__", fail_if_constructed)

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)

    assert constructor_calls == 0


def test_depth_26_doubled_alias_v3_merge_rejects_before_unique_loader_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = ['level_0: &level_0 {version: "us-price-baseline-v\\x33"}']
    for level in range(1, 27):
        lines.extend(
            (
                f"level_{level}: &level_{level}",
                f"  <<: [*level_{level - 1}, *level_{level - 1}]",
            )
        )
    lines.append("<<: *level_26")
    path = tmp_path / "doubled-alias-v3.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    constructor_calls = 0

    def fail_if_constructed(self, stream) -> None:
        nonlocal constructor_calls
        constructor_calls += 1
        pytest.fail("_UniqueKeyLoader must not be constructed for merged v3 YAML")

    monkeypatch.setattr(config_module._UniqueKeyLoader, "__init__", fail_if_constructed)

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)

    assert constructor_calls == 0


@pytest.mark.parametrize(
    ("boundary_document", "over_limit_document"),
    (
        (
            "- item\n" * 1023,
            "- item\n" * 1024,
        ),
        (
            "\n".join(("- &shared item", *(["- *shared"] * 4095))) + "\n",
            "\n".join(("- &shared item", *(["- *shared"] * 4096))) + "\n",
        ),
        (
            "- " * 63 + "item\n",
            "- " * 64 + "item\n",
        ),
    ),
    ids=("unique-nodes", "child-reference-edges", "graph-path-depth"),
)
def test_v3_yaml_inspection_budgets_accept_exact_boundary_and_reject_boundary_plus_one(
    tmp_path: Path,
    boundary_document: str,
    over_limit_document: str,
) -> None:
    boundary_path = tmp_path / "boundary.yml"
    boundary_path.write_text(boundary_document, encoding="utf-8")
    with pytest.raises(ValueError, match="Scoring config must be a mapping"):
        load_scoring_config(boundary_path)

    over_limit_path = tmp_path / "over-limit.yml"
    over_limit_path.write_text(over_limit_document, encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="^Scoring config structure exceeds the v3 inspection budget$",
    ):
        load_scoring_config(over_limit_path)


@pytest.mark.parametrize(
    ("wrapper_count", "expected_message", "expected_constructor_calls"),
    (
        pytest.param(61, "^config is missing keys:", 1, id="actual-depth-64"),
        pytest.param(
            62,
            "^Scoring config structure exceeds the v3 inspection budget$",
            0,
            id="actual-depth-65",
        ),
    ),
)
def test_v3_yaml_inspection_depth_follows_deeper_alias_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrapper_count: int,
    expected_message: str,
    expected_constructor_calls: int,
) -> None:
    path = tmp_path / "us-price-baseline-v3.yml"
    path.write_text(
        f"shallow: &shared [leaf]\ndeep: {'[' * wrapper_count}*shared{']' * wrapper_count}\n",
        encoding="utf-8",
    )
    constructor_calls = 0
    original_init = config_module._UniqueKeyLoader.__init__

    def track_construction(self, stream) -> None:
        nonlocal constructor_calls
        constructor_calls += 1
        original_init(self, stream)

    monkeypatch.setattr(
        config_module._UniqueKeyLoader,
        "__init__",
        track_construction,
    )

    with pytest.raises(ValueError, match=expected_message):
        load_scoring_config(path)

    assert constructor_calls == expected_constructor_calls


@pytest.mark.parametrize(
    ("filename", "document"),
    (
        (
            "root-v3-nested-merge.yml",
            'version: "us-price-baseline-v\\x33"\nnested:\n  <<: {legacy: true}\n',
        ),
        (
            "merge-carried-v3.yml",
            'source: &source {version: "us-price-baseline-v\\u0033"}\n<<: *source\n',
        ),
        (
            "us-price-baseline-v3.yml",
            "version: us-price-baseline-v2\nnested:\n  <<: {legacy: true}\n",
        ),
    ),
    ids=("root-claim-with-nested-merge", "merge-carried-escaped-claim", "canonical-path"),
)
def test_v3_rejects_merge_tags_anywhere_before_construction(
    tmp_path: Path,
    filename: str,
    document: str,
) -> None:
    path = tmp_path / filename
    path.write_text(document, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^YAML merge keys are not supported for us-price-baseline-v3$",
    ):
        load_scoring_config(path)


def test_quoted_merge_spelling_is_an_ordinary_strict_unknown_key(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    ).read_text(encoding="utf-8")
    path = tmp_path / "quoted-merge-spelling.yml"
    path.write_text(source + '"<<": {legacy: true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"^config has unknown keys: <<$"):
        load_scoring_config(path)


def test_nested_v3_string_does_not_select_strict_loading(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    ).read_text(encoding="utf-8")
    source += (
        "\nparser_notes:\n  version: us-price-baseline-v3\n  repeated: first\n  repeated: second\n"
    )
    path = tmp_path / "legacy-with-nested-v3-string.yml"
    path.write_text(source, encoding="utf-8")

    config = load_scoring_config(path)

    assert config.version == "us-price-baseline-v2"


def test_explicit_string_tags_preserve_v3_loading_and_hash(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    ).read_text(encoding="utf-8")
    tagged = source.replace(
        "version: us-price-baseline-v3",
        "!!str version: !!str us-price-baseline-v3",
        1,
    )
    path = tmp_path / "tagged-v3.yml"
    path.write_text(tagged, encoding="utf-8")

    config = load_scoring_config(path)

    assert config_hash(config) == (
        "aee8ae47cc82092f2167d778381e55bccc7c393913a997afd1ca391ca182585c"
    )


def test_v3_config_rejects_multiple_yaml_documents(tmp_path: Path) -> None:
    path = tmp_path / "multiple-documents.yml"
    path.write_text(
        "version: us-price-baseline-v3\n---\nversion: us-price-baseline-v2\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.YAMLError):
        load_scoring_config(path)


@pytest.mark.parametrize("document", ("[]\n", "us-price-baseline-v3\n"))
def test_scoring_config_rejects_non_mapping_root(
    tmp_path: Path,
    document: str,
) -> None:
    path = tmp_path / "non-mapping.yml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_scoring_config(path)


def test_v2_score_and_recommendation_are_split_invariant() -> None:
    path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    config = load_scoring_config(path)
    original = _rich_indicators()
    transformed_values = {
        **original.values,
        "last_close": original.values.get("last_close", 100.0) / 10,
        "macd_histogram": original.values["macd_histogram"] / 10,
        "avg_volume_20d": original.values["avg_volume_20d"] * 10,
    }
    transformed = IndicatorResult(
        values=transformed_values,
        observation_count=original.observation_count,
        last_date=original.last_date,
    )

    original_components = score_components(original, ResearchValues(values={}), config)
    transformed_components = score_components(transformed, ResearchValues(values={}), config)
    original_score = aggregate_score(original_components, config)
    transformed_score = aggregate_score(transformed_components, config)
    risk = RiskAssessment(score=20, risk_class="low")
    original_decision = decide_recommendation(
        original_score.overall,
        risk,
        original_score.confidence,
        config,
        scenarios={"short": _evidenced_scenarios()["short"]},
        indicators=original,
    )
    transformed_decision = decide_recommendation(
        transformed_score.overall,
        risk,
        transformed_score.confidence,
        config,
        scenarios={"short": _evidenced_scenarios()["short"]},
        indicators=transformed,
    )

    assert transformed_components.factor_scores == pytest.approx(original_components.factor_scores)
    assert transformed_score.overall == pytest.approx(original_score.overall)
    assert transformed_decision.recommendation == original_decision.recommendation
    assert transformed_decision.gates == original_decision.gates


def test_config_requires_supported_horizon_gates_and_active_factor_counts() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v1.yml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["recommendation"]["buy_max_bear_downside"]["medium"] = -0.30
    with pytest.raises(ValueError, match="buy_max_bear_downside"):
        ScoringConfig.from_mapping(raw)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del raw["component_factor_counts"]["market_sector"]
    with pytest.raises(ValueError, match="component_factor_counts"):
        ScoringConfig.from_mapping(raw)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["factor_policy"] = {
        "macd_indicator": "absolute_price",
        "liquidity_indicator": "avg_volume_20d",
    }
    with pytest.raises(ValueError, match="macd_indicator"):
        ScoringConfig.from_mapping(raw)


def test_v2_config_rejects_nonboolean_strict_finite_policy() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["factor_policy"]["strict_finite_inputs"] = "false"

    with pytest.raises(ValueError, match="strict_finite_inputs"):
        ScoringConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("factor_policy", "macd_score_low", float("nan")),
        ("factor_policy", "macd_score_high", float("inf")),
        ("factor_policy", "liquidity_score_low", float("nan")),
        ("factor_policy", "liquidity_score_high", float("inf")),
        ("recommendation", "buy_min_liquidity_20d", float("nan")),
        ("recommendation", "buy_min_liquidity_20d", float("inf")),
    ],
)
def test_v2_config_rejects_nonfinite_policy_values(
    section: str,
    key: str,
    value: float,
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw[section][key] = value

    with pytest.raises(ValueError, match="finite"):
        ScoringConfig.from_mapping(raw)


def test_v1_preserves_legacy_nonfinite_factor_behavior() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v1.yml"
    )
    components = score_components(
        IndicatorResult(
            values={
                "macd_histogram": float("nan"),
                "abnormal_volume": 0.0,
                "avg_volume_20d": float("nan"),
            }
        ),
        ResearchValues(values={}),
        config,
    )

    assert components.factor_scores["momentum.macd"] == 100
    assert components.factor_scores["risk.abnormal_volume"] == 0
    assert components.factor_scores["risk.avg_volume"] == 100


def test_v2_withholds_nonfinite_price_scale_factors() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    )
    components = score_components(
        IndicatorResult(
            values={
                "macd_histogram_pct": float("nan"),
                "abnormal_volume_strict": float("nan"),
                "avg_dollar_volume_20d": float("nan"),
            }
        ),
        ResearchValues(values={}),
        config,
    )

    assert "momentum.macd" not in components.factor_scores
    assert "risk.abnormal_volume" not in components.factor_scores
    assert "risk.avg_volume" not in components.factor_scores
    assert components.missing["momentum.macd"] == "Input must be finite"


def test_v3_uses_exactly_sixteen_yaml_mapped_factors_and_excludes_beta() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    components = score_components(
        _rich_indicators(),
        ResearchValues(values={}),
        config,
    )

    assert config.short_scoring is not None
    assert len(components.factor_scores) == 16
    assert set(components.factor_scores) == set(config.short_scoring.factor_maps)
    assert "risk.beta" not in components.factor_scores
    assert components.coverage == 1.0
    assert set(components.components) == {
        "momentum_technical",
        "risk_liquidity",
        "market_sector",
    }


@pytest.mark.parametrize(
    ("rsi", "expected"),
    [(30.0, 0.0), (50.0, 50.0), (70.0, 100.0), (-10.0, 0.0), (110.0, 100.0)],
)
def test_v3_rsi_uses_continuous_affine_30_70_map(rsi: float, expected: float) -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    indicators = IndicatorResult(values={"rsi_14": rsi})

    components = score_components(indicators, ResearchValues(values={}), config)

    assert components.factor_scores["momentum.rsi"] == pytest.approx(expected)


def test_v3_rsi_map_is_bounded_monotone_and_2_5_lipschitz() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    inputs = [float(value) for value in range(-20, 121)]
    scores = [
        score_components(
            IndicatorResult(values={"rsi_14": value}),
            ResearchValues(values={}),
            config,
        ).factor_scores["momentum.rsi"]
        for value in inputs
    ]

    assert all(0.0 <= score <= 100.0 for score in scores)
    assert scores == sorted(scores)
    for left_x, right_x, left_score, right_score in zip(
        inputs[:-1],
        inputs[1:],
        scores[:-1],
        scores[1:],
        strict=True,
    ):
        assert abs(right_score - left_score) <= 2.5 * abs(right_x - left_x) + 1e-12


def test_v3_risk_requires_all_four_penalties_and_preserves_valid_zero() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    zero = IndicatorResult(
        values={
            "annualized_volatility": 0.0,
            "downside_volatility": 0.0,
            "max_drawdown": 0.0,
            "beta": 0.0,
        }
    )
    incomplete = IndicatorResult(values={**zero.values})
    del incomplete.values["beta"]

    zero_risk = assess_risk(zero, ResearchValues(values={}), config)
    incomplete_risk = assess_risk(incomplete, ResearchValues(values={}), config)

    assert zero_risk.score == 0.0
    assert zero_risk.risk_class == "low"
    assert incomplete_risk.score is None
    assert incomplete_risk.risk_class == "insufficient"
    assert "absolute_beta" in incomplete_risk.insufficiency_reason


def test_v3_risk_is_exact_average_of_four_yaml_penalties() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )

    risk = assess_risk(
        IndicatorResult(
            values={
                "annualized_volatility": (0.12 + 0.65) / 2,
                "downside_volatility": (0.08 + 0.50) / 2,
                "max_drawdown": -(0.05 + 0.60) / 2,
                "beta": 1.0,
            }
        ),
        ResearchValues(values={}),
        config,
    )

    assert risk.score == pytest.approx(50.0)
    assert risk.risk_class == "medium"


def test_v3_beta_penalty_is_symmetric_absolute_exposure() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    base = {
        "annualized_volatility": 0.12,
        "downside_volatility": 0.08,
        "max_drawdown": -0.05,
    }

    positive = assess_risk(
        IndicatorResult(values={**base, "beta": 1.0}),
        ResearchValues(values={}),
        config,
    )
    negative = assess_risk(
        IndicatorResult(values={**base, "beta": -1.0}),
        ResearchValues(values={}),
        config,
    )
    saturated = assess_risk(
        IndicatorResult(values={**base, "beta": 2.5}),
        ResearchValues(values={}),
        config,
    )

    assert positive.score == negative.score == pytest.approx(12.5)
    assert saturated.score == pytest.approx(25.0)


@pytest.mark.parametrize("value", [0.01, 0.25, 0.99, 1.0, 1.75, 2.0, 4.5])
def test_v3_abnormal_volume_matches_v2_on_positive_domain(value: float) -> None:
    root = Path(__file__).resolve().parents[1]
    v2 = load_scoring_config(root / "config/scoring/us-price-baseline-v2.yml")
    v3 = load_scoring_config(root / "config/scoring/us-price-baseline-v3.yml")
    indicators = IndicatorResult(
        values={
            "abnormal_volume": value,
            "abnormal_volume_strict": value,
        }
    )

    v2_score = score_components(indicators, ResearchValues(values={}), v2)
    v3_score = score_components(indicators, ResearchValues(values={}), v3)

    assert v3_score.factor_scores["risk.abnormal_volume"] == pytest.approx(
        v2_score.factor_scores["risk.abnormal_volume"]
    )


def test_v3_abnormal_volume_preserves_intentional_zero_discontinuity() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )

    at_zero = score_components(
        IndicatorResult(values={"abnormal_volume_strict": 0.0}),
        ResearchValues(values={}),
        config,
    )
    near_zero = score_components(
        IndicatorResult(values={"abnormal_volume_strict": 1e-12}),
        ResearchValues(values={}),
        config,
    )

    assert at_zero.factor_scores["risk.abnormal_volume"] == 0.0
    assert near_zero.factor_scores["risk.abnormal_volume"] == pytest.approx(80.0)


def test_v3_missing_risk_blocks_buy_but_avoid_gates_remain_independent() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    insufficient = assess_risk(
        IndicatorResult(values={}),
        ResearchValues(values={}),
        config,
    )
    scenarios = {"short": _evidenced_scenarios()["short"]}
    indicators = IndicatorResult(values={"avg_dollar_volume_20d": 10_000_000.0})

    avoid = decide_recommendation(
        30.0,
        insufficient,
        60.0,
        config,
        scenarios=scenarios,
        indicators=indicators,
    )
    hold = decide_recommendation(
        60.0,
        insufficient,
        60.0,
        config,
        scenarios=scenarios,
        indicators=indicators,
    )

    assert avoid.recommendation == "avoid"
    assert avoid.gates["avoid_score"] is True
    assert hold.recommendation == "hold"
    assert hold.gates["buy_risk_present"] is False


def test_v3_raw_macd_and_atr_are_not_direct_factor_inputs() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    baseline = _rich_indicators()
    changed = IndicatorResult(
        values={
            **baseline.values,
            "macd": 1e9,
            "macd_signal": -1e9,
            "macd_histogram": 2e9,
            "atr_14": 5e8,
            "atr_14_pct": 99.0,
            "last_close": 0.01,
        },
        observation_count=baseline.observation_count,
        last_date=baseline.last_date,
    )

    original_scores = score_components(baseline, ResearchValues(values={}), config)
    changed_scores = score_components(changed, ResearchValues(values={}), config)

    assert changed_scores == original_scores


def test_v3_scoring_has_no_nominal_price_band_lookup() -> None:
    import stanstock.research.scoring as scoring_module

    source = inspect.getsource(scoring_module)

    assert "classify_price_band" not in source
    assert "UNDER_10_BAND" not in source
    assert "LatestMarketData" not in source


def test_v3_fixed_volume_price_counterexample_can_change_hold_to_buy() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
    )
    favorable = {
        "return_20d": 0.15,
        "return_63d": 0.30,
        "return_126d": 0.45,
        "close_vs_sma_50": 0.10,
        "close_vs_sma_200": 0.20,
        "rsi_14": 70.0,
        "macd_histogram_pct": 0.02,
        "52w_position": 0.95,
        "annualized_volatility": 0.12,
        "downside_volatility": 0.08,
        "max_drawdown": -0.05,
        "beta": 0.0,
        "abnormal_volume_strict": 1.0,
        "relative_return_20d": 0.08,
        "relative_return_63d": 0.15,
        "relative_return_252d": 0.25,
    }
    baseline_indicators = IndicatorResult(
        values={**favorable, "avg_dollar_volume_20d": 2_500_000.0}
    )
    scaled_indicators = IndicatorResult(values={**favorable, "avg_dollar_volume_20d": 10_000_000.0})
    baseline_components = score_components(
        baseline_indicators,
        ResearchValues(values={}),
        config,
    )
    scaled_components = score_components(
        scaled_indicators,
        ResearchValues(values={}),
        config,
    )
    baseline_score = aggregate_score(baseline_components, config)
    scaled_score = aggregate_score(scaled_components, config)
    risk = assess_risk(baseline_indicators, ResearchValues(values={}), config)
    scenarios = {"short": _evidenced_scenarios()["short"]}

    baseline = decide_recommendation(
        baseline_score.overall,
        risk,
        baseline_score.confidence,
        config,
        scenarios=scenarios,
        indicators=baseline_indicators,
    )
    scaled = decide_recommendation(
        scaled_score.overall,
        risk,
        scaled_score.confidence,
        config,
        scenarios=scenarios,
        indicators=scaled_indicators,
    )

    assert baseline_score.overall >= config.recommendation.buy_min_score
    assert scaled_score.overall > baseline_score.overall
    assert baseline.gates["buy_liquidity"] is False
    assert scaled.gates["buy_liquidity"] is True
    assert all(value for key, value in scaled.gates.items() if key.startswith("buy_"))
    assert baseline.recommendation == "hold"
    assert scaled.recommendation == "buy"


def test_score_bounds_confidence_cap_and_freshness_penalty() -> None:
    config = load_scoring_config()
    components = score_components(_rich_indicators(), _rich_fundamentals(), config)
    fresh = aggregate_score(
        components,
        config,
        decision_date=date(2026, 9, 5),
        indicators=_rich_indicators(),
    )
    stale_indicators = IndicatorResult(
        values=_rich_indicators().values,
        observation_count=260,
        last_date=date(2026, 7, 1),
    )
    stale = aggregate_score(
        components,
        config,
        decision_date=date(2026, 9, 5),
        indicators=stale_indicators,
    )

    assert all(0 <= value <= 100 for value in components.components.values())
    assert all(0 <= value <= 100 for value in fresh.horizon_scores.values())
    assert 0 <= fresh.overall <= 100
    assert fresh.confidence <= config.coverage.confidence_cap
    assert stale.overall < fresh.overall
    assert stale.confidence <= config.freshness.stale_confidence_cap + 10


def test_risk_classes_and_recommendation_gates_are_deterministic() -> None:
    config = load_scoring_config()
    indicators = _rich_indicators()
    low_risk = assess_risk(_rich_indicators(), _rich_fundamentals(), config)
    buy = decide_recommendation(
        80, low_risk, 70, config, scenarios=_evidenced_scenarios(), indicators=indicators
    )
    avoid = decide_recommendation(
        30, low_risk, 70, config, scenarios=_evidenced_scenarios(), indicators=indicators
    )
    insufficient_risk = assess_risk(
        IndicatorResult(values={}),
        ResearchValues(values={}),
        config,
    )
    insufficient_decision = decide_recommendation(
        90,
        insufficient_risk,
        70,
        config,
        scenarios=_evidenced_scenarios(),
        indicators=indicators,
    )

    assert low_risk.risk_class in {"low", "medium"}
    assert buy.recommendation == "buy"
    assert all(value for key, value in buy.gates.items() if key.startswith("buy_"))
    assert avoid.recommendation == "avoid"
    assert insufficient_risk.score is None
    assert insufficient_risk.risk_class == "insufficient"
    assert insufficient_risk.insufficiency_reason
    assert insufficient_decision.recommendation == "hold"
    assert insufficient_decision.gates["buy_risk_present"] is False


def test_buy_requires_scenario_evidence_and_liquidity() -> None:
    config = load_scoring_config()
    low_risk = RiskAssessment(score=20, risk_class="low")
    missing_short = {
        **_evidenced_scenarios(),
        "short": Scenario(
            bear=None,
            base=None,
            bull=None,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="missing",
            method="test",
        ),
    }
    no_scenario = decide_recommendation(
        90, low_risk, 80, config, scenarios=missing_short, indicators=_rich_indicators()
    )
    no_liquidity = decide_recommendation(
        90,
        low_risk,
        80,
        config,
        scenarios=_evidenced_scenarios(),
        indicators=IndicatorResult(values={}),
    )
    thin_liquidity = decide_recommendation(
        90,
        low_risk,
        80,
        config,
        scenarios=_evidenced_scenarios(),
        indicators=IndicatorResult(values={"avg_volume_20d": 10}),
    )

    assert no_scenario.recommendation == "hold"
    assert no_scenario.gates["buy_short_scenario_present"] is False
    assert no_liquidity.recommendation == "hold"
    assert no_liquidity.gates["buy_liquidity_present"] is False
    assert no_liquidity.gates["buy_liquidity"] is False
    assert thin_liquidity.recommendation == "hold"
    assert thin_liquidity.gates["buy_liquidity_present"] is True
    assert thin_liquidity.gates["buy_liquidity"] is False


def test_v2_buy_gate_cannot_be_satisfied_by_share_volume() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v2.yml"
    )
    decision = decide_recommendation(
        90,
        RiskAssessment(score=20, risk_class="low"),
        80,
        config,
        scenarios={"short": _evidenced_scenarios()["short"]},
        indicators=IndicatorResult(values={"avg_volume_20d": 50_000_000}),
    )

    assert decision.recommendation == "hold"
    assert decision.gates["buy_liquidity_present"] is False
    assert decision.gates["buy_liquidity"] is False


def test_buy_requires_configured_bear_downside_by_horizon() -> None:
    config = load_scoring_config()
    adverse = {
        **_evidenced_scenarios(),
        "medium": Scenario(
            bear=-0.31,
            base=0.10,
            bull=0.20,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
    }

    decision = decide_recommendation(
        90,
        RiskAssessment(score=20, risk_class="low"),
        80,
        config,
        scenarios=adverse,
        indicators=_rich_indicators(),
    )

    assert decision.recommendation == "hold"
    assert decision.gates["buy_medium_scenario_present"] is True
    assert decision.gates["buy_medium_bear_downside"] is False


def test_missingness_penalty_lowers_sparse_scores_without_zero_filling() -> None:
    config = load_scoring_config()
    sparse = score_components(
        IndicatorResult(values={"return_20d": 0.20}, last_date=date.today() - timedelta(days=1)),
        ResearchValues(values={}),
        config,
    )
    aggregate = aggregate_score(sparse, config)

    assert sparse.coverage < config.coverage.minimum_component_coverage
    assert aggregate.missingness_penalty < 1
    assert aggregate.confidence <= 35
