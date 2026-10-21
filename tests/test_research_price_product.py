from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from unittest.mock import Mock
from uuid import UUID

import numpy as np
import pytest

import stanstock.research.price_product as price_product_module
from stanstock.research.price_product import (
    EvidenceGrade,
    FilteredReturns,
    PriceInputIdentity,
    PriceProductInput,
    PriceProductInputError,
    PriceSeries,
    SimulationTerminals,
    SourceExecutionBinding,
    SourceExecutionMode,
    apply_recommendation_policy,
    calculate_momentum,
    calculate_price_product,
    complete_input_hash,
    deterministic_seed,
    filter_historical_returns,
    prediction_model_version,
    project_fhs,
    simulate_fhs_terminal_logs,
)
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    load_price_product_config,
)


def _dates() -> tuple[date, ...]:
    first = date(2023, 8, 18)
    return tuple(first + timedelta(days=index) for index in range(757))


def _closes(
    *,
    base: float,
    drift: float,
    wave: float,
    phase: float = 0.0,
) -> tuple[float, ...]:
    returns = np.asarray(
        [
            drift
            + wave * math.sin(index * 0.113 + phase)
            + wave * 0.35 * math.cos(index * 0.071 + phase)
            for index in range(756)
        ],
        dtype=np.float64,
    )
    return tuple(
        float(value)
        for value in np.concatenate((np.asarray([base]), base * np.exp(np.cumsum(returns))))
    )


def _identity(
    asset_id: str,
    *,
    subject: str,
    provider: str = "twelve_data",
    admitted_at: datetime | None = None,
) -> PriceInputIdentity:
    timestamp = admitted_at or datetime(2025, 9, 13, 12, tzinfo=UTC)
    return PriceInputIdentity(
        asset_id=UUID(asset_id),
        provider=provider,
        subject=subject,
        sha256=("a" if subject == "ACME" else "b") * 64,
        retrieved_at=timestamp,
        available_at=timestamp,
    )


def _product_input(
    *,
    stock_closes: tuple[float, ...] | None = None,
    benchmark_closes: tuple[float, ...] | None = None,
    stock_scale: float = 1.0,
    volume_scale: float = 1.0,
    source_mode: SourceExecutionMode = "provider",
    evidence_grade: EvidenceGrade = "research",
    stock_provider: str = "twelve_data",
    benchmark_provider: str = "twelve_data",
) -> PriceProductInput:
    dates = _dates()
    stock_values = stock_closes or _closes(base=100.0, drift=0.0008, wave=0.006)
    benchmark_values = benchmark_closes or _closes(base=400.0, drift=0.00025, wave=0.004, phase=0.3)
    return PriceProductInput(
        listing_id=UUID("aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb"),
        target_date=dates[-1],
        decision_time=datetime(2025, 9, 13, 14, tzinfo=UTC),
        calendar_sessions=dates,
        stock=PriceSeries(
            identity=_identity(
                "11111111-1111-4111-8111-111111111111",
                subject="ACME",
                provider=stock_provider,
            ),
            currency="USD",
            dates=dates,
            closes=tuple(value * stock_scale for value in stock_values),
            volumes=tuple(2_000_000.0 * volume_scale for _date in dates),
            volume_adjustment_compatible=True,
        ),
        benchmark=PriceSeries(
            identity=_identity(
                "22222222-2222-4222-8222-222222222222",
                subject="SPY",
                provider=benchmark_provider,
            ),
            currency="USD",
            dates=dates,
            closes=benchmark_values,
        ),
        source_execution=SourceExecutionBinding(
            mode=source_mode,
            evidence_grade=evidence_grade,
        ),
    )


def test_momentum_uses_symmetric_t_minus_252_to_t_minus_21_endpoints() -> None:
    stock = [100.0] * 757
    benchmark = [100.0] * 757
    stock[504] = 80.0
    stock[735] = 120.0
    benchmark[504] = 90.0
    benchmark[735] = 110.0
    # Values at asymmetric/off-by-one candidates must not affect the result.
    stock[503], stock[736], benchmark[503], benchmark[736] = 1.0, 999.0, 2.0, 888.0

    result = calculate_momentum(tuple(stock), tuple(benchmark))

    assert result.stock_log_momentum == pytest.approx(math.log(120.0 / 80.0))
    assert result.benchmark_log_momentum == pytest.approx(math.log(110.0 / 90.0))
    assert result.relative_log_momentum == pytest.approx(
        math.log(120.0 / 80.0) - math.log(110.0 / 90.0)
    )
    assert result.stock_price_return == pytest.approx(0.5)
    assert result.direction == "positive"


@pytest.mark.parametrize(
    ("stock_end", "benchmark_end", "expected"),
    [
        (90.0, 95.0, "negative"),
        (110.0, 120.0, "mixed"),
        (100.0, 100.0, "mixed"),
    ],
)
def test_momentum_sign_policy(
    stock_end: float,
    benchmark_end: float,
    expected: str,
) -> None:
    stock = [100.0] * 757
    benchmark = [100.0] * 757
    stock[735] = stock_end
    benchmark[735] = benchmark_end

    assert calculate_momentum(tuple(stock), tuple(benchmark)).direction == expected


def test_filter_matches_population_variance_indexing_and_residual_normalization() -> None:
    closes = _closes(base=50.0, drift=0.0003, wave=0.007)
    result = filter_historical_returns(closes)
    returns = np.diff(np.log(np.asarray(closes)))
    mean = float(np.mean(returns))
    variance = float(np.mean((returns - mean) ** 2))
    q = variance
    standardized: list[float] = []
    for observed_return in returns:
        innovation = float(observed_return - mean)
        standardized.append(innovation / math.sqrt(q))
        q = 0.01 * variance + 0.94 * q + 0.05 * innovation**2

    assert result.mean_log_return == pytest.approx(mean)
    assert result.population_variance == pytest.approx(variance)
    assert result.terminal_variance == pytest.approx(q)
    assert result.standardized_returns == pytest.approx(standardized)
    assert len(result.residuals) == 504
    residuals = np.asarray(result.residuals)
    assert float(np.mean(residuals)) == pytest.approx(0.0, abs=1e-14)
    assert float(np.mean(residuals**2)) == pytest.approx(1.0, abs=1e-14)


def test_future_recursion_uses_innovation_and_path_major_random_indices() -> None:
    filtered = filter_historical_returns(_closes(base=75.0, drift=0.0012, wave=0.005))
    seed = 918273
    result = simulate_fhs_terminal_logs(
        filtered,
        seed=seed,
        horizons=(1, 2),
        path_count=3,
    )
    residuals = np.asarray(filtered.residuals)
    indices = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(residuals), size=(3, 2), dtype=np.int64
    )
    q1 = filtered.terminal_variance
    innovation1 = math.sqrt(q1) * residuals[indices[0, 0]]
    q2 = 0.01 * filtered.population_variance + 0.94 * q1 + 0.05 * innovation1**2
    innovation2 = math.sqrt(q2) * residuals[indices[0, 1]]
    expected_one = filtered.mean_log_return + innovation1
    expected_two = expected_one + filtered.mean_log_return + innovation2

    assert result.with_drift[0][0] == pytest.approx(expected_one)
    assert result.with_drift[1][0] == pytest.approx(expected_two)
    assert result.zero_drift[0][0] == pytest.approx(expected_one - filtered.mean_log_return)
    assert result.zero_drift[1][0] == pytest.approx(expected_two - 2 * filtered.mean_log_return)


def test_diagnostic_path_doubling_preserves_the_complete_production_prefix() -> None:
    filtered = filter_historical_returns(_closes(base=80.0, drift=0.0005, wave=0.006))
    production = simulate_fhs_terminal_logs(
        filtered,
        seed=12345,
        horizons=(126, 252, 756, 1260),
        path_count=8192,
    )
    doubled = simulate_fhs_terminal_logs(
        filtered,
        seed=12345,
        horizons=(126, 252, 756, 1260),
        path_count=16384,
    )

    for production_values, diagnostic_values in zip(
        production.with_drift, doubled.with_drift, strict=True
    ):
        np.testing.assert_array_equal(production_values, diagnostic_values[:8192])
    for production_values, diagnostic_values in zip(
        production.zero_drift, doubled.zero_drift, strict=True
    ):
        np.testing.assert_array_equal(production_values, diagnostic_values[:8192])


def test_projection_uses_cumulative_expm1_linear_quantiles_and_same_shock_sensitivity() -> None:
    config = load_price_product_config()
    filtered = filter_historical_returns(_closes(base=100.0, drift=0.0007, wave=0.006))
    seed = 7755
    forecast = project_fhs(filtered, target_close=125.0, seed=seed, config=config)
    terminals = simulate_fhs_terminal_logs(
        filtered,
        seed=seed,
        horizons=(126, 252, 756, 1260),
        path_count=8192,
    )

    assert forecast.path_count == 8192
    for index, projection in enumerate(forecast.projections):
        expected = np.quantile(
            np.expm1(terminals.with_drift[index]),
            (0.2, 0.5, 0.8),
            method="linear",
        )
        expected_zero = np.quantile(
            np.expm1(terminals.with_drift[index] - projection.sessions * filtered.mean_log_return),
            (0.2, 0.5, 0.8),
            method="linear",
        )
        assert projection.central_model_mass == 0.6
        assert projection.raw_returns is not None
        assert projection.zero_drift_raw_returns is not None
        assert (
            projection.raw_returns.lower,
            projection.raw_returns.median,
            projection.raw_returns.upper,
        ) == pytest.approx(expected)
        assert (
            projection.zero_drift_raw_returns.lower,
            projection.zero_drift_raw_returns.median,
            projection.zero_drift_raw_returns.upper,
        ) == pytest.approx(expected_zero)
        assert projection.raw_prices is not None
        assert projection.raw_prices.median == pytest.approx(
            125.0 * (1 + projection.raw_returns.median)
        )
        assert projection.ledger_returns is not None
        assert projection.ledger_prices is not None
        assert projection.ledger_returns.median == Decimal(
            str(projection.raw_returns.median)
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
        assert projection.ledger_prices.median == Decimal(
            str(projection.raw_prices.median)
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)


def test_seed_and_complete_input_hash_have_separate_pinned_encodings() -> None:
    product_input = _product_input(
        stock_closes=tuple(float(64 + index) for index in range(757)),
        benchmark_closes=tuple(float(512 + 2 * index) for index in range(757)),
    )
    seed = deterministic_seed(
        method_version=FHS_METHOD_VERSION,
        effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        listing_id=product_input.listing_id,
        target_date=product_input.target_date,
    )

    assert seed == 138191051323385255095579043285383288333
    assert complete_input_hash(product_input) == (
        "7d73894d6e47a3fc8365d3025af8157aaa6183ad71e1eab91cfe8ad7fbe2685a"
    )
    changed_asset = replace(
        product_input,
        stock=replace(
            product_input.stock,
            identity=replace(
                product_input.stock.identity,
                asset_id=UUID("33333333-3333-4333-8333-333333333333"),
            ),
        ),
    )
    assert complete_input_hash(changed_asset) != complete_input_hash(product_input)
    changed_mode = replace(
        product_input,
        source_execution=replace(product_input.source_execution, mode="synthetic_demo"),
    )
    changed_grade = replace(
        product_input,
        source_execution=replace(product_input.source_execution, evidence_grade="observed"),
    )
    perturbed_closes = list(product_input.stock.closes)
    perturbed_closes[123] = math.nextafter(perturbed_closes[123], math.inf)
    changed_one_ulp = replace(
        product_input,
        stock=replace(product_input.stock, closes=tuple(perturbed_closes)),
    )
    assert complete_input_hash(changed_mode) != complete_input_hash(product_input)
    assert complete_input_hash(changed_grade) != complete_input_hash(product_input)
    assert complete_input_hash(changed_one_ulp) != complete_input_hash(product_input)
    assert (
        deterministic_seed(
            method_version=FHS_METHOD_VERSION,
            effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
            listing_id=changed_asset.listing_id,
            target_date=changed_asset.target_date,
        )
        == seed
    )


@pytest.fixture
def f800_input_encoder():
    import json

    # The f8008c5 document encoding, independent of the reusable encoder.
    def base_bytes(product_input):
        def series_payload(series):
            return {
                "identity": {
                    key: value.isoformat() if isinstance(value, datetime) else str(value)
                    for key, value in asdict(series.identity).items()
                },
                "currency": series.currency,
                "dates": [value.isoformat() for value in series.dates],
                "closes": [float(value).hex() for value in series.closes],
                "volumes": (
                    None
                    if series.volumes is None
                    else [None if value is None else float(value).hex() for value in series.volumes]
                ),
                "volume_adjustment_compatible": series.volume_adjustment_compatible,
            }

        document = {
            "domain": "stanstock-research-product-input-v1",
            "listing_id": str(product_input.listing_id),
            "target_date": product_input.target_date.isoformat(),
            "decision_time": product_input.decision_time.isoformat(),
            "source_execution": asdict(product_input.source_execution),
            "calendar_sessions": [value.isoformat() for value in product_input.calendar_sessions],
            "source_eligible": product_input.source_eligible,
            "source_ineligibility_reasons": list(product_input.source_ineligibility_reasons),
            "stock": series_payload(product_input.stock),
            "benchmark": series_payload(product_input.benchmark),
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    return base_bytes


def test_exact_run_hasher_preserves_complete_base_bytes_for_every_input_field(
    monkeypatch, f800_input_encoder
) -> None:
    product_input = _product_input()
    product_input = replace(
        product_input,
        benchmark=replace(
            product_input.benchmark,
            volumes=tuple(None if index % 7 == 0 else float(index) for index in range(757)),
        ),
    )
    variants = [product_input, product_input]
    for field, value in {
        "listing_id": UUID(int=5),
        "target_date": product_input.target_date - timedelta(days=1),
        "decision_time": product_input.decision_time + timedelta(seconds=1),
        "calendar_sessions": product_input.calendar_sessions[:-1],
        "source_execution": SourceExecutionBinding("synthetic_demo", "research"),
        "source_eligible": False,
        "source_ineligibility_reasons": ("missing",),
    }.items():
        variants.append(replace(product_input, **{field: value}))
    variants.append(
        replace(product_input, source_execution=SourceExecutionBinding("provider", "observed"))
    )
    for role in ("stock", "benchmark"):
        series = getattr(product_input, role)
        for field, value in {
            "currency": "EUR",
            "dates": series.dates[:-1],
            "closes": (math.nextafter(series.closes[0], math.inf), *series.closes[1:]),
            "volumes": None if series.volumes is not None else (None,) * 757,
            "volume_adjustment_compatible": not series.volume_adjustment_compatible,
        }.items():
            variants.append(replace(product_input, **{role: replace(series, **{field: value})}))
        for field, value in {
            "asset_id": UUID(int=6),
            "provider": "synthetic_demo",
            "subject": "OTHER",
            "sha256": "c" * 64,
            "retrieved_at": series.identity.retrieved_at + timedelta(seconds=1),
            "available_at": series.identity.available_at + timedelta(seconds=1),
        }.items():
            variants.append(
                replace(
                    product_input,
                    **{role: replace(series, identity=replace(series.identity, **{field: value}))},
                )
            )
    variants.append(product_input)  # Return to the original objects after changed identities.
    sha256 = price_product_module.hashlib.sha256
    digest = Mock(wraps=sha256)
    monkeypatch.setattr(price_product_module.hashlib, "sha256", digest)
    hasher = price_product_module._CompleteInputHasher()
    for variant in variants:
        expected = f800_input_encoder(variant)
        assert hasher(variant) == sha256(expected).hexdigest()
        assert digest.call_args.args[0] == expected  # Complete bytes, not component digests.


@pytest.mark.parametrize("field", ["calendar_sessions", "benchmark"])
@pytest.mark.parametrize("public", [False, True], ids=["fresh-hasher", "public-hash"])
def test_none_input_rejection_matches_f800_encoder(f800_input_encoder, field, public) -> None:
    product_input = replace(_product_input(), **{field: None})
    encoder = complete_input_hash if public else price_product_module._CompleteInputHasher()
    with pytest.raises((TypeError, AttributeError)) as expected:
        f800_input_encoder(product_input)
    with pytest.raises(type(expected.value)) as actual:
        encoder(product_input)
    assert actual.value.args == expected.value.args and vars(actual.value) == vars(expected.value)


def test_exact_run_hasher_reuses_only_its_exact_benchmark_object(monkeypatch) -> None:
    product_input = _product_input()
    payload = Mock(wraps=price_product_module._input_series_payload)
    monkeypatch.setattr(price_product_module, "_input_series_payload", payload)
    hasher = price_product_module._CompleteInputHasher()
    original = hasher(product_input)
    assert hasher(product_input) == original
    assert payload.call_count == 3  # Both stocks, one benchmark.
    equivalent = replace(product_input, benchmark=replace(product_input.benchmark))
    assert equivalent == product_input and equivalent.benchmark is not product_input.benchmark
    assert hasher(equivalent) == original
    assert payload.call_count == 5  # Equal copied identity is not reused.
    assert price_product_module._CompleteInputHasher()(product_input) == original
    assert payload.call_count == 7  # Another invocation owns no previous representation.
    assert complete_input_hash(product_input) == complete_input_hash(product_input) == original
    assert payload.call_count == 11  # Public hashing remains one-shot.


@pytest.mark.parametrize("role", ["stock", "benchmark"])
@pytest.mark.parametrize(
    "value", [True, None, math.nan, math.inf, -math.inf, pytest.param(10**1000, id="overflow")]
)
def test_warm_exact_run_hasher_preserves_numeric_failures(role, value) -> None:
    product_input = _product_input()
    hasher = price_product_module._CompleteInputHasher()
    original = hasher(product_input)
    series = getattr(product_input, role)
    changed = replace(product_input, **{role: replace(series, closes=(value, *series.closes[1:]))})
    with pytest.raises((PriceProductInputError, OverflowError)) as expected:
        complete_input_hash(changed)
    with pytest.raises(type(expected.value)) as actual:
        hasher(changed)
    assert actual.value.args == expected.value.args and vars(actual.value) == vars(expected.value)
    assert hasher(product_input) == original


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        math.nextafter(0.0, 1.0),
        -math.nextafter(0.0, 1.0),
        float.fromhex("0x1.fffffffffffffp+1023"),
        -float.fromhex("0x1.fffffffffffffp+1023"),
        0,
        1,
        -1,
        2**63,
        pytest.param(10**1000, id="overflowing-int"),
        True,
        False,
        None,
        "1",
        np.float32(1),
        np.float64(1),
        np.int64(1),
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_canonical_float_preserves_exact_base_bytes_and_exceptions(value) -> None:
    def original():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise PriceProductInputError(
                "input_hash_nonfinite", "Canonical input values must be finite"
            )
        return float(value).hex()

    try:
        expected = original()
    except (PriceProductInputError, OverflowError) as error:
        with pytest.raises(type(error)) as actual:
            price_product_module._canonical_float(value)
        assert actual.value.args == error.args and vars(actual.value) == vars(error)
    else:
        assert price_product_module._canonical_float(value) == expected


def test_canonical_float_preserves_subclass_conversion_instead_of_custom_hex() -> None:
    class CustomFloat(float):
        calls = 0

        def __float__(self):
            self.calls += 1
            return 2.0

        def hex(self):
            raise AssertionError("subclass hex must not replace original float conversion")

    value = CustomFloat(1)
    assert price_product_module._canonical_float(value) == (2.0).hex()
    assert value.calls == 2


def test_method_identity_is_stable_while_issuance_versions_remain_unique() -> None:
    first = prediction_model_version(
        method_version=FHS_METHOD_VERSION,
        issuance_id=UUID("11111111-1111-4111-8111-111111111111"),
    )
    second = prediction_model_version(
        method_version=FHS_METHOD_VERSION,
        issuance_id=UUID("22222222-2222-4222-8222-222222222222"),
    )

    assert first == "us-price-fhs-v1-11111111"
    assert second == "us-price-fhs-v1-22222222"
    assert first != second
    assert FHS_METHOD_VERSION == "us-price-fhs-v1"


def test_split_equivalent_scaling_preserves_signal_risk_returns_and_turnover() -> None:
    config = load_price_product_config()
    baseline = calculate_price_product(_product_input(), config=config)
    scaled = calculate_price_product(
        _product_input(stock_scale=0.5, volume_scale=2.0),
        config=config,
    )

    assert baseline.momentum is not None and scaled.momentum is not None
    assert scaled.momentum.stock_log_momentum == pytest.approx(baseline.momentum.stock_log_momentum)
    assert scaled.momentum.relative_log_momentum == pytest.approx(
        baseline.momentum.relative_log_momentum
    )
    assert scaled.recommendation.raw_direction == baseline.recommendation.raw_direction
    assert scaled.risk.relative_volatility == pytest.approx(baseline.risk.relative_volatility)
    assert scaled.risk.average_dollar_turnover_20d == pytest.approx(
        baseline.risk.average_dollar_turnover_20d
    )
    for original, transformed in zip(
        baseline.forecast.projections, scaled.forecast.projections, strict=True
    ):
        assert original.raw_returns is not None
        assert transformed.raw_returns is not None
        assert (
            transformed.raw_returns.lower,
            transformed.raw_returns.median,
            transformed.raw_returns.upper,
        ) == pytest.approx(
            (
                original.raw_returns.lower,
                original.raw_returns.median,
                original.raw_returns.upper,
            )
        )
        assert original.raw_prices is not None
        assert transformed.raw_prices is not None
        assert transformed.raw_prices.lower == pytest.approx(original.raw_prices.lower * 0.5)


def test_affordability_is_separate_from_the_scale_invariant_raw_signal() -> None:
    config = load_price_product_config()
    normal = calculate_price_product(_product_input(), config=config)
    cheap = calculate_price_product(
        _product_input(stock_scale=0.01, volume_scale=100.0),
        config=config,
    )

    assert normal.recommendation.raw_direction == cheap.recommendation.raw_direction
    assert normal.momentum is not None and cheap.momentum is not None
    assert normal.momentum.relative_log_momentum == pytest.approx(
        cheap.momentum.relative_log_momentum
    )
    assert cheap.recommendation.suggestion == "hold"
    assert "target_close_below_buy_minimum" in cheap.recommendation.blocking_reasons
    assert (
        cheap.recommendation.allocation_restriction == "speculative_watch_0_percent_new_allocation"
    )


def test_missing_volume_withholds_buy_without_withholding_forecast() -> None:
    config = load_price_product_config()
    product_input = _product_input()
    product_input = replace(
        product_input,
        stock=replace(product_input.stock, volumes=None),
    )

    result = calculate_price_product(product_input, config=config)

    assert all(
        projection.insufficiency_reason is None for projection in result.forecast.projections
    )
    if result.recommendation.raw_direction == "positive":
        assert result.recommendation.suggestion == "hold"
        assert "dollar_turnover_unavailable" in result.recommendation.blocking_reasons


def test_advisory_projection_failure_cannot_change_an_eligible_momentum_buy() -> None:
    config = load_price_product_config()
    raw_stock = _closes(base=100.0, drift=0.025, wave=0.0005)
    raw_benchmark = _closes(base=400.0, drift=0.005, wave=0.0005)
    stock = tuple(value * (100.0 / raw_stock[-1]) for value in raw_stock)
    benchmark = tuple(value * (400.0 / raw_benchmark[-1]) for value in raw_benchmark)

    result = calculate_price_product(
        _product_input(stock_closes=stock, benchmark_closes=benchmark),
        config=config,
    )

    assert result.momentum is not None
    assert result.momentum.direction == "positive"
    assert result.risk.relative_volatility is not None
    assert result.risk.relative_volatility <= 2
    assert result.risk.maximum_drawdown >= -0.5
    assert result.risk.average_dollar_turnover_20d is not None
    assert result.risk.average_dollar_turnover_20d >= 5_000_000
    projections = {projection.horizon: projection for projection in result.forecast.projections}
    assert projections["5y"].insufficiency_reason == "projection_database_unrepresentable"
    assert result.recommendation.suggestion == "buy"
    assert result.recommendation.blocking_reasons == ()


def test_explicit_demo_mode_accepts_only_honest_synthetic_research_assets() -> None:
    config = load_price_product_config()
    demo_input = _product_input(
        source_mode="synthetic_demo",
        evidence_grade="research",
        stock_provider="synthetic_demo",
        benchmark_provider="synthetic_demo",
    )

    result = calculate_price_product(demo_input, config=config)

    assert result.momentum is not None
    assert result.source_execution == demo_input.source_execution
    assert result.input_hash == complete_input_hash(demo_input)
    assert all(
        projection.insufficiency_reason is None for projection in result.forecast.projections
    )


@pytest.mark.parametrize(
    ("product_input", "reason"),
    [
        (
            _product_input(
                stock_provider="synthetic_demo",
                benchmark_provider="twelve_data",
            ),
            "stock_provider_invalid",
        ),
        (
            _product_input(
                stock_provider="synthetic_demo",
                benchmark_provider="synthetic_demo",
            ),
            "stock_provider_invalid",
        ),
        (
            _product_input(
                source_mode="synthetic_demo",
                stock_provider="twelve_data",
                benchmark_provider="twelve_data",
            ),
            "stock_provider_invalid",
        ),
        (
            _product_input(
                source_mode="synthetic_demo",
                stock_provider="synthetic_demo",
                benchmark_provider="twelve_data",
            ),
            "benchmark_provider_invalid",
        ),
        (
            _product_input(
                source_mode="synthetic_demo",
                evidence_grade="observed",
                stock_provider="synthetic_demo",
                benchmark_provider="synthetic_demo",
            ),
            "synthetic_demo_observed_forbidden",
        ),
    ],
)
def test_source_execution_mode_rejects_mislabeled_or_mixed_assets(
    product_input: PriceProductInput,
    reason: str,
) -> None:
    with pytest.raises(PriceProductInputError) as caught:
        calculate_price_product(product_input, config=load_price_product_config())

    assert caught.value.reason_code == reason


def test_degenerate_filter_withholds_every_triplet_without_fake_zero() -> None:
    config = load_price_product_config()
    constant = tuple(100.0 for _index in range(757))
    result = calculate_price_product(
        _product_input(stock_closes=constant),
        config=config,
    )

    assert result.forecast.insufficiency_reason == "filter_variance_degenerate"
    assert result.forecast.mean_log_return is None
    for projection in result.forecast.projections:
        assert projection.central_model_mass == 0.6
        assert projection.raw_returns is None
        assert projection.ledger_returns is None
        assert projection.raw_prices is None
        assert projection.insufficiency_reason == "filter_variance_degenerate"
    assert result.overall_score is None
    assert result.risk_score is None
    assert result.confidence is None
    assert result.probability_positive is None


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda value: replace(
                value,
                calendar_sessions=value.calendar_sessions[:-1],
            ),
            "calendar_session_count",
        ),
        (
            lambda value: replace(
                value,
                stock=replace(
                    value.stock,
                    dates=(
                        *value.stock.dates[:-2],
                        value.stock.dates[-2],
                        value.stock.dates[-2],
                    ),
                ),
            ),
            "stock_target_mismatch",
        ),
        (
            lambda value: replace(
                value,
                stock=replace(value.stock, currency="EUR"),
            ),
            "stock_currency_invalid",
        ),
        (
            lambda value: replace(
                value,
                stock=replace(
                    value.stock,
                    identity=replace(
                        value.stock.identity,
                        available_at=value.decision_time + timedelta(seconds=1),
                    ),
                ),
            ),
            "stock_asset_after_decision",
        ),
        (
            lambda value: replace(
                value,
                stock=replace(
                    value.stock,
                    identity=replace(
                        value.stock.identity,
                        subject="SPY",
                    ),
                ),
            ),
            "stock_subject_is_benchmark",
        ),
    ],
)
def test_invalid_or_asof_unsafe_windows_fail_explicitly(mutation, reason: str) -> None:
    config = load_price_product_config()

    with pytest.raises(PriceProductInputError) as caught:
        calculate_price_product(mutation(_product_input()), config=config)

    assert caught.value.reason_code == reason


def test_negative_signal_remains_avoid_when_buy_inputs_are_missing() -> None:
    config = load_price_product_config()
    stock = [100.0] * 757
    benchmark = [100.0] * 757
    stock[504], stock[735] = 120.0, 80.0
    benchmark[504], benchmark[735] = 110.0, 90.0
    momentum = calculate_momentum(tuple(stock), tuple(benchmark))
    risk = calculate_price_product(
        _product_input(stock_closes=tuple(stock), benchmark_closes=tuple(benchmark)),
        config=config,
    ).risk

    recommendation = apply_recommendation_policy(
        momentum=momentum,
        momentum_insufficiency_reason=None,
        risk=replace(risk, relative_volatility=None, average_dollar_turnover_20d=None),
        target_close=1.0,
        source_eligible=False,
        source_ineligibility_reasons=("source_unverified",),
        config=config,
    )

    assert recommendation.suggestion == "avoid"
    assert recommendation.blocking_reasons == ()
    assert recommendation.allocation_restriction == "speculative_watch_0_percent_new_allocation"


def test_affordability_restriction_applies_to_mixed_and_unavailable_signals() -> None:
    config = load_price_product_config()
    closes = tuple(100.0 for _index in range(757))
    mixed = calculate_momentum(closes, closes)
    risk = calculate_price_product(_product_input(), config=config).risk

    mixed_result = apply_recommendation_policy(
        momentum=mixed,
        momentum_insufficiency_reason=None,
        risk=risk,
        target_close=1.0,
        source_eligible=True,
        source_ineligibility_reasons=(),
        config=config,
    )
    unavailable_result = apply_recommendation_policy(
        momentum=None,
        momentum_insufficiency_reason="momentum_history_insufficient",
        risk=risk,
        target_close=1.0,
        source_eligible=True,
        source_ineligibility_reasons=(),
        config=config,
    )

    assert mixed_result.suggestion == "hold"
    assert mixed_result.blocking_reasons == ("mixed_momentum_signal",)
    assert unavailable_result.suggestion is None
    assert unavailable_result.blocking_reasons == ("momentum_history_insufficient",)
    assert (
        mixed_result.allocation_restriction
        == unavailable_result.allocation_restriction
        == "speculative_watch_0_percent_new_allocation"
    )


def test_future_numeric_failure_preserves_completed_terminal_horizon() -> None:
    filtered = FilteredReturns(
        mean_log_return=0.0,
        population_variance=1.0,
        terminal_variance=1.0,
        standardized_returns=tuple(0.0 for _index in range(756)),
        residuals=tuple(6.0 for _index in range(504)),
        residual_center=0.0,
        residual_scale=1.0,
    )

    terminals = simulate_fhs_terminal_logs(
        filtered,
        seed=17,
        horizons=(1, 1000),
        path_count=1,
    )

    assert terminals.horizons == (1, 1000)
    assert terminals.with_drift[0][0] == pytest.approx(6.0)
    assert terminals.zero_drift[0][0] == pytest.approx(6.0)
    assert np.isnan(terminals.with_drift[1]).all()
    assert np.isnan(terminals.zero_drift[1]).all()


def test_partial_numeric_failure_keeps_earlier_triplets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_price_product_config()
    path_count = config.simulation.production_paths
    first = np.linspace(-0.1, 0.1, path_count)
    second = np.linspace(-0.2, 0.2, path_count)
    partial = SimulationTerminals(
        horizons=(126, 252, 756, 1260),
        path_count=path_count,
        with_drift=(
            first,
            second,
            np.full(path_count, np.nan),
            np.full(path_count, np.nan),
        ),
        zero_drift=(
            first,
            second,
            np.full(path_count, np.nan),
            np.full(path_count, np.nan),
        ),
    )
    monkeypatch.setattr(
        price_product_module,
        "simulate_fhs_terminal_logs",
        lambda *args, **kwargs: partial,
    )
    filtered = filter_historical_returns(_closes(base=100.0, drift=0.0007, wave=0.006))

    forecast = project_fhs(filtered, target_close=100.0, seed=91, config=config)

    assert [item.insufficiency_reason for item in forecast.projections] == [
        None,
        None,
        "simulation_nonfinite",
        "simulation_nonfinite",
    ]
    assert all(item.ledger_returns is not None for item in forecast.projections[:2])
    assert all(item.raw_returns is None for item in forecast.projections[2:])


def test_unrepresentable_forward_paths_withhold_instead_of_clipping() -> None:
    config = load_price_product_config()
    residuals = tuple((-1.0 if index % 2 else 1.0) for index in range(504))
    filtered = FilteredReturns(
        mean_log_return=10.0,
        population_variance=1.0,
        terminal_variance=1.0,
        standardized_returns=tuple(0.0 for _index in range(756)),
        residuals=residuals,
        residual_center=0.0,
        residual_scale=1.0,
    )

    forecast = project_fhs(filtered, target_close=100.0, seed=9, config=config)

    assert all(
        projection.insufficiency_reason is not None and projection.raw_returns is None
        for projection in forecast.projections
    )
