# Deterministic Forecast Roadmap

StanStock's research roadmap adds medium- and long-horizon forecasts without
LLMs, machine-learning models, analyst targets, or generated recommendations.
Every forecast comes from versioned arithmetic, point-in-time source data, and
empirical historical distributions.

## Forecast contract

Every forecast must:

- use only information published by its decision time;
- store its source assets, formula version, inputs, output, and code revision;
- show bear, base, and bull cases rather than one precise target;
- explain the numerical contribution from growth, valuation, momentum, and
  risk, while reporting supported shareholder distributions separately;
- remain unavailable when required inputs or comparable observations are
  missing;
- be evaluated through walk-forward tests and later immutable live outcomes;
- distinguish price return from total return when dividend data is absent.

The implementation will not use an LLM, neural network, fitted black-box
model, or automated parameter search. Medians, percentiles, compounding,
shrinkage, and fixed formulas are the permitted tools.

## Released forecast foundation

The persistence and evaluation boundary now distinguishes scoring groups from
forecast identities:

- scoring continues to use `short`, `medium`, and `long`;
- new advisory predictions use explicit `6m`, `12m`, `3y`, and `5y`
  identities;
- historical `medium` and `long` predictions retain their original labels and
  252-/756-session maturity semantics;
- analyses expose one schema-versioned scenario document while retaining the
  three legacy columns temporarily for rollback compatibility;
- every new prediction records a decision/advisory role, immutable evidence
  grade/source mode, exact price provider and source subject when proven, and
  structured calculation provenance;
- advisory outcomes use direction correctness, interval coverage, signed
  error, and benchmark return, with no recommendation-success value;
- decision performance and opportunity policy explicitly exclude advisory
  forecasts, and advisory metrics remain separated by exact method,
  configuration, provider, evidence grade, and horizon.

## Stage 1: price-only 6-12 month ranges

**Released.** This stage uses the existing Twelve Data histories before
fundamentals are available and replaces an unconditional rolling-return range
with separate conditional empirical 6- and 12-month ranges.

For each historical decision date, StanStock describes the stock using
only values known on that date:

- SPY-relative 12-month momentum;
- drawdown from the 52-week high;
- trailing volatility;
- SPY trend and volatility regime.

The panel also records 50/200-session trend, downside volatility, and dollar
liquidity as eligibility, risk, and explanation inputs rather than additional
matching axes. The current stock is matched to prior observations in the same versioned
state buckets. The 20th, 50th, and 80th percentiles of the subsequent 126- and
252-session price returns become the bear, base, and bull cases.

Sparse buckets will be shrunk toward the unconditional market distribution:

```text
weight = effective_non_overlapping_cohorts
         / (effective_non_overlapping_cohorts + shrinkage_prior_cohorts)
forecast_quantile =
    weight * matched_bucket_quantile
    + (1 - weight) * market_quantile
```

Rows are weighted so each market cohort contributes equal total weight even
when many current-universe stocks share it. Bucket boundaries, fallback order,
support floors, shrinkage, and calibration gates are frozen in
`config/forecasts/us-price-medium-v1.yml`, not selected by maximizing backtest
results. Probability of a positive return remains hidden until effective
cohort support, listing diversity, calendar span, matched market-regime
breadth, and walk-forward calibration all qualify.
The complete panel is stored as a private immutable Parquet asset with source,
calendar, configuration, content, and code hashes. This stage is explicitly
labeled `price-only` and cannot produce a fundamental long-term forecast or
alter recommendation policy.

## Stage 2: point-in-time US fundamentals

**Released.** The current 100-stock US universe now uses the official SEC
ticker/exchange/CIK mapping, submissions plus referenced history files, and
Companyfacts.

Long-term forecasts require business fundamentals. The first supported scope
will be the current US universe using SEC submissions and Companyfacts.

The ingestion layer retains accession, reporting period, unit, filing
acceptance time, amendment, and first-seen time. It will derive trailing
twelve-month or annual values only from filings available at the decision
time. Required canonical inputs are:

- revenue, operating income, net income, and diluted EPS;
- operating cash flow, capital expenditure, and free cash flow;
- cash, separately identified debt components, equity, and interest expense;
- diluted shares and per-share values;
- dividends and repurchases only when the source supports them consistently.

Growth is calculated per share where dilution matters. Restatements never
rewrite an older prediction's input. A concept that cannot be mapped
unambiguously remains missing.

The released implementation also:

- preserves identical raw content once and records idempotent job recovery;
- distinguishes instant, duration, and conservatively unclassified periods;
- appends changed observations under the same accession as source revisions;
- joins Companyfacts `accn` values to exact submissions acceptance times;
- links each normalized fact to the immutable submissions/history asset that
  supplied that acceptance boundary;
- uses next-day New York availability only when the SEC exposes a filing date
  without acceptance time;
- derives compatible discrete quarters, annual series, TTM flows, weighted
  diluted shares, and free cash flow without treating instant values as flows;
- stores current SIC only as a retrieval-time immutable observation;
- polls submissions daily, retries a newly missing Companyfacts accession once
  daily for seven days, and bounds later refreshes with staggered
  reconciliation;
- keeps SPY and every ETF outside corporate fundamentals.

## Stage 3: deterministic 6-12 month fundamental forecast

The medium-horizon base case will use the identity:

```text
price = per_share_fundamental * valuation_multiple
```

For a supported fundamental such as EPS or free cash flow per share:

```text
growth_base = shrink(historical_growth, sector_growth)
target_multiple = geometric_median(company_history, sector_peers)
future_multiple =
    current_multiple ** (1 - reversion_fraction)
    * target_multiple ** reversion_fraction

base_price_return =
    (1 + growth_base)
    * (future_multiple / current_multiple)
    - 1
```

Stored scenario returns remain price returns so they can be compared directly
with price-only outcomes. Supported dividends or other cash distributions may
be shown as separate context, but they cannot be added to these fields unless
a future version introduces an explicit total-return source, storage contract,
and matching outcome evaluator.

The historical growth estimate will use robust medians of distinct prior
period growth rates and will be capped to prevent one unusual comparison from
dominating the result. Shrinkage pulls a short or unstable company history
toward the point-in-time sector median. The fixed reversion fraction represents
partial, not complete, valuation normalization over one year.

Bear and bull cases will change only explicit assumptions:

| Input | Bear | Base | Bull |
|---|---|---|---|
| Per-share growth | Lower historical/peer percentile | Shrunk median | Upper capped percentile |
| Margin | Recent weak case | Normalized median | Supported recovery case |
| Valuation | Lower historical/peer percentile | Partial normalization | Upper capped percentile |
| Market regime | Negative fixed overlay | No overlay | Positive fixed overlay |
| Risk spread | Wider for volatile or leveraged firms | Normal | Never narrower than the empirical floor |

Momentum may provide a small, capped overlay; it cannot compensate for missing
fundamentals or turn an unsupported forecast into a BUY.

## Stage 4: deterministic 3- and 5-year forecast

The long-horizon forecast will estimate business growth first and valuation
second.

Sustainable growth is bounded by reinvestment economics:

```text
reinvestment_rate = retained_operating_cash / supported_operating_base
sustainable_growth = ROIC * reinvestment_rate
```

The base growth path blends the company's historical per-share growth,
sustainable growth, and the sector median. It then fades each year toward a
conservative terminal growth rate:

```text
growth_year_t =
    fade_t * company_growth
    + (1 - fade_t) * terminal_growth

fundamental_year_t =
    fundamental_year_(t-1) * (1 + growth_year_t)

terminal_price =
    fundamental_year_T * normalized_terminal_multiple

annualized_return =
    (terminal_price / current_price) ** (1 / T) - 1
```

When free cash flow is the reliable input, the same calculation uses a
normalized free-cash-flow yield instead of inventing EPS. Net debt and dilution
are carried explicitly. Negative or structurally inconsistent fundamentals
produce `insufficient evidence`, not a substituted metric.

The scenario assumptions will be mechanical:

- **Bear:** lower growth, margin compression, no balance-sheet improvement,
  and a lower normalized multiple.
- **Base:** shrunk sustainable growth, normalized margins, and partial
  valuation convergence.
- **Bull:** capped growth and margin improvement supported by company history,
  plus an upper but bounded normalized multiple.

Long-term probability will remain unavailable until enough comparable
walk-forward or live outcomes exist.

## Stage 5: walk-forward validation and uncertainty

Research validation will recreate historical decision dates and expose each
forecast only to data available on that date. Parameters are frozen before the
evaluation window. StanStock will compare each forecast with both the stock's
realized price return and SPY.

Reported evidence will include:

- sample count and coverage;
- median absolute forecast error;
- directional accuracy;
- bear/base/bull interval coverage;
- return and excess-return calibration by forecast bucket;
- results by sector, risk class, and market regime;
- separate research-grade and genuinely on-time live results.

Scenario widths will eventually use the 20th and 80th percentiles of historical
forecast residuals from the same formula version. Until that evidence is
sufficient, a conservative volatility-based floor remains in force.

## Delivery order

1. Build on the released forecast-identity and advisory-outcome schema, then
   implement the conditional price-only 126- and 252-session scenarios and
   expose them as research-grade medium forecasts.
2. Complete point-in-time SEC ingestion and canonical US fundamental
   calculations.
3. Add the versioned medium and long formula configurations, calculation
   records, and deterministic explanation breakdown.
4. Add 3- and 5-year scenario views and immutable prediction records.
5. Add walk-forward evaluation, interval calibration, and model-portfolio
   attribution by formula version.
6. Enable serving only after leakage, missing-data, accounting, and
   reproducibility tests pass.

European long-term forecasts remain out of scope until an equally defensible
point-in-time filing pipeline exists.
