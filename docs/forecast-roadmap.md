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
- advisory outcomes use base-case sign match, bear-to-bull inclusion, signed
  base-case error, and benchmark return, with no recommendation-success value;
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
matching axes. The current stock is matched to prior observations in the same
versioned state buckets. The matched and unconditional cohort-weighted 20th,
50th, and 80th percentiles of the subsequent 126- and 252-session price
returns are each shrinkage blended. The published base is the blended p50;
bear and bull are the blended p20 and p80.

Sparse buckets will be shrunk toward the unconditional market distribution:

```text
weight = effective_non_overlapping_cohorts
         / (effective_non_overlapping_cohorts + shrinkage_prior_cohorts)
forecast_quantile =
    weight * matched_bucket_quantile
    + (1 - weight) * market_quantile
```

Rows are weighted so each market cohort contributes equal total weight even
when many current-universe stocks share it. Bear-to-bull is a nominal central
60% analog-return range. It is not a calibrated prediction, credible, or
confidence interval and has no coverage guarantee.

Bucket boundaries, fallback order,
support floors, shrinkage, and probability-publication gates are frozen in
`config/forecasts/us-price-medium-v1.yml`, not selected by maximizing backtest
results. The positive-return estimate remains hidden until effective
cohort support, listing diversity, calendar span, matched market-regime
breadth, base-case MAE comparisons against the unconditional and SPY-relative
baselines, and the configured absolute Brier threshold all qualify. The
persisted `empirical_calibrated` name records only passage of that publication
gate; it does not prove probability calibration or interval coverage.
The complete panel is stored as a private immutable Parquet asset with source,
calendar, configuration, content, and code hashes. This stage is explicitly
labeled `price-only` and cannot produce a fundamental long-term forecast or
alter recommendation policy.

### Explicit research-only medium v2

`us-price-medium-v1` is frozen and remains the sole default and scheduled
method. The prospective `us-price-medium-v2` lane is selected only by an
explicit service-level config path and only for a research-grade US/USD stock
snapshot, SPY, exact `us-price-baseline-v2`, and literal
`issued_on_time=False`.

V2 keeps the same panel/features/fallback support boundary but uses one
cohort-equal empirical CDF mixture: matched and unconditional components are
normalized independently, conditional weight is `K / (K + 4)`, quantiles use
the generalized left inverse, and `P(return > 0) = 1 - F(0)`. It refuses a
finite historical return below `-1.0`; exactly `-1.0` remains valid.
Prequential evidence trains at origin `o` only on labels ending on or before
`o`, uses a prior-only unconditional probability reference, and gives each
test date equal aggregate weight. Probability requires current support plus
strictly positive raw Brier skill. Base MAE and descriptive central-60%
coverage, miss rates, mean width, and alpha-0.40 interval scores remain
independent evidence.

Every selected panel source must have `available_at <= AnalysisRun.data_cutoff`
before any source file is read; a late selected vintage causes zero physical
reads and no fallback. V2 remains current-universe/survivorship-biased
research evidence. It makes no calibration, statistical-significance,
profitability, alpha, or live-skill claim. See
[`docs/methodology.md`](methodology.md#explicit-research-only-medium-v2).

## Stage 2: point-in-time US fundamentals

**Released.** The current 100-stock US universe now uses the official SEC
ticker/exchange/CIK mapping, submissions plus referenced history files, and
Companyfacts.

Long-term forecasts require business fundamentals. The first supported scope
is the current US universe using SEC submissions and Companyfacts.

The ingestion layer retains accession, reporting period, unit, filing
acceptance time, amendment, and first-seen time. It derives trailing
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

## Stage 3: deferred 6-12 month fundamental variant

**Not enabled.** The released 6m/12m method remains intentionally price-only.
The following identity is retained as a possible separately reviewed future
method, not as current application behavior.

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

**Released.** The long-horizon engine estimates business growth first and
valuation second without changing the short recommendation policy.

Sustainable growth uses bounded accounting proxies:

```text
gaap_accrual_tax_proxy =
    TTM_income_tax_expense / TTM_pretax_income
NOPAT =
    TTM_operating_income * (1 - bounded_gaap_accrual_tax_proxy)
invested_capital = compatible_debt + equity - cash
reinvestment_rate =
    (ending_invested_capital - beginning_invested_capital) / NOPAT
sustainable_growth = ROIC * reinvestment_rate
```

The tax input is a bounded GAAP accrual proxy, not cash taxes paid or a cash
tax rate. Reinvestment is a compatible balance-sheet invested-capital-change
proxy, not directly observed capex and not a proven causal reinvestment rate.

The base growth path blends the company's historical per-share growth,
sustainable growth, and the point-in-time same-family SIC-peer median. It then
fades each year toward a conservative terminal growth rate:

```text
g0 = cap(
    0.45 * historical_per_share_growth
  + 0.30 * sustainable_growth
  + 0.25 * peer_per_share_growth
)

growth_year_t = fade_t * g0 + (1 - fade_t) * terminal_growth

bounded_current_multiple = cap(actual_current_multiple)

terminal_multiple = capped_geometric_interpolation(
    bounded_current_multiple,
    adjusted_peer_multiple,
    reversion
)

cumulative_price_return =
    product(1 + growth_year_t)
    * terminal_multiple / actual_current_multiple
    - 1

annualized_return =
    (1 + cumulative_price_return) ** (1 / T) - 1
```

The 3-year and 5-year calculations use separate frozen horizon-specific fade
sequences and multiple-reversion settings. Neither is a slice or extrapolation
of one coherent shared 5-year path.

Positive compatible FCF/share takes priority. EPS/share is considered only
when FCF evidence is genuinely unavailable; negative, weak, or
share-inconsistent FCF blocks silent switching. Both branches require at
least three contiguous annual periods, a reported diluted-EPS consistency
check for every selected period, and TTM diluted shares within 15% of the
latest overlapping annual basis. Beginning and ending invested capital must
also use identical canonical and source concepts for cash, equity, and debt.
A raw current multiple below its configured family floor is withheld rather
than mechanically raised before reversion. Unsupported financial SICs,
missing classifications, stale metrics, missing sustainable-growth terms,
incompatible price evidence, and peer sets below fixed SIC-4/SIC-3/SIC-2
floors produce `Insufficient evidence`.

The `us-sec-long-v1` configuration is frozen and remains reproducible exactly
as originally released: reported diluted-EPS consistency for every selected
annual period plus TTM-to-latest-annual-share continuity, with no check
between adjacent selected annual periods. `us-sec-long-v2` retains every
v1 assumption except that it enables
`adjacent_selected_annual_diluted_share_continuity`, which also requires each
adjacent pair of selected annual diluted-share bases to stay within the same
15% tolerance; an incompatible adjacent pair is withheld as unverified
continuity, never asserted as a confirmed split. Persisted output is not
otherwise identical: v2 also persists the structured assessed
share-consistency evidence on share-basis failures, while frozen v1's
withheld-failure payloads are unchanged. `us-sec-long-v2` is the default
configuration for new analyses. Predictions, method versions, configuration
hashes, and performance cohorts from the two configurations never mix: v1
predictions already recorded remain immutable and keep scoring under their
original pinned configuration hash.

### `us-sec-long-v3`: prospective evidence selection only

**Not the default, and not approved for activation.** (Scope: the SEC
correction-availability integrity fix shipped alongside it *is* active in the
default configuration and benefits every reader; only the long-v3 reader
below is inactive.) `us-sec-long-v3` exists
as a prospective configuration for the replay phase. It changes evidence
selection only; every formula weight, bound, cap, fade path, multiple
reversion, peer floor, metric-family rule, freshness limit, tax proxy,
scenario constant, probability withholding, return basis, and
unsupported-SIC policy is identical to long-v2, and the pinned
long-v1/long-v2 config bytes, effective hashes, behavior, reason strings, and
persisted payloads stay frozen. New analyses continue to run on
`us-sec-long-v2`.

**No eligibility improvement is claimed.** Nothing here has been measured
against a historical panel. Long-v3 changes *which* already-persisted
observations the unchanged arithmetic reads, and it can just as easily
withhold a forecast that long-v2 produced -- for example when the alias that
legitimately anchors the newest quarter cannot supply a homogeneous
four-quarter tail. Whether the change is a net benefit is an open question
that a cutoff-safe replay must answer before any activation decision.

Long-v3 declares three default-off capabilities. A configuration that does
not declare them parses to "absent", which is removed from the effective
hash, so adding them cannot change an already-frozen version's hash.

- `newest_quarter_anchored_homogeneous_ttm_alias_selection`: for each
  canonical TTM concept, anchor on the newest eligible quarter end, rank the
  observations at exactly that quarter, and build the trailing twelve months
  only from four contiguous compatible quarters supplied by that same source
  alias (350-380 day span). No cross-alias stitching, no preference for a
  stale-but-complete alias over a newer restated newest-quarter observation,
  and no annual current-period fallback. Annual history selection, generic
  consumers, and long-v1/long-v2 keep the frozen legacy path.

  A quarter can be reported directly or derived from the difference of two
  year-to-date filings. In the derived case the observation is ranked by the
  single *controlling* source fact -- the dependency that actually gates its
  knowability under the existing lexicographic availability/revision/source
  priority/accession order -- never by independently maximizing availability,
  revision, and accession across the two filings. That synthesis would
  describe a vintage nobody ever filed and could hand the anchor to the wrong
  alias. The selected alias, the controlling fact, and the per-quarter
  lineage of the chosen window are recorded in the calculation provenance.
- `joint_compatible_invested_capital_pair_selection`: jointly search all
  viable beginning and ending balance-sheet snapshots within the unchanged
  +/-7-day tolerance, accept only pairs whose debt method, debt components,
  and canonical/source concept bases match exactly, and rank the eligible
  pairs purely on deterministic evidence criteria (combined and per-side
  target-date distance, eligible-evidence recency, then declared alias
  priority, canonical source basis, and stable date/fact-id tie-breaks).
  Ranking never consults the resulting forecast, ROIC, reinvestment, growth,
  or scenario favorability. With no compatible pair the forecast is withheld
  with an explicit reason.

  The normalized instant series collapses every canonical concept to one
  winning alias per period identity, so a pair that exists only through a
  *non-winning same-date* equity, cash, or debt alias is invisible there.
  Long-v3 therefore reads an additional candidate surface that retains the
  latest eligible vintage per `(canonical concept, source alias, period
  identity)` and enumerates the permitted same-date source-basis
  combinations before pairing. A reported long-term debt roll-up still
  forbids the component basis at that date, so no debt component is ever
  counted twice, and superseded revisions are still discarded. Legacy
  `instants`, long-v1, and long-v2 are untouched.
- `proven_observation_correction_availability`: resolve each same-accession
  correction against the observation that proves it rather than against a
  recorded availability an earlier ingestion may have backdated to the
  original filing acceptance. A correction whose timing is not proven at the
  requested cutoff is withheld from the series, the revision it superseded
  stands in its place, and the withheld row is reported as assessed evidence
  with its resolved availability and reason. The capability is declared and
  hashed on its own, so adopting either capability above never silently
  acquires it; frozen long-v1 and long-v2 declare no such key and keep
  reading recorded availability exactly as released.

Every pair search is recorded as an assessment, including the ones that
reject the evidence. A missing beginning side, a missing ending side, zero
compatible pairs, a selected pair later withheld for an unrelated reason
(such as an insufficient peer set), and a successful selection all persist
the targets, the candidate snapshots with their facts and source bases, the
compatible-pair count, the status, and the rejection reason. Rejected
evidence is labeled *assessed*, never verified, and never implies a
confirmed selection.

Three boundaries are refused rather than resolved, because resolving them
would silently decide the answer:

- **Non-canonical instant identity.** The same-date alias join is only sound
  while every instant observation for one balance-sheet date carries exactly
  the canonical `instant::<date>` period identity. A conflicting or
  mislabelled identity would either split one date into pseudo-dates or make
  two different observations look interchangeable, so long-v3 withholds that
  listing with the offending rows named. It never breaks such a tie by row
  identifier.
- **Generated row identifiers.** No long-v3 ordering, ranking, or tie-break
  reads a fact's primary key. Alias ordering, latest-revision selection,
  same-date candidate ordering, and pair ranking all fall through to the
  persisted observation identity, so reassigning UUIDs cannot change a
  selection.
- **Same-date combination ceiling.** The number of permitted same-date
  source-basis combinations is computed from the per-axis alias counts
  *before* any Cartesian product is built, and a date above the reviewed
  ceiling of 256 withholds that listing explicitly. The refusal keeps the
  evidence that disqualified the date: each independent axis is recorded with
  its concept, alias source concepts, fact ids, option count, and the factor
  it contributes, so every fact responsible for the bound is named and
  manifest-covered without the run ever enumerating a single combination.
  Nothing is truncated, the ceiling is not raised, no product is listed, an
  already-assessed opposite side is retained, and the surrounding analysis
  run and audit continue for every other listing.

Every fact these assessments referenced -- selected, rejected, unpaired,
responsible for a refused combination space, or withheld later for an
unrelated reason -- is covered by the immutable forecast `source_assets`
manifest through its companyfacts and filing-evidence assets. The payload
separates that evidence explicitly into three lists rather than leaving the
classification to be inferred:

- `selected_input_fact_ids` -- exactly the facts described in `input_facts`,
  each one an input to the metric, share-consistency, or sustainable-growth
  arithmetic;
- `assessed_evidence_fact_ids` plus the described `assessed_evidence` --
  candidates that were read and considered but not selected, including every
  failure-path candidate. A no-compatible-pair, missing-side, refused-
  boundary, or unusable-metric result selects nothing, so its candidates are
  assessed only;
- `manifest_evidence_fact_ids` -- the union the source-asset manifest closes
  over.

The first two are disjoint by construction, so a rejected filing can never
read as a verified one. Frozen long-v1/long-v2 payloads gain none of this and
keep their original `input_facts` classification unchanged.

Long-v3 also explicitly binds the SEC fundamentals configuration identity it
selects against (`us-sec-fundamentals-v1`, unchanged and unedited).

#### Cutoff-safe replay observation (not an activation basis)

A read-only, provider-free replay of the 97 listings analyzed in the most
recent completed local run (target date 2026-09-08, that run's own
`data_cutoff` and as-of boundary, no database write) reproduced the run's
persisted long-v2 eligibility exactly and then re-ran the same inputs under
long-v3:

| | 3y | 5y |
|---|---|---|
| long-v2 eligible | 2 / 97 | 2 / 97 |
| long-v3 eligible | 5 / 97 | 5 / 97 |
| recovered by long-v3 | 3 | 3 |
| lost under long-v3 | 0 | 0 |

All three recoveries (ABT, ADI, LIN) were listings long-v2 withheld with
"Beginning/end invested-capital evidence uses incompatible source
definitions". Counting that snapshot's 19 listings in that family
individually: 14 reached a later gate and failed the unchanged peer floor, 3
became eligible, and 2 still report an explicit "no compatible beginning/end
invested-capital pair", so the legacy reason falls from 19 listings to 0. One
further listing that long-v2 withheld for a missing TTM pretax input now
reaches the pair search and becomes the third no-compatible-pair result, so
that family moves from 3 listings to 2. Every other withholding family
(unsupported SIC, FCF-branch failure, the remaining missing-TTM-input cases)
is unchanged. These are counts from one snapshot, independently re-derived
from that replay's per-listing reasons; they describe where withholding moved,
not whether the change is beneficial.

**This is one local run on one target date and is not evidence of benefit.**
Three recoveries out of 97 on a single snapshot cannot distinguish a genuine
selection improvement from a fixture artifact, the sample says nothing about
forecast accuracy, and the dominant constraint at this cutoff is the peer
floor rather than evidence selection. Long-v3 remains prospective and
inactive; `us-sec-long-v2` remains the default.

`manage.py audit_long_evidence` reports newest-quarter alias selection with
its controlling source fact, homogeneous four-quarter tail availability,
alias collisions and stale-complete alternatives, and invested-capital pair
assessment. It is read-only and offline: it reads persisted immutable rows
through the point-in-time gate, makes no provider request, and writes no row,
asset, or file.

The audit mirrors the forecast's boundaries instead of collapsing them, and
fails closed rather than guessing any of them:

```text
python manage.py audit_long_evidence \
    --listing-ids <uuid>[,<uuid>...] \
    --target-date 2026-02-27 \
    --available-through 2026-03-01T12:00:00+00:00 \
    --decision-time 2026-03-01T12:00:00+00:00 \
    --json
```

- `--target-date` bounds which reporting periods may enter the window, so a
  quarter ending between the target date and the audit run is excluded
  exactly as a forecast would exclude it;
- `--available-through` is the historical data cutoff applied to fact
  availability, so a post-cutoff restatement stays out of a reconstruction
  even when a later reader can see it;
- `--decision-time` is the as-of boundary for evidence and source-asset
  visibility, which is what admits a *later-retrieved* source asset into a
  research-grade reconstruction.

A naive timestamp, an `--available-through` after the decision time, a
`--target-date` after the cutoff, or an `AsOfData` reader whose decision
boundary differs from the requested one is rejected rather than reconciled.

Listings are identified by their immutable listing ID. A ticker is accepted
only as operator convenience and only when exactly one listing matches; the
same symbol on two exchanges, or a symbol an exchange reused after a
delisting, is an explicit ambiguity error naming the candidates, never a
silently chosen winner.

The scenario assumptions are mechanical:

- **Bear:** fixed negative growth delta, lower reinvestment multiplier, and
  lower peer-multiple multiplier.
- **Base:** no growth delta and neutral reinvestment/peer-multiple multipliers.
- **Bull:** fixed positive growth delta and bounded higher multipliers.

Each advisory prediction stores complete target fact values, periods,
accessions, availability timestamps, source revisions, Companyfacts assets,
and filing-evidence assets, plus exact peer fact references, classification
observations, price assets, formula paths, and configuration hashes. Target
facts are inlined completely; peer inputs use immutable fact references plus
self-contained derived growth and multiple values to avoid duplicating the
full peer panel in every row.
Stored scenarios are cumulative split-adjusted price returns; annualized
values are display-only, dividends are excluded, and probability remains
unavailable until qualifying prospective outcomes exist. Each calculation
also records the days between its latest verified SEC share basis and target
as residual, unverified post-period split exposure.

## Stage 5: walk-forward validation and uncertainty

Research validation will recreate historical decision dates and expose each
forecast only to data available on that date. Parameters are frozen before the
evaluation window. StanStock will compare each forecast with both the stock's
realized price return and SPY.

The released advisory performance reader now reports:

- exact method, configuration, provider, horizon, evidence-grade, and
  code-revision groups;
- candidate target-date cohorts, selected non-overlapping cohorts, listing
  breadth floors, and target-date span;
- signed base-case error;
- base-case sign match;
- 6m/12m analog-range inclusion, compared with its nominal central 60% target
  only after the reporting support floors pass;
- 3y/5y deterministic scenario-envelope inclusion without a nominal target;
- canonical observed, on-time, provider-backed outcomes only. Reconstructed
  research-grade results remain excluded from this reader.

Each selected target date receives equal weight. Exact code revisions remain
separate, and malformed evidence or insufficient overlap-aware support
withholds all three advisory metrics for that group.

Future validation work still includes:

- median absolute base-case error;
- positive-return reliability by forecast bucket;
- results by sector, risk class, and market regime;
- benchmark-relative summaries and portfolio-level reporting.

Neither the released support floors nor these planned summaries claim that
current forecasts or probability estimates are calibrated.

Scenario widths will eventually use the 20th and 80th percentiles of historical
forecast residuals from the same formula version. Until that evidence is
sufficient, a conservative volatility-based floor remains in force.

## Delivery order

1. **Released:** forecast identity, advisory outcomes, and compatibility
   migration.
2. **Released:** deterministic price-only 6m/12m forecasts.
3. **Released:** point-in-time SEC ingestion and canonical US fundamentals.
4. **Released:** deterministic SEC-backed 3y/5y scenario views and immutable
   advisory predictions.
5. **Released:** overlap-aware advisory base-case error, sign-match, and
   inclusion reporting by exact method/configuration/provider/horizon/
   evidence/revision group.
6. **Next:** accumulate prospective outcomes and add positive-return
   reliability, benchmark-relative, sector/regime, and portfolio reporting
   without weakening the released support gate.

European long-term forecasts remain out of scope until an equally defensible
point-in-time filing pipeline exists.
