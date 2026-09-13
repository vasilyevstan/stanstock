# Methodology

StanStock v1 uses deterministic, versioned rules. It does not fit a predictive
machine-learning model and does not permit generated text to alter a score,
scenario, risk class, or recommendation.

## Principles versus fixed policy

StanStock uses broad literature-supported principles: momentum and trend,
explicit risk treatment, base rates and shrinkage, the sustainable-growth
accounting identity, valuation mean reversion, and point-in-time evaluation.
Those principles do not validate StanStock's exact implementation constants.

Lookbacks, factor maps, the RSI score transform, horizon weights, state
buckets, caps, confidence and support formulas, fallback order, thresholds,
and publication/recommendation gates are fixed StanStock policy choices. They
must not be described as literature-standard, optimized, causal, or
statistically calibrated.

## Inputs

Supported inputs are grouped into:

- price returns, trend, momentum, range, volume, volatility, beta, and
  drawdown;
- filing-derived growth, margins, cash flow, profitability, leverage,
  liquidity, and valuation;
- regional/sector relative behavior and price-derived market regime;
- data completeness, freshness, liquidity, and source quality.

An indicator is unavailable when its minimum history or source inputs are
missing. Unsupported values are never approximated from unrelated fields.

The current US price-only v2 policy removes nominal share-price scale from
its technical and liquidity gates. It scores the MACD histogram only after
dividing by the latest positive close, and it measures liquidity as the
20-session mean of `close * volume`. A split-equivalent transformation of
price and share volume therefore cannot change either factor. The initial
dimensionless MACD range (`-0.02` to `+0.02`), dollar-volume factor range
($1 million to $50 million), and BUY floor ($5 million/day) are explicit,
versioned policy assumptions; they were not selected by optimizing later
outcomes. Historical v1 predictions retain their original absolute-MACD and
share-volume configuration and are never recomputed or pooled with v2
performance.

### Explicit research-only short v3

`us-price-baseline-v3` is prospective and available only by explicit config
selection. `default_us_scoring_config_path()`, scheduled/live US work, refresh
verification, and every current medium/long `enabled_scoring_versions` list
remain v2-only. V3 persists only its supported `short` decision prediction.
It does not automatically reissue history. Existing generic latest/serving,
opportunity, stock-detail, prediction-history, watchlist, latest-method
performance, contribution-planner, and sample-selection readers are
method-neutral: if a caller explicitly creates a v3 analysis, those readers
may display or consume it as the latest qualifying row. That is existing
reader behavior, not production activation.

V3 validates raw evidence before a lossy cast or filter. Asset and supplied
benchmark frames require unique, non-null dates represented as `Date`,
`Datetime` (normalized to calendar date), or strict ISO `YYYY-MM-DD`; finite
positive closes; and, when volume exists, finite nonnegative volume and finite
`close * volume`. Missing volume is insufficiency, while reported zero volume
is a valid numeric zero. Accepted rows are sorted ascending once, and the same
chronological asset frame feeds the existing indicator engine and unchanged
scenario engine. No exchange-session continuity is inferred.

For finite `x` and `low < high`, define:

```text
H(x; low, high) = 100 * min(1, max(0, (x - low) / (high - low)))
L(x; low, high) = 100 - H(x; low, high)
```

The strict v3 YAML is the only transform authority. It contains exactly these
16 factor maps:

| Component factor | Raw input | Transform |
|---|---|---|
| `momentum.return_20d` | 20-close return | `H(x; -0.10, 0.15)` |
| `momentum.return_63d` | 63-close return | `H(x; -0.20, 0.30)` |
| `momentum.return_126d` | 126-close return | `H(x; -0.30, 0.45)` |
| `momentum.sma_50` | close / 50-close SMA - 1 | `H(x; -0.10, 0.10)` |
| `momentum.sma_200` | close / 200-close SMA - 1 | `H(x; -0.15, 0.20)` |
| `momentum.rsi` | Cutler/SMA RSI(14) | `H(x; 30, 70)` |
| `momentum.macd` | MACD(12,26,9) histogram / latest positive close | `H(x; -0.02, 0.02)` |
| `momentum.52w` | latest position in 252-close range; flat = 0.5 | `H(x; 0.15, 0.95)` |
| `risk.annualized_volatility` | common-window sample SD × sqrt(252) | `L(x; 0.12, 0.65)` |
| `risk.downside_volatility` | common-window RMS of `min(return, 0)` × sqrt(252) | `L(x; 0.08, 0.50)` |
| `risk.max_drawdown` | common-window minimum drawdown | `H(x; -0.60, -0.05)` |
| `risk.abnormal_volume` | latest volume / 20-observation mean | piecewise map below |
| `risk.avg_volume` | 20-observation mean USD `close * volume` | `H(x; 1m, 50m)` |
| `market.relative_20d` | asset return - benchmark return | `H(x; -0.08, 0.08)` |
| `market.relative_63d` | asset return - benchmark return | `H(x; -0.15, 0.15)` |
| `market.relative_252d` | asset return - benchmark return | `H(x; -0.25, 0.25)` |

Counts are exactly 8 momentum/technical, 5 risk/liquidity, and 3 market
factors. Available factors are averaged within components. Beta is not a
factor and contributes nothing to a component, horizon score, overall score,
or conviction.

RSI uses the existing Cutler/SMA convention over the latest 14
close-to-close changes:

```text
G = mean(max(change, 0))
D = mean(max(-change, 0))
RSI = 50 when G = D = 0
RSI = 100 when D = 0 < G
RSI = 0 when G = 0 < D
otherwise RSI = 100 - 100 / (1 + G / D)
```

V3 then applies `H(RSI; 30, 70)`, so `30/50/70` maps to `0/50/100`.
This affine map is bounded, continuous, nondecreasing, and
2.5-Lipschitz. It is not Wilder smoothing and is not a literature-standard
RSI score transform.

For positive finite abnormal-volume ratio `x`, v3 preserves v2's
`clamp(100 - 20 * abs(x - 1))`. At `x <= 0` the score is exactly zero.
Consequently `A(0) = 0` while the right-hand limit at zero is `80`; this
intentional discontinuity must not be described as continuous.

### Common-window risk and recommendation independence

V3 intersects the clean asset and benchmark by observed date, requires their
latest eligible dates to match, selects exactly the latest 252 common closes,
and computes exactly 251 aligned simple-return pairs. It neither pads, fills,
interpolates, deduplicates, nor reaches backward around a bad row. The number
252 is also the fixed annualization constant. Fewer than 252 common closes or
a latest-date mismatch is explicit common-risk insufficiency. The separate
252-session relative-return factor needs 253 overlapping closes.

The four mandatory YAML-authoritative penalties are:

```text
Pvol  = H(annualized volatility; 0.12, 0.65)
Pdown = H(downside volatility; 0.08, 0.50)
Pdraw = H(abs(min(max drawdown, 0)); 0.05, 0.60)
Pbeta = H(abs(beta); 0, 2)
risk  = (Pvol + Pdown + Pdraw + Pbeta) / 4
```

Beta is sample covariance of the 251 asset/benchmark return pairs divided by
sample benchmark-return variance. Absolute beta is symmetric: beta `0`, `1`,
and absolute beta `>= 2` produce beta penalties `0`, `50`, and `100`.
Closeness to beta one is not quality or alpha. Zero benchmark variance
withholds beta; volatility, downside, and drawdown can remain numeric, but the
complete risk score is null whenever any one penalty is unavailable. Valid
zero return, downside deviation, drawdown, volume, or beta remains numeric
zero.

BUY retains the v2 score (`>= 72`), complete risk (`<= 55`), confidence
(`>= 45`), 20-session dollar-turnover (`>= $5m`), complete short scenario,
and bear-downside (`>= -8%`) gates. Missing risk, liquidity, or scenario
blocks BUY. AVOID is still independently triggered by score `<= 38`, numeric
risk `>= 82`, or confidence `<= 15`; therefore insufficiency does not force
HOLD. Empirical short scenarios are calculated independently and can remain
numeric while composite risk is null.

### Price scale and source boundary

A compatible split-equivalent transform multiplies all OHLC/close values by
finite `k > 0` and divides share volume by `k`. Returns, close/SMA ratios,
RSI, normalized MACD, range position, abnormal-volume ratio, USD turnover,
the four common-risk metrics, factors, component/horizon/overall scores,
coverage, confidence, scenarios, every recommendation gate, and the final
recommendation remain invariant. Raw OHLC, SMA, EMA, MACD, and ATR values
scale by `k`; raw share volume scales by `1/k`.

Price multiplied by `k` while volume is fixed is not split-equivalent:
20-session USD turnover becomes exactly `k` times larger. The configured
liquidity factor, risk/liquidity component, overall score, `$5m` BUY gate,
and recommendation may therefore change. Composite risk does not change when
its four normalized inputs do not change. No override suppresses this normal
dollar-liquidity effect. Current USD price bands and raw price differences
remain display/filter/execution metadata, never direct score, risk,
confidence, or recommendation inputs. Split-only price provenance does not
prove provider-reported share volume is split-compatible.

For persisted v3 work, each listing and each non-null requested benchmark is
selected once with `AsOfData.latest_asset`. The exact returned `DataAsset` is
then physically read once, SHA-256 checked once, clipped through the target,
and used together with its UUID, checksum, provider, subject,
`retrieved_at`, and `available_at` provenance. There is no separate
provenance selection. The source matrix per listing is:

| Benchmark boundary | Selection/read count | Result |
|---|---:|---|
| omitted (`None`) | `0/0` | valid common-risk insufficiency |
| requested and eligible | `1/1` | exact source is used and persisted |
| requested but unavailable | `1 failed/0` | selection error propagates; rollback |
| selected but corrupt | `1/1 attempted` | checksum error propagates; rollback |

Listing sources use the corresponding required `1/1`, `1 failed/0`, or
`1/1 attempted` behavior. Snapshot counts repeat per listing (`N/N` for an
eligible requested benchmark); no cache, fallback, substitute, reselection,
or downgrade is used. V1/v2 keep their historical convenience-reader path.
A source failure leaves no invocation-owned run, analysis, prediction,
manifest, panel, row, or file; pre-existing immutable source rows/files remain
untouched. V3 refuses non-USD listings rather than mixing currency or adding
implicit FX.

Actual generation time, logical target date, historical data cutoff, and
source retrieval/availability timestamps remain distinct. An exceptional
observed v3 issuance is a direct
`analyze_snapshot(..., issued_on_time=True, ...)` service call—not
`manage.py analyze`, which is research-grade. For that explicit observed-v3
request, the service enforces the exact v3 config version and effective config hash,
`provider="twelve_data"`, SPY, and a raw lowercase 40-hex
`STANSTOCK_CODE_REVISION` equal to the checkout's exact clean committed HEAD.
The caller still owns reviewed-production-universe selection and independent
pre-invocation proof of the next-session-open deadline and every source's
cutoff safety. Existing service deadline and source-cutoff checks remain
fail-closed. An unsafe explicit observed request raises rather than silently
downgrading.

The methodology is broadly contextualized by Wilder (1978), whose
Wilder-smoothed RSI is expressly not used here; Jegadeesh and Titman (1993)
and Moskowitz, Ooi, and Pedersen (2012) on momentum/trend; Sharpe (1964) on
beta as market sensitivity; and Amihud (2002) on separate liquidity
treatment. Dollar turnover is not the Amihud measure. Those works do not
validate the exact v3 windows, maps, weights, thresholds, risk classes,
confidence rules, recommendation gates, causal interpretation, profitability,
or alpha.

## Horizon weights

The initial hypothesis weights are:

| Component | 1-10 trading days | 6-12 months | 3+ years |
|---|---:|---:|---:|
| Fundamental quality | 5 | 20 | 30 |
| Growth and reinvestment | 5 | 20 | 25 |
| Valuation | 5 | 20 | 20 |
| Momentum and technical trend | 40 | 15 | 5 |
| Risk and liquidity | 25 | 15 | 15 |
| Market and sector regime | 20 | 10 | 5 |

Horizon weights and some bounds, thresholds, coverage requirements, and
scenario parameters live in versioned YAML. YAML/config policy is bound by the
stored configuration hash; code-defined transforms are identified by the
stored `code_revision`. Scheduled observed production automatically binds an
exact clean commit SHA. Demo and direct research can record `working-tree`
unless an exact committed revision is explicitly supplied. Not every threshold
is configurable in YAML. These weights are starting assumptions; they are not
optimized against the period later used to report performance.

## Fixed factor maps and missingness

Active scoring does not cross-sectionally rank or winsorize factors. Each
available raw value is independently transformed to 0-100 by a fixed affine
or piecewise policy map and clamped to that range. Available factor scores are
then averaged within each component, and versioned horizon weights combine the
available component averages. The completed overall score also applies the
configured missingness and freshness penalties.

The opportunities page orders completed analyses by that completed overall
score. This presentation ranking is downstream of factor scoring; it is not a
factor-normalization step.

`rsi_14` is Cutler/SMA-style RSI: it uses simple averages of the gains and
losses over the latest 14 close-to-close changes, not Wilder smoothing. The
subsequent RSI-to-score piecewise transform is a fixed StanStock heuristic,
not a literature-standard RSI transform.

Missingness reduces coverage and confidence. A configured minimum coverage can
block a BUY result entirely. A missing value is never converted to zero.
Source filing concepts are mapped to canonical calculation names without
changing the stored source concept. When multiple vintages restate the same
reporting period, the latest eligible vintage replaces that period before
growth is calculated against the prior distinct period.

## Risk

Risk combines supported observations such as:

- annualized and downside volatility;
- beta and maximum drawdown;
- liquidity and stale-trading risk;
- leverage, interest coverage, and balance-sheet resilience;
- earnings/free-cash-flow stability;
- valuation dispersion and sector cyclicality;
- data completeness and freshness.

Versioned thresholds map supported 0-100 scores to LOW, MEDIUM, HIGH, or VERY
HIGH. When no supported risk input exists, risk is `INSUFFICIENT EVIDENCE`,
the numeric score is null, and BUY is blocked rather than inventing a
conservative-looking number.

## Recommendations

BUY, HOLD, and AVOID are deterministic gates over:

- overall and component scores;
- downside in the relevant scenario;
- risk class;
- evidence sufficiency and confidence;
- liquidity and blocking quality flags.

Explanations are selected from the actual factors that crossed documented
thresholds. They are not free-form rationalizations.

## Highlighted opportunities

The opportunity highlight is a separate, versioned presentation policy in
`config/opportunities/great-opportunity-v2.yml`; it does not alter the
underlying recommendation. A highlighted result must be BUY, clear the
policy's score and confidence thresholds, remain LOW or MEDIUM risk, and have
a positive base case for the analysis mode's supported horizon. V2 also
prevents the Under $10 speculative watchlist from becoming a highlighted
new-allocation idea. The historical v1 policy remains unchanged.

Full analyses use the label `Great opportunity` and require fundamentals.
The US price-only baseline instead uses `Strong short-term setup`, checks only
its supported short horizon, and remains visibly identified as price-only.
The policy version is shown on stock detail so a later policy revision cannot
silently masquerade as the original rule.

## Current USD price bands

The opportunities page classifies each latest valid persisted USD close into
four neutral, non-overlapping affordability bands:

- `0 < price < $10`: `Under $10 - speculative watchlist`;
- `$10 <= price < $50`;
- `$50 <= price < $300`;
- `$300+`.

Each band displays the close's market-session date. Price bands are filters
and execution context only: they do not enter factor arithmetic, score,
confidence, valuation, or recommendation. The `$300+` band is not a quality
penalty.

Under $10 remains research-only with a 0% new-allocation cap. It is excluded
from current opportunity promotion. Newly constructed sample portfolios apply
the same boundaries to the immutable decision-run reference close, never a
later mutable market row, while existing holdings and frozen historical
portfolios remain visible. Its long-horizon forecast is explicitly unavailable
pending joint Under-$10 review and candidate-specific eligibility. Released
foundations are point-in-time SEC facts with adverse-versus-missing branch
behavior; the long-v2 diluted-share/per-share continuity assessment with
withholding (which does not verify post-period corporate actions); and the
deterministic 3-year/5-year formula engine with missing-input withholding.
These foundations do not establish candidate qualification. The only
still-unreleased activation control is a verified split/reverse-split event
source. Previously issued immutable long-horizon ledger evidence remains
visible, with its original horizon and evidence role preserved, and is labeled
with the current activation context. The configured stock universe is not
expanded merely to populate a price band.

## Under-$10 shadow diagnostics (`us-under10-shadow-v1`)

The solvency/cash-runway and Under-$10 dollar-liquidity capabilities are
released as **shadow diagnostics**. They are recorded, never acted on:
`UNDER10_ACTIVATED` is `false`, `activation_eligible` is `false` in every
branch, and the assessment changes no score, confidence, recommendation,
gate, scenario, forecast, prediction, outcome, performance denominator,
opportunity qualification, sample basket, or contribution plan. New allocation
stays 0%.

An assessment is written as an additive `data_quality["under10_assessment"]`
key on a **newly created** analysis whose *decision-run* USD reference close --
the same six-decimal close `StockAnalysis.current_price` persists -- falls in
the Under-$10 band. Nothing is backfilled and no historical row is rewritten,
so an absent key means *not assessed*, never *assessed and failed*. The policy
constants, required concepts, selection and availability rules, basis
requirements, refusal behavior, serialization rules, listing-identity binding,
and the reviewed SEC fundamentals configuration's own effective hash are
hashed into one pinned `policy_hash`; `assessment_hash` is a recomputation
checksum over the complete payload. Neither hash is tamper protection: both
are corruption-detection checksums for an otherwise-trusted canonical
payload, never proof storage was never modified.

The stock-detail reader renders a stored assessment only after binding it
exactly to its parent decision: the permanent `Listing.id` recorded in
`evaluated_for.listing_id`, the parent `AnalysisRun`'s own exact target date
and data cutoff, its persisted decision-run reference close and currency
(cross-checked against the immutable original decision predictions), and the
exact immutable price-asset UUID *and* content checksum recorded alongside
it. A matching checksum alone -- even across every one of those fields but
one -- never substitutes for this: two genuine, unrelated candidates can
otherwise share the same target date, cutoff, reference close, and currency,
and copying an entire genuine `data_quality` blob (the payload plus its own
internal anchors) from one analysis onto another moves every other internal
anchor along with it, so only the permanent listing id closes that
transplant.

`StockAnalysis.data_quality` is mutable storage, so the reader independently
replays the accepted builder before displaying the stored values. It resolves
the original decision-prediction cohort, reads the exact immutable price asset
named by that cohort through the original target-date cutoff, and reselects
the exact cutoff-qualified SEC fact lineage. The stored and replayed solvency
and liquidity blocks must be canonically identical. Missing, unreadable, or
mismatched evidence fails closed; replayed values are never substituted into
the response or written back.

**Solvency and obligation.** Facts are read through `AsOfData` at the run's
`data_cutoff`. Only fixed canonical-concept rows whose fact provider and source
asset provider both identify SEC enter the SEC calculation, assessed lineage,
or on-time SEC asset cutoff check; foreign and provider-mismatched rows are not
SEC evidence. Same-accession corrections whose timing is not provable at that
cutoff are then deferred, leaving the proven prior vintage in place. The
canonical SEC series is built on the frozen legacy TTM path with no alias
candidate surface. Five balance-sheet inputs (cash, short-term debt, current
long-term debt, current assets, current liabilities) must share exactly one
`period_end`; free cash flow comes from compatible TTM evidence, otherwise the
latest compatible annual value, derived as operating cash flow minus absolute
capital expenditure. Every selected date must satisfy `0 <= target_date -
period_end <= 200` days. A missing debt component is missing, never zero, and a
non-positive current-liabilities figure withholds the ratio. The state is one
of four, in strict first-match order:

1. `insufficient_evidence` -- any required input missing, unusable,
   incompatible, stale, future-dated, or not cutoff-safe. Never read as
   adverse.
2. `adverse_near_term_obligation` -- near-term debt above cash *and* either
   current assets below current liabilities or a negative free cash flow with
   under four quarters of cash runway.
3. `elevated_obligation_risk` -- any one of negative free cash flow,
   near-term debt above cash, or current assets below current liabilities.
4. `no_adverse_evidence_observed` -- every remaining complete-input case.

All comparisons use exact Decimal operands; the reported current ratio and
runway are quantized separately to four places and never participate in a
decision. The four-quarter boundary is applied as a cross-multiplication, so a
displayed `4.0000` cannot override an exact below-four classification. A
non-negative free cash flow gives runway status `not_applicable_positive_fcf`
with a null value -- not zero and not infinity. Runway is reportable from cash
and free cash flow alone even when another obligation input is missing.

**Dollar liquidity.** The median of `close x volume` over the latest 252
*observed* sessions, computed from the already cutoff-clipped price frame. The
raw window is validated before preparation, so an unparseable, null,
non-finite, or non-positive close (or a negative volume) is an explicit
invalid-input refusal rather than "insufficient history"; duplicate session
dates are refused outright; calendar gaps are never padded and an invalid row
is never replaced by reaching further back. Zero reported volume is valid data,
and a computed zero median is displayed as zero. The diagnostic requires the
same price asset the analysis manifest recorded, with observed `interval=1day`,
`adjustment=splits`, `return_definition=split_adjusted_price_return`, and
`currency=USD` metadata, and a last session no later than the target date and
no more than seven calendar days old. There is **no liquidity threshold**: the
figure carries no pass/fail conclusion. Split-only adjustment is proven for
prices but not for the provider's reported volume, so the basis stays labeled
`provider_reported_unverified_split_basis` even when the number is computed.

**Split verification.** No reviewed corporate-actions source is integrated for
any provider, so `split_verification.status` is always `unavailable`. A
recorded Twelve Data Basic plan reports `provider_plan_not_entitled`; every
other provider or plan -- absent, empty, malformed, or unknown -- reports
`no_reviewed_corporate_actions_source`. A split is never inferred from adjusted
prices, share-count discontinuities, or SEC facts, and no branch can return a
verified state. This is why candidate activation cannot pass.

An on-time issuance additionally proves that each referenced SEC asset
satisfies the same `available_at`/`retrieved_at` cutoff rule the core source
manifest uses; a violation withholds the diagnostic with
`evidence_not_cutoff_safe` rather than failing the run. A research-grade
reconstruction may read later-retrieved evidence, while fact availability
still has to qualify at the historical cutoff.

## ETF evidence boundary

Common stocks and ADRs are the only security types accepted by the stock
analysis, prediction, opportunity, and sample-stock services. An ETF sent
through those paths fails explicitly instead of receiving a stock rating.

SPY is the first and only enabled investable ETF. The daily US workflow uses
the already-required SPY benchmark response for both benchmark evidence and
the ETF's latest market row, so ETF support consumes no additional provider
credit and creates no stock-universe membership. Its ETF page reports
split-adjusted cumulative price return, annualized close-to-close volatility,
and peak-to-trough maximum drawdown from at most 253 persisted closes ending
at the displayed market session, even if the immutable asset contains later
rows. The source asset must explicitly prove daily, split-only,
dividend-excluding price-return metadata; incompatible or missing provenance
is rejected rather than defaulted. If Twelve Data omits its optional MIC
field, the normalized asset records that ARCX came from StanStock's reviewed
SPY identity rather than pretending it came from the provider. Dividends are
not included, so none of these values is a total return. SPY has no
fundamental score, BUY/HOLD/AVOID recommendation, opportunity status, or stock
forecast.

## Scenarios

- **1-10 trading days:** empirical/rule ranges driven primarily by trend,
  momentum, volume, volatility, liquidity, and market regime.
- **6 months and 12 months:** separate price-only advisory ranges built from
  fixed-epoch, non-overlapping 126- and 252-session cohorts. Matching uses
  SPY-relative 12-month momentum, 52-week drawdown, trailing volatility, and
  SPY trend/volatility regimes. Each cohort receives equal aggregate weight
  in both the matched and unconditional distributions. The matched and
  unconditional p20, p50, and p80 estimates are each shrinkage blended. The
  published base is the matched/unconditional blended p50; bear and bull are
  the corresponding blended p20 and p80.
- **3 years and 5 years:** separate deterministic advisory cases using
  point-in-time SEC facts, current point-in-time SIC peers, and the exact
  split-adjusted Twelve Data price asset. Positive compatible FCF/share takes
  priority. EPS/share is permitted only when FCF evidence is genuinely
  unavailable; negative, inconsistent, or incomplete FCF cannot trigger a
  more favorable fallback.

For medium forecasts, bear-to-bull is a nominal central 60% analog-return
range. It is not a calibrated prediction interval, credible interval, or
confidence interval and provides no coverage guarantee.

The medium-horizon positive-return estimate is omitted with an insufficiency
reason until non-overlapping cohort support, listing diversity, calendar span,
matched market-regime breadth, and the frozen probability-publication gate all
pass. That gate compares base-case mean absolute error with both the
unconditional baseline and a SPY-relative decomposition baseline (the
historical SPY median for the matching market regime plus the historical
excess-return median for the matching relative-momentum bucket), and applies
the configured absolute Brier-score threshold.

The persisted `empirical_calibrated` status means only that those
support/diversity, base-case error, and Brier-threshold checks passed. It does
not establish calibrated probabilities or interval coverage. Presentation
therefore labels it `Probability gate passed — not calibrated`;
`empirical_range_only` is labeled
`Analog range only — probability withheld`. A published positive-return value
is a shrinkage-weighted analog estimate, not a calibrated probability claim.

Because the narrowest fallback levels explicitly condition on market regime,
they may yield a useful range while still failing the matched-regime breadth
gate for the estimate. StanStock does not switch to a broader fallback merely
to publish that value.

### Explicit research-only medium v2

`us-price-medium-v1` is frozen and remains the default and scheduled medium
method. `us-price-medium-v2` is an explicit research-only service path. It
requires literal `issued_on_time=False`, a research-grade current-universe
snapshot, exact `us-price-baseline-v2`, SPY, and stock-research-eligible US/USD
listings (common stock or depositary receipt).

For a support set `S`, each of its `K` origin cohorts contributes total mass
`1/K`; each row in cohort `c` receives `1/(K*n_c)`. The matched and
unconditional CDFs are normalized separately. For a conditional fallback:

```text
w = matched_cohorts / (matched_cohorts + 4)
F_v2(x) = w * F_matched(x) + (1 - w) * F_unconditional(x)
```

The unconditional fallback uses `w=0`. Quantiles are the first support value
whose cumulative mass is at least `q`, without interpolation. Bear/base/bull
are p20/p50/p80 from this one mixture, and strict positive-return probability
is `1 - F_v2(0)`; zero return is non-positive. A finite support return below
`-1.0` aborts the run before panel serialization or persistence, with no
floor, clip, tolerance, or omission. Exactly `-1.0` is valid.

At historical origin `o`, training includes only eligible prior rows with
`label_end_date <= o`; equality is admitted and a later-ending label is
excluded from model and reference support and every gate. Range/base/interval
evidence uses every prior-only range forecast at an origin. Probability
evidence uses only the subset whose prior-only matched support also passes its
cohort, listing, span, and regime floors. Metrics average listings within each
origin and then average origins equally.

The probability reference is the origin's prior-only unconditional event
rate. Raw Brier skill is `1 - BS_model / BS_reference`; a zero reference score
has null skill, and probability publication requires every current support
floor plus raw `BSS > 0` with at least four evaluable origins. Base p50 MAE is
reported independently against prior-only unconditional p50 and the existing
SPY-relative baseline. The nominal central-60% p20-p80 interval reports
inclusive coverage, strict lower/upper miss rates, date-equal mean width, and
the alpha-0.40 interval score:

```text
(upper - lower)
+ 5 * (lower - actual) when actual < lower
+ 5 * (actual - upper) when actual > upper
```

These results use a current-universe reconstruction and are
survivorship-biased. They are descriptive research evidence, not calibrated
probabilities, statistical significance, profitability, alpha, or observed
live skill. V2 does not report bootstrap intervals, CRPS, reliability bins,
median width, or a coverage publication gate.

The medium panel stores 50/200-session trend, downside volatility, and dollar
liquidity for eligibility and explanation, but those values do not add hidden
matching dimensions. All panel inputs are capped at their historical anchor,
and a forward label is present only when its complete outcome ends on or
before the current forecast target.

The long engine requires at least three contiguous annual per-share periods,
compatible TTM metric and diluted-share periods, reported diluted-EPS
share-basis checks for every selected annual period, TTM diluted shares
within 15% of the latest overlapping annual share basis, a bounded GAAP
accrual tax proxy, beginning and ending invested capital using identical
canonical and source concept definitions, and a same-family SIC peer set
meeting frozen floors. The default `us-sec-long-v2` configuration additionally checks
diluted-share basis continuity between every adjacent pair of selected
annual periods (same 15% tolerance); the frozen `us-sec-long-v1`
configuration never evaluates that adjacent check and remains reproducible
exactly as originally released.

**Scope note.** The SEC correction-availability integrity fix that this work
also delivers is *active* in the shipped default `us-sec-long-v2`
configuration: ingestion binds every same-accession correction to the
retrieval that carried it, and every reader benefits. Only the long-v3
*reader* below -- alias selection, joint pair selection, and read-time
correction resolution -- is inactive. The change as a whole is therefore not
wholly inactive.

`us-sec-long-v3` is a prospective, evidence-selection-only version that is
**not the default and not approved for activation**. It keeps every long-v2
formula weight, bound, cap, fade path, multiple reversion, peer floor,
metric-family rule, freshness limit, tax proxy, scenario constant,
probability withholding, return basis, and unsupported-SIC policy
byte-identical, and changes only *which* already-persisted SEC observations
the same arithmetic reads:

- **Newest-quarter-anchored homogeneous TTM alias selection.** For each
  canonical TTM concept, long-v3 finds the newest eligible quarter end,
  ranks the observations at exactly that quarter under the existing
  availability/revision/source-priority/accession order, and then requires
  the winning source alias to supply four contiguous compatible quarters
  spanning 350-380 days. Aliases are never stitched across quarters, a stale
  but complete alias never displaces a newer restated newest-quarter
  observation, and there is no annual current-period fallback. Annual
  history selection stays on the frozen legacy path, as does every
  long-v1/long-v2 and generic consumer.

  A quarter derived from the difference of two year-to-date filings is
  ranked by the single *controlling* filing that gates its knowability, not
  by taking the maximum availability from one dependency and the maximum
  revision from another. That synthesis would report a vintage that was
  never filed and could hand the anchor to the wrong alias. The chosen
  alias, its controlling source fact, and the per-quarter lineage of the
  selected window are recorded in the calculation provenance.
- **Deterministic joint compatible invested-capital pair selection.**
  Instead of choosing the beginning and ending balance-sheet snapshots
  independently, long-v3 searches every candidate pair inside the unchanged
  +/-7-day tolerance and accepts only pairs whose debt method, debt
  components, and canonical/source concept bases match exactly. Pairs are
  ranked purely on evidence: combined and per-side target-date distance,
  then eligible-evidence recency, then declared alias priority, canonical
  source basis, and stable date/fact-id tie-breaks -- never on the resulting
  ROIC, reinvestment, growth, forecast, or scenario favorability. When no
  compatible pair exists the forecast is withheld with an explicit reason.

  Because the normalized instant series keeps only one winning alias per
  canonical concept and period, long-v3 additionally reads a candidate
  surface that retains the latest eligible vintage per `(canonical concept,
  source alias, period identity)` and enumerates the permitted same-date
  source-basis combinations before pairing. A reported long-term debt
  roll-up still excludes the component basis at that date, so no component
  is double counted, and superseded revisions are still discarded.

Every invested-capital pair search is persisted as an assessment -- missing
beginning side, missing ending side, zero compatible pairs, a pair selected
and then withheld for an unrelated reason, and success alike -- with the
targets, candidate snapshots and their source bases, compatible-pair count,
status, and rejection reason. Rejected evidence is recorded as *assessed*,
never as verified.

Within one alias, a directly reported quarter and a year-to-date-derived
quarter for the same period end -- and two direct observations whose period
identities differ but whose ends coincide -- are all retained and resolved by
the full rank of one real controlling filing, not by availability alone. No
long-v3 ordering or tie-break reads a generated row identifier: alias
ordering, latest-revision selection, and pair ranking fall through to the
persisted observation identity instead. A balance-sheet date whose instant
observations do not all carry the canonical `instant::<date>` identity, and a
date whose permitted same-date source-basis combinations exceed the reviewed
ceiling declared by the configuration (counted from per-axis alias counts before any product is
built, with every alias, fact, and axis multiplier behind that count
recorded), each withhold that one listing explicitly while the rest of the
run continues.

- **Prospective deferral of unproven corrections.** Long-v3 resolves each
  same-accession correction against the observation that proves it, not
  against a recorded availability that a legacy ingestion may have backdated
  to the original filing acceptance. A correction whose timing is not proven
  at the requested cutoff is withheld from the series, the revision it
  superseded -- which *is* proven there -- stands in its place, and the
  withheld row is recorded under `deferred_unproven_corrections` with its
  recorded availability, its resolved availability, and the reason. A legacy
  revision with no actual observation boundary -- a reversion whose bytes
  deduplicated onto an earlier revision's asset, or a revision from an asset
  retrieved before the revision it supersedes became knowable -- is deferred
  at every boundary. A chain-ordering lower bound is reported as assessed
  context and never admits a row. Nothing is
  rewritten: resolution is a read-time decision over immutable rows, and no
  persisted fact or prediction is mutated. See `docs/point-in-time.md` for
  the three clocks involved.

  This policy is configuration-gated by its own declared, hashed capability
  `proven_observation_correction_availability`, so adopting alias or joint
  pair selection never silently acquires it. Frozen long-v1/long-v2 declare
  no such key, read recorded availability exactly as released, and defer
  nothing; the offline evidence audit applies the identical gate, so a frozen
  version is never reported against a selection it would not make. The
  same-date combination ceiling is likewise a declared, hashed config value
  (`maximum_same_date_source_combinations: 256`) accepted only at the one
  reviewed number -- never tuned, defaulted, or truncated.

Every fact any of these assessments referenced -- including rejected,
unpaired, and later-withheld candidates, the facts responsible for a refused
combination space, every trailing-twelve-month dependency, every deferred
correction, and every fact an alias-tail assessment examined and rejected
before any quarter candidate could be constructed (an annual-only alternate
alias, or a year-to-date pair whose derivation was refused) -- has its source
and filing-evidence assets in the immutable forecast `source_assets`
manifest. The payload separates that evidence into explicit, disjoint lists:
the selected formula inputs described in `input_facts`, the assessed
candidates that were read but not selected, and the union the manifest closes
over. A failing path -- no compatible pair, a missing side, a refused
boundary, or an unusable metric branch -- selects nothing, so its candidates
appear only as assessed evidence and never as verified inputs.

**No eligibility improvement is claimed.** Long-v3 has not been measured
against a historical panel, and stricter alias homogeneity can withhold a
forecast that long-v2 produced. A single read-only cutoff-safe replay of one
local 97-listing run moved 3y/5y eligibility from 2 to 5 listings with no
losses (see `docs/forecast-roadmap.md`), but one snapshot on one target date
says nothing about forecast accuracy and is not an activation basis. A
cutoff-safe replay across a historical panel must establish whether the
change is a net benefit before any activation decision.

Long-v3 also binds the SEC fundamentals configuration identity it selects
against (`us-sec-fundamentals-v1`, itself unchanged by this work).
`us-sec-long-v2` remains the default configuration; long-v1 and long-v2
config bytes, effective hashes, behavior, reason strings, payloads, and
already-issued immutable predictions are unchanged, and that is proven by a
differential test that executes the pre-change source against the same
deterministic fixture and compares complete successful and withheld
payloads.

### Explicit research-only long-v4

`us-sec-long-v4` is an unactivated schema-2 research method. It is admitted
only through the exact tracked config with an explicit true long request,
literal `issued_on_time=False`, an authoritative research snapshot, exact
`us-price-baseline-v2` scoring identity, Twelve Data prices, SEC
fundamentals-v1, the byte/effective/source-pinned SEC CIK-v1 configuration and
raw ticker/exchange mapping, and active US/USD security identity. Every
cohort listing must match one exact config/raw row and the reviewed
`Nasdaq -> XNAS` or `NYSE -> XNYS` rule. Common stocks alone can enter the
metric and peer calculations. A correctly mapped depositary receipt receives
one explicit withheld 3y/5y pair after admission and never enters a peer
cohort; no ADS ratio is inferred.

V4 uses reported GAAP evidence without NOPAT, a tax proxy, invested capital,
reinvestment, ROIC/ROIIC, sustainable growth, R&D capitalization, or a
return-on-new-capital/project-IRR claim. FCF is `operating cash flow -
abs(capex)`. For each assessed entity V4 derives configured duration
observations from every Companyfacts asset underlying its cutoff-visible
facts plus the latest decision-visible Companyfacts observation/source. It
records raw authority as `absent`, `present_complete`, or
`present_normalization_incomplete`. Net income is eligible only for
`absent`; present but missing, incompatible, stale, nonpositive, or otherwise
failed FCF blocks fallback, while incomplete normalized closure records
`fcf_normalization_incomplete` for that entity. Weighted-average diluted
shares are an accounting-period denominator, not a count of issued shares.

For each target or peer the same nonrecursive assessor requires a newest-
quarter-anchored homogeneous TTM metric and shares, no more than 200 days old,
and the latest four distinct contiguous annual period identities. Each
annual tuple spans 350-380 days, uses matching entity/share periods and
compatible units, derivations, and source concepts, and has finite strictly
positive entity values, diluted shares, derived per-share values, and prices.
There is no nominal per-share floor. Reported diluted EPS is reconciled to
net income/shares within an inclusive 15%, using exactly the larger absolute
EPS as the relative-difference denominator (and zero difference when both
values are exactly zero). Adjacent annual share bases and
the TTM/latest-annual basis must also remain within 15%. Selecting periods
before applying these tests prevents an older favorable tail from replacing
new adverse evidence.

Let \(M_i\) be the four annual entity metrics and \(S_i\) the matching diluted
shares. V4 stores all three raw changes and computes:

```text
G_target = median(clamp(M_i / M_(i-1) - 1, -0.20, 0.25))
D_base   = clamp(max(0, median(S_i / S_(i-1) - 1)), 0, 0.15)
G_peer   = clamp(median(admitted_peer_G_target), -0.15, 0.25)
```

The peer cohort is locked *before* core evidence assessment at the first
cutoff-safe SIC-4/3/2 cohort meeting identity floors 3/5/8. It excludes the
target company, depositary receipts, inactive/non-US/non-USD listings, and
deduplicates companies by ticker then permanent listing UUID. It never
widens after evidence failures. Every admitted peer uses the target's metric
family; unfavorable but valid evidence remains in the medians.
Every attempted lock records the target classification, each examined
SIC-4/3/2 prefix/floor/candidate set, and either the selected level or an
explicit no-floor result. No-floor evidence retains every examined candidate
and classification source.

For scenario \(s\):

```text
T_s = clamp(G_target + delta_s, -0.20, 0.25)
P_s = clamp(G_peer   + delta_s, -0.15, 0.25)
E_s = clamp(0.5*T_s + 0.5*P_s, -0.15, 0.25)
D_s = clamp(D_base * dilution_multiplier_s, 0, 0.15)
g_per_share,t = (1 + g_entity,t) / (1 + D_s) - 1
```

Bear/base/bull use growth deltas `-.04/0/.03`, dilution multipliers
`1.25/1/.75`, and peer-multiple multipliers `.80/1/1.15`. The single
five-year entity path fades by `[.80,.60,.40,.20,0]` toward 2.5%; geometric
multiple reversion progresses by `[.14,.28,.42,.56,.70]`. Levels and factors
must independently produce the same return. The raw current multiple remains
the return denominator while its capped value is only the reversion anchor.
The stored 3y and 5y views are exact years 3 and 5 of one path.

The V4 policy pins the canonical SEC fundamentals file bytes/effective hash,
the canonical SEC CIK file bytes/effective hash, and the source hash of the
raw ticker/exchange mapping. Before either immutable prediction is written,
the service locks and reloads the complete eligible snapshot membership,
reselects cutoff-safe classifications, recomputes the peer-lock trace, and
replays the complete cohort.

The schema-2 evidence manifest follows one deterministic traversal: the
mapping asset; each assessed owner's Companyfacts, current submissions, and
filename-sorted submissions history in cohort order; each fact's
source/filing/context triple; classification assets; then each cohort price's
normalized Parquet and linked raw payload, deduplicating only global first
occurrences. All unique files are checksum-read.

Outcome authentication first checks the persisted run, snapshot, cohort,
mapping, prices, assessed owners, facts, classifications, peer lock, and both
ordered manifests against database authority. Only after those checks pass
does it checksum-read and re-derive the retained raw SEC closure for the target
and actually assessed peers. A missing, unreadable, or changed Companyfacts,
current-submissions, or submissions-history file yields the existing
`identity_mismatch` unresolved outcome before any stock or benchmark lookup.
This outcome replay performs no provider request, forecast rebuild, or reread
of the issuance-price file.

Each canonical price binds listing ID, provider symbol, exchange MIC,
currency, session date, and both asset IDs/checksums while retaining two
values from the exact physical Parquet cell. `P_val` is the finite positive,
unrounded `Decimal(str(close))` and drives the exact per-share/multiple gate
and scenario level returns. `P_ledger = canonical_long_v4_price(P_val)` is the
six-decimal value used by `StockAnalysis.current_price`,
`Prediction.price_at_prediction`, and ledger identity. Both roles are
persisted and reconstructed from physical bytes; substituting the ledger
value into valuation arithmetic or V4 outcome accounting fails replay.

The shared outcome evaluator authenticates a V4 prediction's complete
calculation, target, target-price, valuation-source, and ledger identities
before loading evaluation prices. It then requires an exact target-date close
whose `Decimal(str(close))` equals `P_val`; unlike legacy outcomes, it never
uses a prior close for this baseline check. The terminal close still matures
on the ordinary horizon session. Under decimal precision 64, V4 computes
`terminal / P_val - 1` and quantizes exactly once to `0.0001` with
`ROUND_HALF_EVEN`; that stored decimal drives direction, inclusive interval
coverage, base error, and signed error. Benchmark calculation is unchanged.

Confidence is not estimated. Both successful and withheld rows store `0.00`
only as the schema-declared unavailable sentinel; successful status is
`not_estimated_uncalibrated`, withheld status is
`not_estimated_insufficient`, and positive-return probability is null. The
UI renders those words rather than `0%`.

Literature supplies context and cautions, not fitted coefficients:
Damodaran on valuation and growth; Nissim-Penman on financial-statement-based
forecasting; Fama-French on industry grouping; Vorst-Yohn on forecast-input
interpretation; FASB Statement 2 / ASC 730 on expensed R&D and ASC 260 on
diluted EPS; Lev-Sougiannis, Peters-Taylor, and Ewens-Peters-Wang on
capitalized intangible-investment research; Bhojraj-Lee and
Bhojraj-Lee-Oler on industry/peer classification and valuation; and
Lo-MacKinlay on data-snooping risk. The caps, weights, scenario shifts, fades,
and reversion rates are policy assumptions, not estimates optimized from
those papers or from StanStock outcomes. A separate cutoff-safe historical
replay is required before any activation decision.

`manage.py audit_long_evidence` reports this selection read-only and offline
from persisted evidence, without any provider call. It takes the forecast's
three boundaries separately and explicitly -- `--target-date` for reporting
period bounds, `--available-through` for the historical data cutoff, and
`--decision-time` for as-of evidence visibility -- and rejects a naive
timestamp, an incoherent ordering, or a reader whose decision boundary
differs from the requested one. Listings are addressed by immutable listing
ID; a ticker resolves only when exactly one listing matches, and a symbol
shared across exchanges or reused after a delisting is an explicit ambiguity
error.

The audit reads through exactly the forecast's gates, including the
correction-availability policy, and states which one it applied
(`correction_availability_policy`). Audited against a frozen version it
therefore reports that version's own recorded-availability selection and
defers nothing; audited against long-v3 it reports the same deferrals the
long-v3 forecast would make, listing each withheld correction. Auditing a
frozen version against the prospective policy would describe a selection that
version never makes, which is why one shared, configuration-gated function
answers the question for both readers.

Recovery of an interrupted or repeated ingestion is likewise fail-closed:
one observation instant names one content, an upgraded database with a
correction chain but no observation evidence refuses to replay rather than
ordering stored assets by retrieval, and a fresh retrieval is the separately
gated way to re-establish proven evidence. See `docs/point-in-time.md`.

A malformed configuration fails closed and the YAML parser message is
withheld, because a parse error quotes the offending source line and an
operator who points `--long-config` at the wrong file would otherwise have
that line echoed into stderr and the operator log.

Its base input is:

```text
g0 = cap(
    0.45 * historical_per_share_growth
  + 0.30 * (ROIC * reinvestment_rate)
  + 0.25 * peer_per_share_growth
)

growth[t] = fade[t] * g0 + (1 - fade[t]) * terminal_growth

bounded_current_multiple = cap(actual_current_multiple)

terminal_multiple = capped_geometric_interpolation(
    bounded_current_multiple,
    bounded_peer_multiple,
    reversion
)

cumulative_price_return =
    product(1 + growth[t])
    * terminal_multiple / actual_current_multiple
    - 1
```

NOPAT uses TTM operating income and a bounded GAAP accrual tax proxy: TTM
income-tax expense divided by TTM pretax income. This is not cash taxes paid
and must not be described as a cash tax rate. Invested capital is compatible
debt plus equity minus cash, averaged between the TTM boundaries;
reinvestment is the compatible balance-sheet invested-capital change divided
by NOPAT. It is an accounting proxy, not directly observed capital
expenditure and not a proven causal reinvestment rate. Both snapshots must use
identical canonical and source concepts for equity, cash, and every debt
component.

The 3-year and 5-year forecasts use separate frozen horizon-specific fade
sequences and multiple-reversion settings. Neither is a slice or extrapolation
of one coherent shared 5-year path. A raw current multiple below its
configured family floor is outside long-v1 and is withheld; high multiples
retain the actual price denominator while using the bounded value only as a
conservative reversion anchor. Bear, base, and bull vary only the frozen
growth delta, reinvestment multiplier, and peer multiple multiplier.
Annualized 3y/5y values are derived for display from the stored cumulative
return. Dividends and cash yield are excluded, and positive-return probability
remains unavailable until genuinely qualifying prospective outcomes exist.
SEC continuity checks verify the share basis only through the latest metric
period. The remaining days through the forecast target are stored as
machine-readable `unverified_post_period_split` exposure and shown beside the
forecast; StanStock does not claim that an adjusted-price series proves no
later split occurred.

The released sequence and exact math-only forecast identities are documented in
[`docs/forecast-roadmap.md`](forecast-roadmap.md). The roadmap explicitly
excludes LLMs, trained machine-learning models, analyst targets, and automated
parameter optimization.

## Predictions and evaluation

A prediction records the actual generation time, target market date,
permanent listing ID, source price, explicit forecast horizon, evidence role,
price provider and subject when the exact listing asset proves them, scenarios,
immutable evidence grade and source classification, confidence,
recommendation, component scores, structured calculation provenance, source
assets, configuration hash, and code revision.

Predictions are append-only. A correction uses a new model/configuration
version; the frozen version's config hash, behavior, and output payloads
(including withheld/failure payloads) stay reproducible, and a stricter gate
ships as a separate version rather than rewriting the old one. Missed runs
can be reconstructed for research but cannot be presented as calls issued on
time. On-time status belongs to each immutable prediction, not only its
parent analysis: no reissue inherits another version's status, and each
version -- including a same-target reissue -- independently proves its own
next-market-session-open deadline from cutoff-safe evidence. A reissue before
that deadline may still be observed. An explicit observed request after the
deadline or against cutoff-unsafe evidence raises rather than silently
downgrading; a separate non-observed reconstruction remains research-grade.

Scoring groups remain `short`, `medium`, and `long`; they are not forecast
identities. Forecast identities are `short`, `6m`, `12m`, `3y`, and `5y`,
while historical `medium` and `long` predictions retain their original labels
and 252-/756-session meanings. The corresponding canonical maturities are 10,
126, 252, 756, and 1260 observed sessions; calendar weekends and holidays are
never manufactured.

Decision performance reports only matured canonical decision outcomes and
retains unresolved and corporate-event counts. Decision metrics display after
30 canonical row-level prediction observations in the current exact
method/configuration/provider cohort. This is a fixed display threshold, not
statistical validation. Decision predictions use recommendation success: BUY
succeeds when actual return is greater than 0; AVOID succeeds when actual
return is less than or equal to 0; HOLD succeeds when actual return lies within
the stored bear/bull range, inclusive. Recommendation success is not advisory
base-case sign match.

Advisory reporting is separate. It uses only matured canonical advisory
outcomes whose prediction and parent run were issued on time, whose immutable
evidence grade is observed, whose source mode is provider-backed, and whose
price provider is non-empty. After selecting the earliest reportable issuance
for each canonical listing/target/horizon/evidence-role/method/configuration/
provider observation, it summarizes outcomes by exact target date and keeps
method version, configuration hash, price provider, evidence grade, horizon,
and code revision in separate groups. Only an exact lowercase 40-character
hexadecimal code revision can publish metrics; `working-tree`, empty,
shortened, uppercase, or otherwise malformed revisions remain visible but
withheld.

Each valid target-date cohort has a closed support interval from its forecast
target through its latest evaluation date. Deterministic earliest-finish
scheduling selects the next cohort only when its target is strictly later than
the prior selected evaluation date. Every structurally valid target date,
including a thin one, participates before support is judged; a selected thin
cohort withholds the group rather than being removed in favor of a broader
date. Missing or inconsistent dates, scenarios, outcomes, or stored advisory
metrics withhold the exact group before scheduling.

The reader also checks the stored numerical evidence defensively without
changing the frozen outcome producer. Actual, bear, base, bull, and signed
error values must all be present, and their grouped minima, maxima, and signed
error sum must be finite decimals. PostgreSQL numeric `NaN` is therefore
malformed evidence; the constrained four-decimal columns reject positive and
negative infinity at storage.

Legacy outcome producers evaluate direction and scenario inclusion from the
raw float return, then store actual return and signed error independently at
four decimal places. V4 is the explicit exception described above: its one
half-even-quantized decimal drives every derived outcome field. For legacy
rows, rounding loses information only at specific boundaries: either
direction Boolean is compatible when stored actual return is zero, and either
inclusion Boolean is compatible when stored actual return equals the stored
bear or bull endpoint. Nonzero signs and values strictly inside or outside the
scenario remain unambiguous. Signed-error consistency requires
`ROUND(actual_return - signed_error - base_return, 4)` to lie from `-0.0001`
through `+0.0001`, inclusive; `+/-0.0002` is rejected. This one-quantum rule is
a storage-compatibility boundary, not an empirical tolerance or calibration
claim.

| Horizon | Non-overlapping target cohorts | Listings in every selected cohort | Selected target-date span |
|---|---:|---:|---:|
| 6m | 8 | 30 | 1,095 days |
| 12m | 6 | 30 | 1,460 days |
| 3y | 3 | 30 | 2,190 days |
| 5y | 3 | 30 | 3,650 days |

Within a selected date, advisory sign match, inclusion, and signed base-case
error use all applicable listings. The published group value is the arithmetic
mean of those date-level values, so a large forecast vintage cannot outweigh a
smaller qualifying vintage. The three metrics share one publication gate.
Medium-horizon inclusion compares realized returns with the analog bear-to-bull
range and shows its nominal 60% target only after support qualifies; this is
not a calibration claim or guarantee. Long-horizon inclusion reports only
whether the realized return fell inside the deterministic bear/base/bull
scenario envelope and has no nominal inclusion target. More than 50,000
grouped target-date summaries withholds the complete advisory report rather
than publishing a truncated result.

Advisory predictions never receive a decision-success value; they record
base-case sign match, bear-to-bull or scenario-envelope inclusion, signed
base-case error, and benchmark return separately. A withheld advisory
prediction (all scenario returns null) is non-evaluable: evaluation resolves
it as unresolved before any price lookup, and reporting defensively excludes
it from advisory denominators.

The decision headline on the performance page aggregates only decision
outcomes from observed universe snapshots whose individual predictions were
issued before the next market session. Reportability uses the immutable
evidence grade, source mode, and exact price provider copied onto each
prediction at issuance; later edits to parent snapshot metadata cannot
reclassify evidence. Within each exact listing/target/horizon/evidence-role/
method/configuration/provider observation, aggregate reporting selects the
earliest reportable issuance before looking at outcome status. A later valid
observed reissue remains in the immutable ledger and is evaluated separately,
but it cannot replace an unresolved or corporate-event original, recount the
market observation, inflate a sufficiency threshold, or change the original
aggregate result. Different methods, configurations, providers, horizons,
listings, and target dates remain separate cohorts. Advisory outcomes cannot
enter the decision denominator. Matured synthetic, unsupported, or
research-grade outcomes remain visibly excluded from live, out-of-sample
claims; a non-canonical observed reissue is not reclassified as research.
The price-only baseline
persists one short decision prediction plus separate `6m` and `12m` advisory
predictions. Historical current-universe panel rows are survivorship-biased
research evidence; only subsequently matured on-time predictions can
contribute observed advisory outcomes.

The evaluator also compares the target-date close in the evaluation vintage
with the immutable prediction source price. A material mismatch is classified
as a corporate event/adjusted-history revision and is excluded from ordinary
return, recommendation-success, and advisory outcome calculations.

## Simulations

Backtests and portfolio comparisons use one accounting implementation.
Signals become tradable only on the configured next eligible observation.
Inputs include universe grade, selection rule, rebalance schedule, starting
cash, costs, slippage, currency convention, benchmark, and missing-price/event
rules.

The current engine converts native prices into one explicit base currency
using point-in-time FX. A selection spanning several native currencies must
name its base currency (`--base-currency USD|EUR|GBP`); a single-currency
selection infers it. Every value a run reports -- execution prices, cash,
holdings, trading costs, the benchmark, and all metrics -- is denominated in
that base currency, while the persisted price input keeps each row's native
price, native currency, and the exact rate applied.

Conversion rates are dated, and every valued date is resolved against its own
cutoff -- the end of that date -- so a rate published later can never change
how an earlier execution was priced. A rate published after the valued date is
refused in every run; a source asset merely *retrieved* later is permitted
only for an explicitly research-grade reconstruction, never for observed
evidence. Because FX series have no weekend, holiday, or (for the synthetic
demo bundles) non-Friday observations, the most recent eligible observation is
carried forward and its carry distance is recorded per converted date. The
reviewed maximum carry is 7 calendar days; a run may tighten it to as little
as 0 but cannot widen it, and a carry beyond the limit fails rather than
pricing from a stale rate. Derivation paths are ranked `identity` > `direct` >
`inverse` > `cross:<pivot>`, and the path used is stored alongside each rate.
Equally-ranked derivations that disagree, a missing pair, and an over-stale
observation each fail the whole run instead of producing a
partially-converted result. `--restrict-native-currency` still selects a
single-currency slice of a mixed universe when conversion is not wanted at
all, and is rejected outright when it would silently exclude an explicitly
selected holding.

A holding whose own market is closed keeps its currency exposure: its last
native quote is carried and revalued at the current eligible rate -- both when
the day is valued and when a rebalance sizes its targets -- so a
foreign-market holiday suspends the stock's price discovery without also
freezing the portfolio's exchange rate. The closed holding remains untradable
on that date; only its value is restated. Every simulated date must resolve a
rate for every non-base currency before any accounting begins, so a date the
FX series cannot cover fails the run instead of producing a portfolio return.

Because FX vintages carry no intraday knowability, availability is resolved
only to end-of-day. A converted run therefore executes on closing prices:
`next_open` and `next_eligible` are rejected, because an opening trade could
otherwise be settled at a rate published hours after the bell.

A converted run also reports the split between stock return and FX
contribution. The stock leg revalues the *same* quantity path at each native
currency's rate on the first simulated date, so the two legs sum exactly to
the reported cumulative return. When any part of that restatement is not
established -- for example a cash settlement with no FX basis -- both figures
are withheld with a stated reason rather than reported as an estimate.

The base currency, the native currencies converted, the longest carry
applied, and the attribution are persisted in the run configuration and
metrics. Explicit buy-and-hold selections must all have a usable price on the
common inception execution date; otherwise the run fails rather than silently
redistributing the missing listing's allocation.

Research-grade reconstructed history, observed-universe history, and actual
live prediction outcomes are always labeled separately. Every persisted run
stores the exact price, signal, benchmark, and FX input frames alongside the
result curve so it can be reproduced without relying on mutable current state.
Observed-grade backtests reject analyses generated after their target date;
research-grade reconstructions retain both their historical data cutoff and
actual later generation timestamp in the persisted signal input. The run input
hash covers complete normalized frame contents -- prices, signals, benchmark,
and, when a run converts, the dated FX frame together with the canonical
native-currency assignment and retained conversion inputs -- plus the explicit
calendar, not aggregate row counts or sums. A run that converts nothing adds
no FX terms at all, so a single-currency run keeps the exact reproducibility
identity it had before FX conversion existed.

## Tracked portfolio valuations

Tracked portfolios are separate from historical simulations. The owner enters
current quantity and average cost for each holding. Current value uses the
latest persisted `LatestMarketData` row. SPY can therefore be held and valued
without a `StockAnalysis`; unsupported ETF symbols are rejected. A snapshot
is refused when a holding is inactive, unpriced, in another currency, or more
than seven calendar days behind the newest holding price.

Unrealized gain is securities value minus entered holding cost basis. Cash is
included in total portfolio value but excluded from gain and return, avoiding
the false treatment of cash as investment profit. Snapshot history includes
changes in cash and holdings and is therefore a value series, not a
time-weighted or money-weighted performance claim. Each snapshot and position
is immutable and retains its source asset and market-session date. A
split-sized price move with unchanged quantity is flagged for manual review;
StanStock does not silently rewrite the owner's quantity or average cost.

External cash contributions are separate immutable events. Initial cash on a
new manual portfolio is recorded as a deposit rather than unexplained mutable
cash, and later cash is added only through the deposit workflow. Each deposit
captures an eligible pre-flow snapshot when fresh, coherent, provider-backed
price evidence exists; otherwise it preserves an explicit boundary issue.
Confirmed planner purchases are separate immutable rows, not brokerage fills.
They retain the exact listing, quantity, persisted close, market session,
price asset, policy version, and plan hash.

The versioned `monthly-allocation-v1` preview uses total NAV including cash:

```text
SPY budget = min(cash, max(0, 0.70 * NAV - SPY value))
satellite budget =
    min(remaining cash, max(0, 0.30 * NAV - other invested value))
```

It never sells. Fractional mode rounds quantities down to eight decimal
places; whole-share mode rounds down to whole units. Any residual remains
cash. At most one stock satellite is selected, and only from provider-backed
analysis on the same session as SPY whose opportunity evidence is explicitly
short-horizon. Under-$10 names remain ineligible for new allocation. The plan
hash binds portfolio settings and holdings, current price assets and sessions,
and the exact qualifying analysis/run/configuration/source-asset evidence.
Confirmation locks and recomputes that complete state; a stale preview is
rejected.

Contribution-adjusted profit/loss is:

```text
current NAV - eligible tracking-boundary NAV - later external deposits
```

The percentage divides by the boundary NAV plus those later deposits, so it is
labeled a simple since-boundary return rather than time-weighted or
money-weighted performance. Cash and quantities must reconcile exactly to the
immutable deposit and planner-purchase ledgers. Current and boundary
valuations require one fresh, non-future market session, Twelve Data
split-adjusted price-return metadata, dividend exclusion, and no unresolved
corporate-action warning.

A supported manual quantity change or removal is neither silently treated as
return nor allowed to disable the metric forever. It appends an immutable
post-change performance baseline and restarts measurement from that value.
If another holding prevents a complete valuation, an immutable
`boundary unavailable` baseline is recorded, the holding edit still commits,
and performance remains withheld. A later valid manual change can append a
newer eligible baseline. Ordinary scheduled snapshots do not themselves reset
the performance boundary.

A StanStock sample portfolio is a distinct frozen research artifact. It
equal-weights eligible opportunities from one provider-backed analysis run,
stores that run and the construction-policy version, and records a baseline at
the analyses' immutable reference closes. Those closes are not represented as
executable fills. Later snapshots use the latest persisted market rows without
rebalancing, and model return is total current value relative to starting
capital. A split-sized move withholds that headline return until reviewed.
The current price-only sample records a short-horizon signal, so later
buy-and-hold performance is observational rather than evidence that the
original short thesis remained valid. New sample construction records the
price-band policy, classifies the analysis reference close at the run's target
date, and excludes both ETFs and Under $10 names without consulting later
market state. An older frozen basket is never rewritten when the policy
advances.
