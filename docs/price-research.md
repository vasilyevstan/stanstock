# Price Research Product

> Release state: see [README](../README.md#release-status).

This is the normative public explanation of `research-product-v1`. The product
uses two deterministic operators. It is not an LLM, fitted model, paper
replication, fair-value engine, or trading system.

## Output contract

For each qualified listing:

- one `StockAnalysis`;
- one `Prediction(horizon="6m", role="decision",
  method="us-relative-momentum-v1")`;
- four advisory predictions for `us-price-fhs-v1` at `6m`, `12m`, `3y`, and
  `5y`.

Scores, numeric confidence, and the original positive-return probability
field are null. Every missing calculation has an explicit reason. A
separately registered outcome-frequency report does not change these frozen
prediction rows or their meaning.

## Required evidence

- US/USD common stock or supported ADR.
- Verified permanent listing and official catalog identity.
- Registered immutable split-adjusted stock history.
- Separate registered SPY benchmark history.
- Exactly 757 consecutive common XNYS-session closes ending at target `T`.
- Finite, strictly positive closes and no duplicate/missing required session.
- Split-adjusted price returns excluding dividends.

Volumes are optional for projections but need independently compatible
adjustment evidence for the BUY liquidity gate.

The live candidate set is the unchanged 100-name curated core plus at most 20
captured owner-saved names, deduplicated by permanent listing ID. SPY is not
stock membership.

## Relative 12–1 momentum

Stock and SPY use identical session endpoints:

\[
m_i=\log(P_{i,T-21}/P_{i,T-252})
\]

\[
m_B=\log(P_{B,T-21}/P_{B,T-252})
\]

\[
M_i=m_i-m_B
\]

StanStock records the stock's skipped-month price return
\(\exp(m_i)-1\) and the benchmark-relative log momentum \(M_i\).

Direction:

- positive: \(m_i>0\) and \(M_i>0\);
- negative: \(m_i<0\) and \(M_i<0\);
- mixed: otherwise, including equality.

This is an adapted single-stock policy, not the momentum-portfolio
construction in the motivating literature.

## Risk and suggestion policy

Using the same aligned history:

\[
\sigma_i=\sqrt{252q_{i,757}},\qquad
\rho_i=\sigma_i/\sigma_B
\]

The product also calculates:

- maximum drawdown over the latest 252 returns / 253 closes; and
- compatible 20-session mean dollar turnover.

Suggestion:

1. negative direction -> AVOID;
2. mixed direction -> HOLD;
3. positive direction -> BUY only when:
   - source/identity/date checks pass;
   - \(\rho_i\le2\);
   - drawdown is no worse than \(-50\%\);
   - compatible dollar turnover is at least $5 million; and
   - target close is at least $10;
4. otherwise positive -> HOLD with exact blockers;
5. missing momentum -> unavailable.

Nominal price never increases conviction. Under $10 adds
`speculative_watch_0_percent_new_allocation` and blocks BUY promotion without
changing raw direction.

The current Twelve Data source documents split-adjusted prices but does not
prove compatible volume adjustment. That missing provenance can keep turnover
unavailable and BUY blocked even when direction and projections calculate.

## Opportunities shortlists

The paginated full comparison appears first, including $300+ listings. On
the first page without a selected price band, secondary shortlists separate
up to three BUY-qualified candidates from up to three positive-momentum
research candidates in each of the Under-$10, $10-$50, and $50-$300 bands.
Price bands use the immutable run's reference close and show its market
date; nominal price is affordability context, not value.

Shortlists order the existing benchmark-relative momentum, then ticker and
permanent listing ID. They do not rank simulation shares, median projections,
or the selected forecast horizon. BUY candidates must additionally pass the
unchanged recommendation and current-source promotion guards. Under $10
remains a speculative watch with 0% new allocation, not a BUY highlight.

Shortlist cards show the decision and its restrictions without a competing
simulation headline. The full comparison and detail pages retain the selected
horizon's advisory scenarios. These are new-purchase research signals, not
portfolio-specific instructions to sell or retain an existing holding.

Filters apply consistently to shortlists and counts. A selected price band
or a later results page shows only the focused full list. Empty BUY and
Under-$10 shortlists are valid outcomes; missing liquidity is never treated
as zero or ignored to fill a section.

The lists update from each newly verified run within the curated core and
captured owner-saved cohort. This is daily re-ranking, not automatic discovery
of additional stocks. Wider intake remains a separate source, rights, and
provenance decision; see [source capabilities](source-spike.md).

## Filtered historical simulation

For \(N=756\) daily log returns:

\[
r_t=\log(P_t/P_{t-1}),\qquad
\mu=N^{-1}\sum r_t
\]

\[
v=N^{-1}\sum(r_t-\mu)^2,\qquad q_1=v
\]

\[
z_t=\frac{r_t-\mu}{\sqrt{q_t}}
\]

\[
q_{t+1}=0.01v+0.94q_t+0.05(r_t-\mu)^2
\]

Discard \(z_1,\dots,z_{252}\). Center and rescale the remaining 504 values:

\[
e_t=(z_t-\bar z)/s_z
\]

The retained residual population has zero arithmetic mean and unit second
moment. Invalid variance/scale withholds output; no floor, clipping, jitter, or
replacement return is introduced.

### Forward paths

Generate 8,192 paths with NumPy PCG64. Residuals are sampled independently and
uniformly with replacement. For path \(j\):

\[
\epsilon^{(j)}_h=\sqrt{q^{(j)}_h}e^{*(j)}_h
\]

\[
r^{(j)}_h=\mu+\epsilon^{(j)}_h
\]

\[
q^{(j)}_{h+1}
=0.01v+0.94q^{(j)}_h+0.05(\epsilon^{(j)}_h)^2
\]

\[
R^{(j)}_H=\exp\left(\sum_{h=1}^{H}r^{(j)}_h\right)-1
\]

Horizon sessions:

| Label | Sessions |
|---|---:|
| 6m | 126 |
| 12m | 252 |
| 3y | 756 |
| 5y | 1260 |

The deterministic seed derives from method version, effective config hash,
permanent listing UUID, and target date. Random indices are generated in
path-major order. Complete inputs and calendar are hashed separately.

## Model-estimated probabilities

The probability-first presentation uses the same 8,192 deterministic terminal
paths, not a distribution inferred from three quantiles and not a new fitted
model. A separate versioned report counts each path once in one of three
exhaustive outcomes:

| Label | Terminal cumulative price return |
|---|---|
| Loss | \(R_H < 0\) |
| Flat to +20% | \(0 \le R_H \le 0.20\) |
| Above +20% | \(R_H > 0.20\) |

For event \(A\), its model-estimated share is
\(\widehat p_A = \#\{j:R_H^{(j)}\in A\}/8192\).
Classification compares terminal log returns with the fixed log-space
thresholds \(0\) and \(\log(1+0.20)\). Exact zero and the upper boundary
belong to the middle outcome. Any nonfinite path withholds that horizon's
summary; the denominator is never filtered to make an estimate available.

The stock detail also shows the nested event \(R_H < -0.20\), using
\(\log(1-0.20)\). This means **finishing more than 20% down**, not reaching a
20% drawdown at any point along the path. It is a subset of Loss, not a fourth
outcome to add to the three shares. The same events on the existing
zero-log-drift paths show dependence on the assumed drift, without another
model or another random sample.

All thresholds concern total price return over the selected horizon, not an
annual rate; dividends, fees, and taxes are excluded. The absolute +20%
threshold has a different economic meaning over six months and five years.
The horizon and dated reference close must remain visible.

### Presentation and nonclaims

The heading is **Model-estimated probabilities**, accompanied by
**Shares of model simulations; not validated real-world odds.** Probability
shares have no plus/minus prefix; projected returns retain their signs.
Median return is not a mean, expected return, or fair value.

Exact integer counts are stored. Whole-percent display uses largest-remainder
rounding with fixed Loss, Flat, Above tie order. Plain numeric shares total
100%. A small nonzero share rounded to zero is labelled `<1%`, and a share
below all paths rounded to 100 is labelled `>99%`; symbolic bounds are not
presented as an exact arithmetic total. Exact counts remain in details.
Genuine zero/all-path counts describe only the simulated sample, never an
impossible/certain market outcome. The complement of Loss includes unchanged
prices and is not a strictly positive-return probability.

The model's historical-drift assumption, fixed variance dynamics, omitted
corporate events, and lack of parameter uncertainty also limit these shares.
More simulated paths can improve numerical precision; they cannot establish
empirical support, calibration, or profitability. No new numeric calibration
or skill claim is introduced by the outcome report. A future calibration
study requires its own frozen, overlap-aware protocol and per-horizon
evidence, distinct from the existing retrospective quantile comparisons.

### Evidence and availability

Registration derives the entire report internally from the exact run's
verified immutable sources and records complete input, seed, listing, horizon,
method, and projection identity. Matching the old quantiles is a necessary
compatibility check, not proof of the counts: different distributions can
share the same three quantiles. No registration interface accepts supplied
probabilities or summary statistics.

The new report is an append-only `DataAsset`, not a new prediction or a
replacement source vintage. Its own `available_at` limits historical reads.
A summary derived later is explicitly a later reconstruction; it never
inherits the prediction's on-time status. Reads verify registered evidence
without simulation, and an explicit offline verification re-derives the
counts. Missing or failed summary evidence has a separate state from the
underlying forecast. An independently valid median/range stays available.

## Detailed Lower, Median, Upper

NumPy's linear quantile convention is used:

- Lower = \(Q_{0.20}(R_H)\)
- Median = \(Q_{0.50}(R_H)\)
- Upper = \(Q_{0.80}(R_H)\)

Corresponding prices are \(P_T(1+Q_p)\).

The interval is central 60% **model mass**. It is not:

- demonstrated real-world coverage;
- 80% coverage;
- a probability of gain;
- calibrated confidence;
- a worst-case envelope;
- a stop-loss recommendation; or
- a fundamental fair value.

Return ledger values use four decimal places and prices use six, both
half-even rounded from the same unrounded trajectory.

## Drift disclosure

The central projection continues the trailing three-year mean log return.
This is a conditional historical-drift extrapolation, not a reliable expected
return estimate.

The zero-drift sensitivity subtracts \(H\mu\) from the same terminal log-return
paths, keeping the sampled shocks identical. It measures drift dependence; it
is not a third fitted model.

Momentum and FHS median may disagree because they answer different questions.
The UI must show that disagreement rather than vote or blend.

### Inline forecast explanations

The **Why this forecast?** disclosure on each active Opportunities comparison
and shortlist card describes the selected horizon using the same verified
run and listing as the displayed result. Independent native disclosures start
collapsed and reset on page/filter/horizon navigation. Opening one performs
no request, source lookup, model calculation, or AI inference.

The explanation shows the recorded mean **daily log return** over 756 returns
(approximately three trading years), the selected cumulative price-return
range, and the median from the stored same-shock zero-log-drift sensitivity.
Historical drift shifts the simulated distribution, but is not itself the
simulated median. Either drift sign can coexist with the opposite median sign.
Zero log drift does not imply a flat price or zero expected arithmetic return.
Rounded-zero results do not imply an absence of risk.

The action explanation remains separate: stock and benchmark skipped-month
momentum use T-252 through T-21, while six months is the decision's future
horizon. Recorded risk and eligibility blockers explain restrictions without
recalculating the action. Missing compatible volume evidence is not evidence
of low trading activity.

Unavailable evidence remains explicit; a missing simulation-frequency report
does not remove an independently valid price projection. Existing restrictions
and the simulation-share disclaimer remain visible without expansion.
The explanation describes conditional price-history assumptions, not company
news, earnings, fundamental value, validated real-world odds, or a promise
of a particular return. Price projections exclude dividends.

## Retrospective protocol

The study is frozen before real holdout inspection:

- fixed anchor epoch: 2019-09-03;
- 756 prior returns required;
- anchors spaced by the evaluated horizon;
- development outcomes complete before 2024-01-01;
- validation anchor and outcome both within calendar 2024;
- final-holdout anchor on/after 2025-01-01 and outcome complete through
  2026-09-11;
- crossing intervals purged;
- no holdout tuning.

### Baselines

On the same exact paired listing/anchor/maturity scope:

1. zero-log-drift Gaussian, variance \(Hv\);
2. historical-log-drift Gaussian, mean \(H\mu\), variance \(Hv\).

Results are averaged within target cohorts before equal-cohort comparison.
Many listings at one anchor do not become many independent time observations.

### Metrics

- mean absolute error of the median forecast;
- p20, p50, and p80 pinball loss;
- interval width;
- interval inclusion; and
- central-60% interval score.

MAE, pinball loss, and interval score are lower-is-better. Width and inclusion
are descriptive, not standalone quality rankings. A narrow interval can miss;
a wide interval can include without being sharp.

Realized losses use recorded actual returns. Simulated path count is not
market support. Empty partitions, unavailable comparisons, and
worse-than-baseline results remain visible.

### Evidence labels

The retrospective report is a
`current-universe_current-vintage_retrospective-math-replay`. It can use a
research-grade provider source, but it cannot claim the source was available
at each historical anchor.

The report records separately:

- replay execution revision;
- source-run revision;
- source generation/cutoff/retrieval identity;
- report generation time; and
- actual registration availability.

## Observed forward evaluation

The six-month decision uses exact stock and SPY endpoints:

- BUY success: stock return is positive and exceeds SPY;
- AVOID success: stock return is negative and is below SPY;
- HOLD/unavailable: no success label.

Advisory rows record realized return/error/interval diagnostics, not
recommendation success. Each immutable version proves its own issuance
deadline. Retrospective replay never creates observed predictions.

## Numerical diagnostics

The frozen diagnostic doubles the path count to 16,384 while preserving the
first 8,192 paths. Quantile movement beyond
`max(0.01 return, 0.02 * production interval width)` is investigated.

This is a numerical convergence check, not financial calibration, evidence
breadth, or a gain probability.

## Research grounding

- Jegadeesh and Titman (1993):
  <https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>
- Carhart (1997):
  <https://doi.org/10.1111/j.1540-6261.1997.tb03808.x>
- Bollerslev (1986):
  <https://doi.org/10.1016/0304-4076(86)90063-1>
- Barone-Adesi, Giannopoulos, and Vosper (1999), *Journal of Futures Markets*
  19(5), 583–602.
- J.P. Morgan/Reuters (1996), *RiskMetrics—Technical Document*, fourth
  edition.

These works motivate ideas, not StanStock's exact coefficients, thresholds,
single-stock suggestions, horizons, or profitability.

## Synthetic drift and parameter-uncertainty research

The separate `research/price_product_drift_study.py` module compares unchanged
historical-drift and same-shock zero-log-drift FHS controls with two synthetic
sensitivities: plug-in half-drift, and half-drift with one persistent drift
draw per path shared across horizons. Its fixed assumptions are `m = mu / 2`
and `u2 = v / (2 * 756)`, using the same historical variance estimate.
They are not an optimal shrinkage rule, calibrated posterior or
dependence-robust uncertainty estimate.

Half-drift moderates negative as well as positive trends. The added uncertainty
illustrates a chosen assumption, not proven forecast improvement. Hypothetical
score examples are not market outcomes, independent confirmation or a model
selection result. Equal cumulative deterministic drift at one horizon is
terminal-equivalent under the same shocks, not a claim about interim drawdowns.
No fading or regime model is implemented in this slice.

This module performs no source loading or persistence and does not alter
`us-price-fhs-v1`, its existing zero-drift display or production predictions.
Real-data adaptation, prospective issuance, evaluation and any adoption require
their own accepted evidence and authorization.

## Unscored shadow-drift preparation

The separate `research/price_product_shadow_drift.py` module prepares three
in-memory research arms: unchanged historical-drift FHS, unchanged same-shock
zero-log-drift FHS, and a fixed half-drift candidate. The candidate adds
`H * (mu / 2)` to the native zero-drift terminal log returns, where `mu` is
the recorded historical mean daily log return and `H` is the horizon in
sessions. It does not refit the filter, resample shocks or add parameter
uncertainty. Halving drift moderates negative as well as positive trends;
this is an assumption to investigate, not a demonstrated improvement.

`shadow-fhs-drift-v1` emits six unscored projections: three arms at 6m and
12m. It retains the native 8,192 paths and all four simulation horizons
internally, because shortening the maximum horizon would change the random
stream. Historical and zero-drift controls are copied from the native result
and checked against complete regenerated projections. Native withholding is
preserved; the candidate cannot rescue an unavailable native horizon.

Results and canonical serialized bytes carry `research_only_unscored`,
`caller_supplied_unverified` and `not_frozen` labels. Their complete hashes
identify supplied inputs and assumptions; they do not authenticate source
assets, ownership, calendar provenance or observed issuance. The pure API
loads no data or configuration, reads no clock, and performs no I/O or
persistence. It has no command, UI, scheduled-job or production consumer.
The existing synthetic drift study and registered retrospective protocol
remain unchanged.

The intended future primary objective is 6m central-60% interval score, with
median absolute error as a guardrail; 12m is a secondary diagnostic.
Neither metric is calculated by this preparation module, and 3y/5y evaluation
is outside its scope. Empirical execution requires its own frozen protocol
and authorization. Already-inspected historical partitions cannot become
fresh confirmation for a candidate chosen afterward. Synthetic preservation
and numerical checks do not establish accuracy, calibration or adoption.

## Synthetic candidate-policy research

`price-candidates-synthetic-v1` is a separate, unwired correctness study in
`research/price_candidate_policy.py` and `price_candidate_policy_study.py`.
It does not change the active momentum decision, FHS projections, Opportunities
shortlists, portfolios or scheduled refresh. It publishes no investment list.

The study assesses three fixed price-pattern hypotheses independently:

| Hypothesis | Fixed research conditions |
|---|---|
| Continuation | Positive absolute and benchmark-relative 12-1 momentum, plus positive absolute and relative 21-session returns |
| Positive-trend pullback | Positive absolute and relative 12-1 momentum; a selected 10%-30% decline followed by at least 5% recovery |
| Deep reversal | Negative absolute and relative 12-1 momentum; a selected decline of at least 30% followed by at least 10% recovery |

The decline episode uses a 126-session lookback, excludes the latest five closes,
and selects the deepest ordered peak-to-trough pair, with latest-trough and
earliest-peak tie rules. Recovery arms require a trough 5-21 sessions old,
positive absolute and relative five-session returns, and three strictly
rising final closes. Recent relative return is a wealth ratio,
`(stock_end / stock_start) / (benchmark_end / benchmark_start) - 1`, not
percentage-point outperformance or log momentum.

Matched entry patterns still require the independent stock, source, USD,
compatible-turnover, volatility, drawdown and $10 gates. The frozen momentum
recommendation is a labelled control, not a veto: the synthetic deep-reversal
case can qualify for research while its unchanged native control says AVOID.
Missing compatible volume withholds entry; it is not zero turnover.

Independent deterioration review requires negative absolute and relative
21-session returns, a close strictly below the preceding 20 closes, and
current drawdown of at least 10%. It does not assume ownership or prior entry.
Entry affordability restrictions do not conceal deterioration in an
Under-$10 synthetic case; this does not authorize investment or live selling.
Current drawdown, maximum annual drawdown and selected episode depth remain
distinct quantities.

Six fixed synthetic cases use 757 aligned XNYS closes ending 2026-09-11.
They establish constructible examples and implementation behavior, not market
support. Six separate hypothetical payoff rows illustrate cash-minus-hold,
benchmark-minus-hold, avoided loss and foregone upside at 126 and 252 sessions.
Their zero cash return, zero differential costs and 5% benchmark return are
illustrative assumptions, not forecasts or achieved trades. They are not
outcomes of the pattern cases and are not averaged into a success statistic.

The policy and driver perform no application/private file access, ORM access,
provider/network calls, subprocess execution, clock reads or environment reads.
Bounded deterministic calendar construction may cause the calendar dependency
to read installed timezone resources on its first invocation. This exception
permits neither arbitrary filesystem access nor application, private or
provider-data reads.

Serialization returns bytes with complete input/configuration hashes and
caller-supplied, independently bound source/dependency identity. Synthetic
identity checks establish fixture consistency, not authority to relabel real
market data. Unrepresentable arithmetic fails explicitly; native insufficiency
and genuine missingness retain their separate meanings.

There is no pooled ranking, top-five selection or evidence of predictive
skill. The thresholds are fixed hypotheses, not validated defaults.
Compatible live-volume provenance, adequate independent confirmation and a
complete real-outcome protocol remain prerequisites for separately authorized
real-data research and any eventual investment-list activation.
