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

Scores, numeric confidence, and positive-return probability are null. Every
missing calculation has an explicit reason.

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

## Lower, Median, Upper

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
