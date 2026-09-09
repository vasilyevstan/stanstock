# Methodology

StanStock v1 uses deterministic, versioned rules. It does not fit a predictive
machine-learning model and does not permit generated text to alter a score,
scenario, risk class, or recommendation.

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

The prospective US price-only v2 policy removes nominal share-price scale from
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

Weights, thresholds, coverage requirements, and scenario parameters live in
versioned YAML. Every issued prediction stores the configuration hash and code
revision. These weights are starting assumptions; they are not optimized
against the period later used to report performance.

## Normalization and missingness

Cross-sectional factors use robust ranks or winsorized standardized values
within an appropriate region/sector peer group. Annual European fundamentals
are not treated as equally fresh to US quarterly facts.

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
  before p20/p50/p80 estimation, and sparse conditional ranges shrink toward
  the unconditional horizon distribution.
- **3 years and 5 years:** separate deterministic advisory cases using
  point-in-time SEC facts, current point-in-time SIC peers, and the exact
  split-adjusted Twelve Data price asset. Positive compatible FCF/share takes
  priority. EPS/share is permitted only when FCF evidence is genuinely
  unavailable; negative, inconsistent, or incomplete FCF cannot trigger a
  more favorable fallback.

Bear, base, and bull are ordered ranges, not precise target prices.
Medium-horizon probability of positive return is omitted with an insufficiency
reason until non-overlapping cohort support, listing diversity, calendar span,
matched market-regime breadth, and frozen walk-forward calibration gates
all pass. Walk-forward calibration compares the conditional range midpoint
with both the unconditional median and a SPY-relative decomposition baseline:
the historical SPY median for the matching market regime plus the historical
excess-return median for the matching relative-momentum bucket.
Because the narrowest fallback levels explicitly condition on market regime,
they may yield a useful range while still failing the matched-regime breadth
gate for probability. StanStock does not switch to a broader fallback merely
to publish a probability.

The medium panel stores 50/200-session trend, downside volatility, and dollar
liquidity for eligibility and explanation, but those values do not add hidden
matching dimensions. All panel inputs are capped at their historical anchor,
and a forward label is present only when its complete outcome ends on or
before the current forecast target.

The long engine requires at least three contiguous annual per-share periods,
compatible TTM metric and diluted-share periods, reported diluted-EPS
share-basis checks for every selected annual period, TTM diluted shares
within 15% of the latest overlapping annual share basis, a bounded cash tax
rate, beginning and ending invested capital using identical canonical and
source concept definitions, and a same-family SIC peer set meeting frozen
floors. The default `us-sec-long-v2` configuration additionally checks
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

NOPAT uses TTM operating income and a bounded tax expense/pretax-income rate.
Invested capital is compatible debt plus equity minus cash, averaged between
the TTM boundaries; reinvestment is its change divided by NOPAT. Both
snapshots must use identical canonical and source concepts for equity, cash,
and every debt component. A raw current multiple below its configured family
floor is outside long-v1 and is withheld; high multiples retain the actual
price denominator while using the bounded value only as a conservative
reversion anchor. Bear, base, and bull vary only the frozen growth delta,
reinvestment multiplier, and peer multiple multiplier. Annualized 3y/5y
values are derived for display from the stored cumulative return. Dividends
and cash yield are excluded, and positive-return probability remains
unavailable until genuinely qualifying prospective outcomes exist. SEC
continuity checks verify the share basis only through the latest metric
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

Performance reports only matured outcomes and retains unresolved corporate
events in coverage counts. Aggregate return, hit-rate, or calibration metrics
are withheld below the configured minimum sample. Decision predictions keep
the existing BUY/HOLD/AVOID success semantics. Advisory predictions never
receive a decision-success value; they record direction correctness, bear/bull
interval coverage, signed base-case error, and benchmark return separately. A
withheld advisory prediction (all scenario returns null) is non-evaluable:
evaluation resolves it as unresolved before any price lookup, and reporting
defensively excludes it from advisory denominators.

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
return and directional-accuracy calculations.

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
