# Research-integrity review checklist

Applies to any change touching provider data, `AsOfData`, scoring/prediction
methodology, outcomes, or simulations. Reviewer: `stanstock-research-integrity`.

## Provider rights and provenance

- [ ] No real provider credential or live network fetch is used in tests or
      CI; only synthetic fixtures.
- [ ] Every `DataAsset` (and derived `FundamentalFact`/`FxRate`/
      `LatestMarketData`) row records `provider`, `sha256`, `retrieved_at`,
      and `available_at`, and traces to a `source_asset`.
- [ ] A new or changed provider integration checks `ProviderRecord.enabled`
      and terms/licensing status before any live-mode code path activates.
- [ ] Asset files stay inside `STANSTOCK_DATA_DIR` (`AssetStore.resolve`
      guard); no path can escape the configured root.

## Permanent IDs and immutable vintages

- [ ] New identity-bearing models use a permanent, non-reassignable ID
      (the existing `UUIDField(default=uuid.uuid4, editable=False)` pattern).
- [ ] `Prediction`, `DataAsset`, `FundamentalFact`, and `FxRate` rows are
      never mutated or deleted in place; a correction is a new row with a new
      vintage/`available_at`.
- [ ] Uniqueness constraints on vintage-bearing tables (e.g.
      `unique_fundamental_vintage`, `unique_fx_vintage`,
      `unique_prediction_version`) are preserved or extended, not weakened.

## As-of / look-ahead correctness

- [ ] Every historical read filters `available_at <= decision_time` (via
      `AsOfData` or an equivalent explicit filter); no code path substitutes
      a provider's "latest" state for a past decision.
- [ ] Eligible price assets are also filtered row-by-row to the requested
      market date, with a normalized date type and explicit schema failure.
- [ ] `retrieved_at` is never used as the decision-time boundary in place of
      `available_at`.
- [ ] Backtests and simulations cannot observe a row whose `available_at` is
      after the simulated observation date.
- [ ] `generated_at`, `data_cutoff`, and source `retrieved_at` remain
      distinguishable; observed-grade backtests reject late-generated signals,
      while research reconstructions remain explicitly labeled.
- [ ] On-time status is recorded on each immutable prediction version. No
      reissue inherits another version's status; the original issuance and
      every same-target reissue are each independently checked against the
      next market-session open from cutoff-safe evidence, a reissue before
      that deadline may still be observed, and simulation/reporting do not
      trust an analysis-level flag alone.
- [ ] Aggregate reporting selects the earliest reportable prediction before
      outcome status and counts it once per exact listing/target/horizon/
      evidence-role/method/config/provider observation. Later valid observed
      reissues remain in the immutable ledger and are evaluated per version,
      but cannot replace an unresolved/corporate-event original, inflate
      sample sufficiency, or become a second market observation.
- [ ] An exceptional direct/manual reissue is a direct
      `analyze_snapshot(..., issued_on_time=True, ...)` service call, never
      `manage.py analyze` (which explicitly requests `issued_on_time=False`
      for every target); it explicitly binds the
      production scoring config, provider, and benchmark and sets
      `STANSTOCK_CODE_REVISION` to the exact committed revision rather than
      relying on a generic demo default or `code_revision()`'s
      `"working-tree"` fallback, and its on-time deadline is independently
      reproved before invocation rather than assumed. An unsafe explicit
      request raises; any separate non-observed reconstruction is
      research-grade.
- [ ] Predictions and persisted horizon scores exist only for the scoring
      configuration's `supported_horizons`; withheld horizons cannot enter
      outcomes, unresolved counts, or performance aggregates.
- [ ] Mutable current-market state advances monotonically by market session
      (with retrieval time only as a same-session tie-breaker); historical
      catch-up and ineligible series cannot replace a newer eligible close.
- [ ] Migration/backfill code does not assign historical cutoffs that the
      legacy calculation path cannot prove; ambiguous legacy runs remain
      conservative or are rejected.
- [ ] Outcome horizons count distinct observed sessions; weekends, holidays,
      and duplicate dates are not treated as extra sessions.

## Forecast identity and decision-policy isolation

- [ ] Score groups (`short`, `medium`, `long`) and persisted forecast
      identities are separate contracts; adding 6m/12m/3y/5y cannot change a
      recommendation or opportunity highlight unless a later, explicitly
      reviewed policy version opts in.
- [ ] Every prediction states whether it is decision evidence or advisory
      forecast evidence. Advisory rows are excluded from headline
      recommendation hit rates even when they were issued on time.
- [ ] Legacy `medium` and `long` predictions retain their original horizon,
      maturity, method version, and evidence meaning; migrations do not
      relabel them as canonical 12m or 3y forecasts.
- [ ] Scenario returns, stored base error, and actual outcomes share one
      cumulative split-adjusted price-return basis. Annualized figures are
      derived display values, and dividends/cash yield are not mixed into
      price-return columns.
- [ ] Advisory outcomes are evaluated with explicit direction, interval, and
      error semantics rather than the short-horizon recommendation copied
      from their parent analysis.

## ETF identity and stock-research isolation

- [ ] An investable benchmark ETF reuses the already-required immutable
      benchmark asset; no second provider request or credit is consumed.
- [ ] The ETF has explicit security identity and no stock-universe membership.
      Analysis, fundamentals, predictions, opportunity promotion, and
      sample-stock construction reject or exclude it at service boundaries.
- [ ] ETF return, volatility, and drawdown use the persisted split-adjusted
      price series and label dividend exclusion; no stock recommendation or
      long-horizon claim is inferred from those metrics.
- [ ] Personal portfolios may value the supported ETF from
      `LatestMarketData` without creating a `StockAnalysis`; unsupported ETF
      symbols remain unavailable.
- [ ] Retry/recovery and upgrade synchronization can reconstruct the ETF
      market row from an existing immutable asset without credentials, quota,
      or network access.
- [ ] ETF identity materialization cannot roll back an already completed stock
      analysis run; a failed final projection is explicit and retryable from
      the run's exact benchmark asset at zero additional credits.

## Empirical 6m/12m panels

- [ ] The complete training panel is persisted as an immutable derived asset
      whose hash covers every row, source asset, calendar, and configuration
      input used by the forecast.
- [ ] Anchor dates use a fixed configured epoch and non-overlapping
      horizon-spaced cohorts. Moving the current target date does not
      arbitrarily shift the historical grid.
- [ ] Every state feature uses only rows at or before its anchor, and every
      126/252-session forward label ends on or before the forecast cutoff.
- [ ] Current-universe/survivorship bias is labeled as reconstructed research
      evidence and cannot enter an observed live-skill headline.
- [ ] Shrinkage and positive-return probability use overlap-aware effective
      support plus distinct-listing, calendar-span, and regime-diversity
      floors. Raw pooled rows alone cannot unlock probability.
- [ ] The stored explanation reports raw matches, effective cohorts,
      diversity, fallback level, shrinkage weight, and comparison with the
      unconditional and SPY-relative baselines.

## SEC fact normalization and long forecasts

- [ ] Official symbol/exchange/CIK mapping, submissions history (including
      referenced historical files), and Companyfacts raw bytes are preserved
      before any normalized fact is written.
- [ ] SEC availability uses accession `acceptanceDateTime`; naive values are
      interpreted in `America/New_York`. A date-only fallback is conservative
      and never treated as midnight UTC on the filed date.
- [ ] Fundamental-fact identity distinguishes instant and duration facts and
      includes the full reporting interval, so quarter and YTD observations
      sharing an end date cannot collide or overwrite one another.
- [ ] A changed source observation under the same accession appends a later
      available vintage. Amendments/restatements replace only the affected
      concept/period after their own availability time and never create a
      synthetic growth period.
- [ ] Discrete-quarter derivation uses compatible YTD facts; TTM requires four
      contiguous comparable quarters, explicit accession lineage, and
      52/53-week-year tolerance. Instant facts, EPS, and weighted-average
      shares are never summed or subtracted as additive flows.
- [ ] Historical per-share growth uses compatible SEC-reported/restated share
      bases. Historical valuation multiples are withheld when the price and
      filing share basis cannot be reconciled across splits.
- [ ] FCF/share and EPS/share are separately declared metric families with
      fixed eligibility. Negative or unsupported FCF cannot silently switch
      to a more favorable EPS branch.
- [ ] Sustainable-growth, peer-normalization, terminal-growth, fade,
      reversion, and scenario caps are fixed in versioned configuration.
      Missing required terms withhold the forecast rather than being dropped
      or reweighted.
- [ ] 3y/5y positive-return probability remains unavailable until genuinely
      qualifying point-in-time/live evidence exists; UI wording does not
      imply that calibration is imminent.

## Research-grade vs. observed history

- [ ] `UniverseSnapshot.grade` (`research` vs `observed`) is preserved end to
      end; reconstructed history is never silently merged with live-captured
      membership.
- [ ] Consumers (views, exports, reports) are told which grade they are
      reading when it affects interpretation.
- [ ] Retry recovery finds unique committed work by universe/config/target
      across retry-time grades before credentials or quota are consumed, and
      conflicting completed runs fail explicitly.

## Missing values

- [ ] Missing or insufficient data is represented with an explicit flag
      (`insufficiency_reason`, `quality_flags`, `confidence_status`), never
      coerced to zero, `None`-as-zero, or a default/success-shaped score.
- [ ] A low-confidence or insufficient result is visibly distinguishable from
      a high-confidence one downstream (admin, view, export).

## Price-scale and affordability invariance

- [ ] Nominal share price is never a positive signal, valuation shortcut, or
      reason to raise score/confidence; affordability bands remain outside
      research arithmetic.
- [ ] A split-equivalent transformation (`price * k`, `volume / k`) preserves
      every prospective normalized momentum/liquidity factor, score,
      recommendation, and eligibility decision.
- [ ] MACD-like price-difference indicators are normalized by a compatible
      positive price basis before cross-security scoring.
- [ ] Liquidity used in scoring or BUY gates is dollar-denominated from
      compatible price/volume observations; raw share count cannot satisfy a
      prospective liquidity gate.
- [ ] Invalid, non-finite, zero-basis, or adjustment-incompatible inputs remain
      explicitly unavailable rather than becoming zero or a passing gate.
- [ ] A prospective scoring change uses a new immutable configuration version;
      historical hashes/calculation paths remain unchanged and performance is
      not pooled across materially different versions.
- [ ] Current USD price bands use the latest valid persisted close, display
      its session date, and never enter score, confidence, valuation, or
      recommendation arithmetic.
- [ ] Under $10 is labeled as a speculative watchlist with a 0%
      new-allocation cap; it is excluded from new opportunity highlights and
      sample construction without removing existing holdings or rewriting
      frozen historical baskets.
- [ ] Current opportunity display/filtering uses `LatestMarketData`, but a
      run-dated sample basket classifies the immutable analysis reference
      close at the run target. Later mutable closes cannot change archive and
      rebuild composition, and missing current USD state fails closed for
      promotion.
- [ ] Under-$10 long-horizon output says `Forecast unavailable` and separates
      released reusable foundations from unreleased activation controls.
      Point-in-time SEC adverse-versus-missing behavior, long-v2
      diluted-share/per-share continuity assessment, and the deterministic
      3y/5y engine are foundations only, not candidate approvals. Dedicated
      solvency/cash-runway, Under-$10 dollar-liquidity, and verified
      split/reverse-split controls remain unreleased; joint review and
      candidate-specific eligibility remain required.

## Tracked contributions and allocation plans

- [ ] External deposits, confirmed plan executions, purchases, and manual
      performance baselines are immutable and idempotent where a request can
      be retried. Cash cannot be changed through an unrelated form/admin save.
- [ ] Preview is side-effect free. Confirmation locks/reloads the portfolio,
      holdings, every valued/purchased market row, and qualifying analysis,
      then recomputes the complete plan hash before writing anything.
- [ ] The plan hash binds settings, cash, quantities, prices, source asset
      UUIDs/checksums/session dates, and the exact satellite
      analysis/run/configuration/code/evidence criteria. A same-session
      analysis replacement invalidates an older preview.
- [ ] Allocation uses total NAV including cash, targets 70% SPY and 30%
      non-SPY without selling, selects at most one explicitly short-horizon
      qualified stock, preserves the Under-$10 0% gate, rounds down for the
      selected share mode, and carries every unspent amount.
- [ ] A recorded purchase is labeled as local research bookkeeping at a
      persisted close, not a broker fill. Preview/confirmation makes no
      provider or brokerage call.
- [ ] Contribution profit/loss reconciles cash and quantities to immutable
      boundaries, deposits, and purchases. Boundary/current valuations require
      fresh non-future one-session provider evidence with compatible
      split-adjusted/dividend treatment and no unresolved corporate action.
- [ ] Manual quantity changes/removals append an immutable post-change
      baseline rather than becoming return. If valuation is unavailable, the
      change remains recoverable through an explicit unavailable-boundary
      record and performance is withheld; an ordinary scheduled snapshot
      alone cannot reset the boundary.
- [ ] Split warnings persist across repeated snapshots while quantity is
      unchanged. A supported quantity correction creates a visible new
      baseline; history is never retroactively rewritten.
- [ ] Contribution percentage is labeled as a simple since-boundary return,
      not time-weighted or money-weighted performance, and all-time deposits
      remain distinguishable from flows after the active boundary.

## Return and FX consistency

- [ ] Return calculations use one consistent price basis; no unit mismatch
      (e.g. price vs. adjusted price) within one calculation.
- [ ] FX conversions resolve each valued date against that date's own
      availability cutoff, so a later correction cannot change how an earlier
      date was priced; no implicit or mixed-currency arithmetic.
- [ ] A rate published after the valued date is refused for every run; a
      merely later-*retrieved* source asset is accepted only for an
      explicitly research-grade reconstruction.
- [ ] Carry across weekends/holidays is bounded (0 to 7 calendar days,
      tightenable but never widenable) and recorded per converted date; a
      missing, over-stale, or ambiguous rate path fails the run instead of
      converting part of a panel.
- [ ] A holding whose market is closed keeps its currency exposure: the last
      native quote is revalued at the current rate rather than carrying a
      frozen conversion, in end-of-day valuation *and* in pre-trade rebalance
      sizing.
- [ ] FX coverage is proven for every accounted date and non-base currency
      before any value is computed; an uncovered date fails the run instead
      of reporting a return from a stale conversion.
- [ ] A converted run executes on closing prices only; opening-price bases
      are rejected while FX availability is resolved to end-of-day.
- [ ] Every simulation records its explicit base currency, the native
      currencies converted, and the exact FX frame used, and keeps native
      prices beside converted values in its persisted inputs.
- [ ] The reproducibility hash covers the native-currency assignment and
      retained conversion inputs, not only the converted prices.
- [ ] A reported stock-versus-FX split is exact by construction, or withheld
      with a stated reason; it is never an estimate presented as measured.
- [ ] A selected portfolio cannot silently omit a holding that lacks an
      inception execution price or redistribute its allocation.

## Difficult corporate events

- [ ] Splits, mergers, delistings, and ticker/listing changes are handled
      explicitly in outcome/backtest evaluation
      (`PredictionOutcome.status == corporate_event` or equivalent), not
      silently folded into a normal return calculation.
- [ ] A delisted or changed listing does not silently disappear from
      historical results without an explicit resolution/status.

## Simulation reproducibility

- [ ] Holdings and trades use permanent listing UUIDs, never ticker text as
      identity.
- [ ] Every completed run identifies checksummed immutable price, signal,
      benchmark, and (when converted) FX inputs as well as its result asset.
- [ ] The simulation input hash covers complete canonical inputs and the
      explicit calendar; materially different paths cannot collide merely
      because dimensions and sums match.

## Methodology

- [ ] A scoring/recommendation/risk/scenario methodology change is
      reproducible from `model_version`, `config_hash`, and `code_revision`.
- [ ] A forecast calculation payload identifies its immutable panel or peer
      asset, evidence role/grade, price provider, exact support statistics,
      selected metric branch, fact/accession lineage, and fixed formula inputs.
- [ ] A methodology change is flagged as material and routed through the
      three-pass simplifier gate (see `.github/agents/README.md`).
- [ ] A frozen version's full output contract -- config bytes/effective
      hash, eligibility, reason wording, and both successful and withheld
      calculation/scenario payloads -- is proven byte-for-byte unchanged by a
      differential base/head reproduction test; a stricter eligibility gate
      ships as a new version rather than mutating the frozen one, and its
      default config asset is tracked and hash-pinned. Same-revision tests
      comparing old/new versions do not substitute for base/head reproduction.
- [ ] An advisory prediction whose scenario returns are all null is treated
      as non-evaluable: evaluation resolves it before any price lookup, and
      it (and any malformed legacy row in the same state) is defensively
      excluded from advisory denominators/reporting, independent of decision
      BUY/AVOID/HOLD success semantics.
- [ ] Evidence that disqualifies or withholds a prediction is recorded as
      assessed (`assessed_through`/an explicit incompatible-or-unverified
      status naming the disqualifying source facts), never as
      `verified_through` or a claimed corporate action.
