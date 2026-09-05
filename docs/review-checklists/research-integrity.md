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
- [ ] Migration/backfill code does not assign historical cutoffs that the
      legacy calculation path cannot prove; ambiguous legacy runs remain
      conservative or are rejected.
- [ ] Outcome horizons count distinct observed sessions; weekends, holidays,
      and duplicate dates are not treated as extra sessions.

## Research-grade vs. observed history

- [ ] `UniverseSnapshot.grade` (`research` vs `observed`) is preserved end to
      end; reconstructed history is never silently merged with live-captured
      membership.
- [ ] Consumers (views, exports, reports) are told which grade they are
      reading when it affects interpretation.

## Missing values

- [ ] Missing or insufficient data is represented with an explicit flag
      (`insufficiency_reason`, `quality_flags`, `confidence_status`), never
      coerced to zero, `None`-as-zero, or a default/success-shaped score.
- [ ] A low-confidence or insufficient result is visibly distinguishable from
      a high-confidence one downstream (admin, view, export).

## Return and FX consistency

- [ ] Return calculations use one consistent price basis; no unit mismatch
      (e.g. price vs. adjusted price) within one calculation.
- [ ] FX conversions use the matching `base_currency`/`quote_currency` pair
      and observation date; no implicit or mixed-currency arithmetic.
- [ ] Until FX conversion is implemented, every simulation rejects
      mixed-currency selections and records its one native currency.
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
- [ ] Every completed run identifies checksummed immutable price, signal, and
      benchmark inputs as well as its result asset.
- [ ] The simulation input hash covers complete canonical inputs and the
      explicit calendar; materially different paths cannot collide merely
      because dimensions and sums match.

## Methodology

- [ ] A scoring/recommendation/risk/scenario methodology change is
      reproducible from `model_version`, `config_hash`, and `code_revision`.
- [ ] A methodology change is flagged as material and routed through the
      three-pass simplifier gate (see `.github/agents/README.md`).
