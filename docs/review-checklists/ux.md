# UX review checklist

Applies to any change touching templates, views that render data, or
user-facing tables/charts. Reviewer: `stanstock-critic-tester`.

## Responsive and accessible tables

- [ ] Data tables (analyses, predictions, holdings, trades) remain readable
      and usable on narrow viewports; no required column is clipped or
      hidden without an explicit, discoverable affordance.
- [ ] Tables use semantic markup (`<table>`, `<th>`, scoped headers) so
      screen readers can associate values with column/row labels.
- [ ] Sort, filter, and pagination controls are keyboard-reachable with
      visible focus states.

## Responsive and accessible charts

- [ ] Charts (scenario bands, backtest equity curves, price history) have a
      text/table fallback or accessible summary; information is not
      conveyed by color alone.
- [ ] Chart containers reflow instead of overflowing or clipping on mobile
      widths.

## Honest data presentation

- [ ] Missing, insufficient, or low-confidence data is visibly distinct from
      a normal value (e.g. `confidence_status`, `insufficiency_reason`
      surfaced in the UI) — never rendered as if it were a normal zero or
      score.
- [ ] Research-grade vs. observed history is labeled when both can appear in
      the same view.
- [ ] Forecast/scenario values are labeled as estimates, consistent with the
      README disclaimer; nothing implies a guaranteed outcome.

## Interaction stability

- [ ] Live or periodic updates do not steal focus, reset an in-progress
      input, or unexpectedly move an active control.
- [ ] Error and empty states give a clear, honest message rather than a
      blank or misleading table/chart.
