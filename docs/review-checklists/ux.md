# UX review checklist

Applies to any change touching templates, views that render data, or
user-facing tables/charts. Reviewer: `stanstock-critic-tester`.

## Navigation and information hierarchy

- [ ] Exercise the user's entry-point journey, not only direct URLs. Primary
      destinations retain recognizable names, and critical filtered views
      are discoverable in the first viewport on narrow and desktop screens.
- [ ] Overview pages are bounded and scannable; filters, selected horizons,
      and pagination remain visible and coherent across navigation.
- [ ] Actual results appear in the initial viewport rather than below a
      large heading, preamble, or control stack. Row/DOM counts alone do not
      prove usable density; inspect content positions and readable values.
- [ ] Shortlist acceptance covers both genuinely eligible candidates and
      realistic empty groups. Measure the first actual candidate after empty
      sections and restrictions, not the position of a heading or placeholder.
- [ ] Primary labels remain intact, secondary menus do not open over content
      by default, and form submission preserves the selected view and filters.
- [ ] Technical detail can be collapsed, but the summary of a restriction,
      failure, adverse comparison, or unavailable result stays visible.
- [ ] Method inapplicability, unavailable inputs, non-observed evidence,
      pending outcomes, and non-evaluable forecasts are not presented as one
      generic insufficiency state.

## Responsive and accessible tables

- [ ] Data tables (analyses, predictions, holdings, trades) remain readable
      and usable on narrow viewports; no required column is clipped or
      hidden without an explicit, discoverable affordance.
- [ ] Tables use semantic markup (`<table>`, `<th>`, scoped headers) so
      screen readers can associate values with column/row labels.
- [ ] Sort, filter, and pagination controls are keyboard-reachable with
      visible focus states.
- [ ] Shared layout changes cover local and Linux/wider-font rendering,
      narrow wrapping and actual visible values. Fewer cards, menus or hidden
      sections alone prove neither usable density nor reduced computation.

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
- [ ] Rejected forms offer an explicit safe GET recovery action, never a
      repeat POST or a GET to a POST-only action. A stale login form recovers
      its safe destination through a fresh form on desktop and mobile.
