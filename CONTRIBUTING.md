# Contributing

## Workflow

- Work on a focused branch and open a pull request into protected `main`.
- Use specific, outcome-focused commit subjects and pull-request titles; avoid
  vague labels such as "updates", "changes", or "fixes".
- Complete every applicable pull-request template section with enough detail
  to explain what changed, why it was needed, and how it works. Include exact
  revision and validation evidence, compatibility and data/methodology
  impact, security/operations impact, risks and rollback, and the required
  agent-chain results. Use `N/A: <factual reason>` instead of omitting an
  inapplicable field, and update stale evidence before merge.
- Do not commit provider keys, market-data downloads, private prediction
  history, database dumps, or session artifacts.
- Keep the application runnable locally and use synthetic fixtures in tests.
- Treat the documented price-provider gate as `NO_GO` until a lawful,
  automation-stable source is explicitly approved. Never bypass a browser
  challenge or substitute an unofficial scraper.
- Use the tiered agent workflow in `.github/agents/README.md`.
- Follow the proportional pre-action contract in
  `docs/change-planning.md`.
- Governance paths are owned by the explicitly designated human/orchestrator,
  not the bounded developer agent.
- Apply the planning contract prospectively; do not relabel released work,
  including the Under-$10 diagnostics or September 9 refresh, as pre-action
  approved.
- Run the smallest relevant checks during development and `make check` before
  release review.

## Architecture

- Keep StanStock a Django modular monolith.
- Preserve the provider boundary and the controlled point-in-time data access
  path.
- Do not add a second API framework, query engine, client build chain,
  scheduler service, distributed queue, or fitted model without an accepted
  requirement and simplification review.
- Missing data remains missing; do not convert it to zero or success-shaped
  output.
