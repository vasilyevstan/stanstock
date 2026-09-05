# Contributing

## Workflow

- Work on a focused branch and open a pull request into protected `main`.
- Do not commit provider keys, market-data downloads, private prediction
  history, database dumps, or session artifacts.
- Keep the application runnable locally and use synthetic fixtures in tests.
- Treat the documented price-provider gate as `NO_GO` until a lawful,
  automation-stable source is explicitly approved. Never bypass a browser
  challenge or substitute an unofficial scraper.
- Use the tiered agent workflow in `.github/agents/README.md`.
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
