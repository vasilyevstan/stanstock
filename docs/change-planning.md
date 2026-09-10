# Change planning

This is StanStock's canonical policy for proportional pre-action planning.
Agent identities, chains, and statuses remain defined in
`.github/agents/README.md`.

**Policy revision:** `change-planning@rev-2`

This policy applies prospectively. Released work, including the Under-$10
shadow diagnostics and the September 9 scheduled refresh, must not be
retroactively described as pre-action approved.

## Proportional gate

- **Ordinary work** uses the ordinary agent chain and requires no planning
  artifact.
- **Data or quantitative work** requires a concise, developer-owned
  pre-action check in the existing handoff and pull request before the first
  edit.
- **Material work** requires a complete architect contract followed by three
  sealed independent simplifier passes from distinct model families and one
  matching-revision synthesis before the first edit.

Filled planning records belong only in existing handoffs and pull requests.
Do not add a root plan, per-slice plan file, generated planning artifact, task
database, or YAML planning schema. The existing YAML block in the agent README
is transport for agent handoffs, not a planning schema.

Every filled record names its exact `contract_revision`; ordinary work uses
`N/A: ordinary change; no planning artifact required`. A data/quant revision
identifies its short check. A material revision identifies the architect
contract and must match all three sealed passes and the synthesis.

## Required contract and tabletop

The data/quant short check records the applicable rows below. A material
architect contract records all applicable rows, dependencies, file ownership,
and acceptance criteria. Evidence is executable or cited from the repository;
memory or a self-consistent payload is not proof.

| Area | Required record or check | Acceptance evidence |
|---|---|---|
| Agent routing | Link the applicable six-agent chain and use no seventh planning role or new status. | The handoff names the existing owners and gates. |
| Classification and timing | State ordinary, data/quant, or material before the first edit. | Ordinary has no planning artifact; data/quant has its short check; material has the architect contract, three eligible sealed passes, and synthesis. |
| Record location | Keep filled instances in handoffs and pull requests only. | No root/per-slice plan, generated artifact, task database, or YAML planning schema is added. |
| Producers | Enumerate every producer of each affected value, identity, or decision. | File/symbol references account for all production paths. |
| Selectors and readers | Enumerate every selector, reader, fallback, replay, and fail-closed path. | Search evidence and tests cover alternate and historical reads. |
| Writers and persistence | Enumerate writers, persisted fields, immutable assets, and update boundaries. | The contract distinguishes append-only evidence from mutable current state. |
| Rendered fields | Enumerate API, admin, template, status, and user-visible projections. | Each rendered claim traces to the accepted reader and evidence boundary. |
| Consumers | Enumerate downstream calculations, rankings, reports, jobs, exports, and tests. | Repository-wide `rg` or equivalent search is cited and every result is accounted for. |
| Authoritative identity | Name an independent authoritative owner for every identity-sensitive comparison. | A produced value, copied field, or recomputed checksum never authenticates itself. |
| Adversarial evidence | Cover applicable wrong-identity, stale/future, missing/malformed, retry/no-op, concurrency, and confidentiality cases. | Name executable evidence for each category or use `N/A: factual reason`. |
| Frozen contracts | Identify frozen hashes, eligibility, reason wording, and successful and withheld payloads. | Base/head differential evidence proves the frozen behavior remains unchanged. |
| Drift | Record requirement, dependency, identity-owner, consumer, or acceptance-evidence drift. | Pause only the affected slice; update a data/quant check or increment and repeat a material contract gate. |
| Implementation defects | Return defects that remain inside the accepted contract to the owning implementation context. | No duplicate agent or unnecessary re-planning is introduced. |
| Prospective adoption | State that the planning record predates the first edit. | No released work is relabeled as pre-action approved. |
| Test and provider boundary | Use synthetic fixtures and existing repository-native tools. | No live provider is contacted, and this policy alone mandates no browser framework, workflow, or verifier. |
| Ordinary tabletop | Apply the ordinary chain to a bounded non-data change. | `N/A: ordinary change; no planning artifact required`. |
| Data/quant tabletop | Apply a provider, as-of, migration, scoring, outcome, or simulation change. | The developer-owned short check exists before editing and covers applicable contract rows. |
| Material tabletop | Apply an architecture, schema, security, scoring, or methodology decision. | Exact matching revisions bind architect, three sealed distinct-family passes, synthesis, and implementation. |
| Identity tabletop | Challenge a plausible value carrying its own checksum or copied identity. | The independently owned identity and replay path decide acceptance. |
| Consumer tabletop | Add a consumer outside the changed module. | Repository-wide search finds it and the contract accounts for it before editing. |
| Critical-flow tabletop | Apply the acceptance subsection below when one critical flow spans all four named layers. | One repository-native end-to-end contract test spans the complete flow. |
| Scheduled/provider tabletop | Apply the acceptance subsection below to refresh, retry, recovery, or no-op behavior. | Target records and registered assets are proved separately from process status, and reuse proves zero additional provider fetches. |
| Factual-N/A tabletop | Mark a category inapplicable only when the reason is specific and reviewable. | Bare `not applicable`, `none`, or an omitted category is insufficient. |
| Static-governance tabletop | Evaluate concurrency and retry for a documentation-only governance edit. | `N/A: static policy text has no runtime concurrency or retry behavior`. |

## Applicability-specific acceptance evidence

- A critical flow that crosses persistence, service or calculation,
  fail-closed reading, and rendered UI must name at least one applicable
  repository-native end-to-end contract test spanning all four layers.
  Separate unit-test collections do not substitute for that contract test.
- Scheduled or provider work must separately prove the expected local
  target-date records and the identity and integrity of their registered
  assets. Exit codes, process completion, stage status, or `JobRun.status`
  alone are insufficient. Retry or no-op evidence must identify the exact
  reused records and assets, prove that no additional provider fetch occurred,
  and explain why none was expected.

## Drift and ownership

Drift pauses only the affected slice. Data/quant work updates its short check
before continuing. Material work increments the contract revision and repeats
the architect and sealed-simplifier gate. An implementation defect that does
not alter the accepted contract returns to the same developer context.

The bounded developer agent owns application implementation paths, not
governance paths. When a slice changes only governance-controlled files, the
explicitly designated human/orchestrator supplies the implementation handoff
and records `N/A: out-of-scope governance ownership` for the bounded
developer slot; it does not fabricate an `IMPLEMENTED_LOCAL` status.
