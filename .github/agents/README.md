# StanStock reusable agent team

Six reusable agents provide an independent-review workflow for StanStock
changes. They complement `CONTRIBUTING.md`; they do not replace human release
or merge authority.

## Agents

| Agent | Role |
|---|---|
| `stanstock-architect` | Read-only. Turns an accepted request into a bounded, dependency-ordered implementation contract. |
| `stanstock-simplifier` | Read-only. One pass for ordinary changes; three sealed distinct-model passes plus synthesis for material changes. |
| `stanstock-developer` | Bounded full-stack editor (`src/`, `templates/`, `static/`, `tests/`). No git or deploy authority. |
| `stanstock-research-integrity` | Read-only. Reviews provider rights/provenance, as-of/look-ahead correctness, methodology, outcomes, and simulations. |
| `stanstock-critic-tester` | Read-only. Adversarial diff review plus test execution, using the security/UX/contracts/operations checklists. |
| `stanstock-final-validator` | Read-only. Milestone/release acceptance across the full chain. |

## Tiered chains

Pick the chain by change type. Each arrow is a completed gate handing off to
the next; do not skip a required gate.

**1. Ordinary change** (templates, view logic, non-data bug fixes, docs-only
code changes):

```
stanstock-developer -> stanstock-critic-tester
```

**2. Data or quantitative change** (new/changed model or migration, provider
integration, scoring/recommendation/risk/scenario logic, outcome/backtest/
simulation logic, as-of or point-in-time handling):

```
stanstock-developer -> stanstock-research-integrity -> stanstock-critic-tester
```

**3. Material decision** (new or changed architecture, schema, security
boundary, scoring/recommendation methodology, or simulation methodology):

```
stanstock-architect
  -> stanstock-simplifier (3 sealed independent passes + synthesis)
  -> stanstock-developer
  -> stanstock-research-integrity
  -> stanstock-critic-tester
  -> stanstock-final-validator
```

**Final validator** also runs standalone at any milestone or release
boundary to confirm the accumulated chain evidence is complete, even when no
single change triggered chain 3.

### What makes a change "material"

Any of: a new or materially changed Django model/migration; a new provider
integration or change to `ProviderRecord` gating; an authentication, session,
or CSRF-relevant change; a change to scoring, recommendation, risk, or
scenario computation; a change to outcome evaluation or simulation
methodology. The architect (or the orchestrator, absent an architect pass)
makes this call and states it explicitly in the handoff.

## Compact handoff fields

Every agent-to-agent handoff includes at minimum:

```yaml
slice_id: <stable-kebab-id>
from_agent: <agent-name>
to_agent: <agent-name>
status: <agent's exact namespaced status>
change_class: ordinary|data-quant|material
base_sha: <exact-or-"no commits yet">
head_sha: <exact-or-null>
files_changed: []
findings:
  open: []
  resolved: []
tests:
  commands: []
  exit_codes: []
risks: []
```

Never put secrets, `.env` values, provider credentials, private financial
data, or session/workspace paths in a handoff.

## Correction-in-same-context rule

A finding returns to the agent conversation that owns the affected gate (the
developer for an implementation defect, the architect for a design gap). Do
not spawn a duplicate agent or a new gate to re-litigate the same finding;
corrections stay inside the originating agent's context until resolved or the
owner is genuinely unavailable.

## Exact revision evidence

Every handoff and every terminal status (`APPROVE_SLICE`,
`INTEGRITY_APPROVED`, `READY_FOR_RELEASE`, etc.) cites the exact base and head
SHA (or explicitly "no commits yet" before the first commit). A stale,
branch-name-only, or SHA-less claim is not evidence and must be treated as
incomplete.

## No self-approval

No agent approves its own prior output. `stanstock-developer` cannot issue
`APPROVE_SLICE` or `INTEGRITY_APPROVED`; `stanstock-architect` cannot issue
`SIMPLIFICATION_READY`; the agent that implemented a slice cannot be the one
that validates it for release. Each gate's approval must come from the next
distinct agent in the chain.

## Protected research invariants

Every applicable gate preserves actual generation/retrieval timestamps
separately from historical `data_cutoff`, rejects late signals from observed
backtests, converts currencies only through point-in-time FX resolved against
each valued date's own availability cutoff and into one explicit base currency
(failing on missing, stale, or ambiguous rate paths), and hashes complete
canonical simulation inputs rather than aggregate summaries. These safeguards
are correctness requirements, not optional complexity.

## No secrets, private data, or session paths

No agent may print or persist: `.env` values, `DJANGO_SECRET_KEY`,
`DATABASE_URL` credentials, provider API keys, real user financial/portfolio
data, or local session/workspace file paths. Use synthetic fixtures in every
example and test.

## Git and deploy authority

No agent in this chain stages, commits, pushes, opens/merges a pull request,
dispatches a workflow, deploys, or mutates production/data state. Only the
orchestrator (the human, or an explicit orchestrator role the user has
designated) performs git actions, and only when the user has explicitly
given it that ownership for the current task. Agents hand off text evidence;
they do not act on the repository beyond their stated read/edit scope.

## Status vocabulary

| Agent | Statuses |
|---|---|
| Architect | `ARCHITECTURE_READY`, `ARCHITECTURE_CHANGES_REQUIRED`, `DECISION_REQUIRED` |
| Simplifier (ordinary) | `SIMPLIFICATION_READY`, `SIMPLIFICATION_DISPUTED` |
| Simplifier (independent pass) | `SIMPLIFICATION_PROPOSED`, `NO_SIMPLIFICATION_FOUND`, `BLOCKED` |
| Simplifier (synthesis) | `SIMPLIFICATION_READY`, `SIMPLIFICATION_DISPUTED`, `SIMPLIFICATION_INCOMPLETE` |
| Developer | `IMPLEMENTED_LOCAL`, `BLOCKED` |
| Research integrity | `INTEGRITY_APPROVED`, `INTEGRITY_CHANGES_REQUIRED` |
| Critic-tester | `APPROVE_SLICE`, `CHANGES_REQUIRED` |
| Final validator | `READY_FOR_RELEASE`, `NO_GO` |

Status lines are namespaced with the agent name, e.g.
`stanstock-critic-tester: APPROVE_SLICE`. No status is merge or deploy
approval by itself.

## Checklists

`stanstock-research-integrity` and `stanstock-critic-tester` apply the
checklists under `docs/review-checklists/`:

- `research-integrity.md` — provenance, as-of, methodology, outcomes.
- `security.md` — auth, CSRF, secrets, input handling.
- `ux.md` — responsive/accessible tables and charts, honest rendering.
- `contracts.md` — migration and interface compatibility.
- `operations.md` — idempotent jobs, backups, deployment safety.
