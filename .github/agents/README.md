# StanStock reusable agent team

Six reusable agents provide an independent-review workflow for StanStock
changes. They complement `CONTRIBUTING.md`; they do not replace human release
or merge authority.

## Agents

| Agent | Role |
|---|---|
| `stanstock-architect` | Read-only. Turns an accepted request into a bounded, dependency-ordered implementation contract. |
| `stanstock-simplifier` | Read-only. Required for material changes as three sealed distinct-model passes plus synthesis; optional user-invoked simplification is outside the ordinary chain. |
| `stanstock-developer` | Bounded full-stack editor (`src/`, `templates/`, `static/`, `tests/`). No git or deploy authority. |
| `stanstock-research-integrity` | Read-only. Reviews provider rights/provenance, as-of/look-ahead correctness, empirical forecast panels, SEC fact semantics, methodology, outcomes, and simulations. |
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
contract_revision: <exact-or-"N/A: factual reason">
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

`docs/change-planning.md` is the canonical proportional pre-action planning
policy. It defines when `contract_revision` is required, where filled evidence
lives, and how drift is handled. Ordinary work has no planning artifact.

Never put secrets, `.env` values, provider credentials, private financial
data, or session/workspace paths in a handoff.

## Correction-in-same-context rule

A finding returns to the agent conversation that owns the affected gate (the
developer for an implementation defect, the architect for a design gap). Do
not spawn a duplicate agent or a new gate to re-litigate the same finding;
corrections stay inside the originating agent's context until resolved or the
owner is genuinely unavailable.

## Orchestrator stall detection

The orchestrator remains accountable for the requested result across every
handoff, wait, CI run, release gate, merge, and local verification. Waiting is
not a terminal status.

Treat each of these as an actionable stall signal:

- an idle or completed agent has unread output;
- a running agent stops making tool progress or repeatedly returns no usable
  evidence;
- the same finding survives another correction cycle;
- an approval or test result names a stale SHA or worktree fingerprint;
- CI, a deployment/environment gate, or a CLI-owned approval is pending
  without inspection;
- a dirty development worktree approaches an operational deadline such as
  the next scheduled refresh, which requires a clean committed revision.

On a stall signal, inspect the current repository and task ledger, record the
specific blocker, and take the smallest progress-making action: read the
completed result, return the defect to its owning agent, send a focused
follow-up, continue independent work, or take over directly. Start a
replacement agent only when the original owner is unavailable or has failed;
preserve the exact handoff and fingerprint when doing so. Never claim
completion, abandon the chain, or poll a background agent repeatedly while
there is independent work to do.

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

No reissue of an immutable prediction inherits another version's on-time
status. Each version, including a same-target reissue, is checked
independently against its own next-market-session-open deadline from
cutoff-safe evidence: a reissue created before that deadline may still be
observed. An unsafe explicit observed request raises; a separate non-observed
reconstruction is research-grade. Aggregate reporting counts the earliest
reportable prediction once per exact listing/target/horizon/evidence-role/
method/config/provider observation, while later valid observed reissues remain
immutable ledger rows evaluated per version. `manage.py analyze` is
demo/research tooling, not the live-US or observed-reissue interface: it
explicitly requests `issued_on_time=False` for every target, regardless of its
provider/config/benchmark flags or `STANSTOCK_CODE_REVISION`. Only an
exceptional direct
`analyze_snapshot(..., issued_on_time=True, ...)` service-level call can
request an observed same-target reissue, and only once it has independently
reproved the next-market-session-open deadline; that call must also
explicitly bind the reviewed production scoring config, `provider=
"twelve_data"`, the reviewed benchmark (currently SPY), and the exact
committed `STANSTOCK_CODE_REVISION`.

A frozen methodology/config version's contract covers its config bytes/
effective hash, eligibility, reason wording, and both successful and withheld
calculation/scenario payloads. A stricter eligibility gate is a new version,
proven with differential base/head reproduction tests showing the frozen
version's hash, behavior, and payloads are unchanged. Evidence that
disqualifies a prediction is recorded as assessed (`assessed_through`/an
explicit incompatible-or-unverified status citing the disqualifying source
facts), never as `verified_through` or a claimed corporate action. An
advisory prediction whose scenario returns are all null is non-evaluable:
evaluation resolves it before any price lookup, and it is excluded from
advisory denominators; this is separate from decision BUY/AVOID/HOLD success
semantics.

Forecast work additionally keeps score groups separate from persisted
forecast identities, labels advisory evidence independently from decision
evidence, uses cumulative price-return units consistently, and prevents
advisory 6m/12m/3y/5y results from changing recommendations, opportunity
highlights, or headline decision hit rates. Empirical panels must use
cutoff-safe immutable inputs and overlap-aware support; SEC facts must preserve
full filing/period identity and exact availability; incompatible split/share
bases and unsupported long-formula inputs are withheld rather than guessed.

Prospective affordability work must also be price-scale invariant: nominal
share price never raises research conviction, price-difference momentum is
normalized, and liquidity gates use compatible dollar volume rather than raw
share counts. Split-equivalent price/volume transformations must preserve
scores and recommendations. Corrections use a new versioned configuration;
historical hashes, predictions, and performance cohorts remain separate.
Current USD price bands remain display/filter/execution metadata and show
their session date. Under $10 is a 0%-new-allocation speculative watchlist,
not a positive signal; it cannot enter highlights or newly constructed sample
baskets. Its disclosure separates released reusable foundations from
unreleased activation controls: foundations are not candidate approvals, and
joint review plus candidate-specific eligibility remain required. Immutable
long-horizon ledger evidence stays visible with its original horizon and
evidence role, labeled only with the current activation context. Sample
construction classifies the decision-run reference close rather than mutable
current market state. Existing holdings and frozen historical baskets are
never rewritten, and missing current USD price state fails closed for new
promotion.

Investable ETF support preserves a separate evidence contract. A benchmark ETF
reuses its single immutable provider series and credit, remains outside stock
universe membership, and cannot enter stock analysis, fundamentals,
predictions, opportunities, or sample-stock baskets. ETF metrics retain their
price-return/dividend basis, while personal portfolio valuation may use the
supported ETF's current market row without inventing a stock rating.

Contribution planning is also an evidence boundary. Deposits, confirmed
purchases, and manual performance baselines are immutable; previews have no
side effects. Confirmation recomputes one hash over locked portfolio state,
market assets/sessions, and exact short-horizon qualification evidence.
Manual quantity changes restart measurement from a visible post-change
baseline, while unavailable valuation records withhold performance without
blocking recovery. Admin and web saves must not write stale ledger-managed
cash or quantities.

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
| Simplifier (optional user-invoked) | `SIMPLIFICATION_READY`, `SIMPLIFICATION_DISPUTED` |
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
