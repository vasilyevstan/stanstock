"""Execute the exact pre-slice source of `stanstock.research.outcomes`.

`refresh-outcome-verification@rev-2` mechanically extracts a pure
`resolve_outcome`/`ResolvedOutcome` seam out of the base revision's single
`evaluate_prediction` function. Proving the extraction is byte-identical in
behavior needs the *actual* base module executing, not the head module with
the seam bypassed -- so this reads the base bytes straight out of the git
object database and executes them in a throwaway module namespace, following
the same idiom as `tests/base_service.py`.

**Why only `research/outcomes.py` itself is bound (unlike `base_service.py`'s
multi-module dependency chain):** base `outcomes.py` imports exactly four
`stanstock.*` modules -- `stanstock.data.asof` (`AsOfData`,
`PriceFrameSchemaError`), `stanstock.data.assets` (`AssetStore`),
`stanstock.data.models` (`DataAsset`), and `stanstock.research.models`
(`Prediction`, `PredictionOutcome`, `Recommendation`). Every line this slice
(and everything between base and head) has ever changed in those four files
is a pure insertion relative to base: no base line was deleted or modified,
only new top-level functions/methods were appended after it. A base
`outcomes.py` bound alongside the *live* (head) versions of those four
modules therefore sees exactly the same classes, exactly the same method
bodies, and exactly the same behavior base `outcomes.py` shipped with --
newer additions those four files may have gained are simply never reached by
a call into base `outcomes.py`'s own unchanged code paths.
`test_dependency_modules_are_pure_insertions_since_base` in
`tests/test_research_outcome_refresh_validation.py` proves this mechanically
via `require_pure_insertion_since_base` below, rather than resting on this
paragraph alone: every pre-existing base top-level statement (represented by
its full `ast.dump(node, include_attributes=False)` -- covering decorators,
bases, defaults, imports, and assignment structure, not merely a name) must
have a semantically identical statement in head, in the same relative order.
A line-based diff (e.g. `difflib.SequenceMatcher`) cannot prove this -- an
edit *inside* an existing function's body, a changed decorator, or a
retargeted import, each with no surrounding line deleted, is reported as a
bare `insert` opcode and would pass unnoticed. A genuinely new head
statement is only accepted if it binds solely new, non-rebinding,
non-duplicate names that are explicitly listed in
`SAFE_TOP_LEVEL_ADDITIONS` for that file; an unbound top-level statement
(anything with no nameable binding: a wildcard import, a compound/tuple
assignment, a bare expression, `if`/`try`/`with`/loop, etc.) can never be a
freely admitted addition -- it must already exist, unchanged, in base. If
this ever finds a changed, removed, reordered, or unlisted-new base binding
in any of the four files, this single-module binding design no longer
holds and `PURE_INSERTION_DEPENDENCY_PATHS`/`SAFE_TOP_LEVEL_ADDITIONS`
below must be revisited.

Nothing on disk is modified and no base file is copied into the working
tree. A shallow or object-pruned checkout simply has no base objects, in
which case `base_outcomes_available()` is ``False`` and the caller skips
rather than silently comparing the working tree against itself.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

#: The revision `fix/scheduled-refresh-verification` branched from.
BASE_SHA = "c8b2e20ff167fab8cf18684bfebcf2f757644b3d"

REPO_ROOT = Path(__file__).resolve().parents[1]

OUTCOMES_MODULE = "stanstock.research.outcomes"
OUTCOMES_PATH = "src/stanstock/research/outcomes.py"

#: The four `stanstock.*` modules base `outcomes.py` imports, whose base and
#: head bytes must be proven pure-insertion relative to each other for the
#: single-module binding above to be sound. See module docstring.
PURE_INSERTION_DEPENDENCY_PATHS: tuple[str, ...] = (
    "src/stanstock/data/asof.py",
    "src/stanstock/data/assets.py",
    "src/stanstock/data/models.py",
    "src/stanstock/research/models.py",
)

#: The exact new top-level names each dependency file has gained since
#: `BASE_SHA` (computed once from the real base/head diff, not derived at
#: runtime) -- the only additions `require_pure_insertion_since_base` may
#: admit; anything else new is an unlisted, and therefore rejected, change.
SAFE_TOP_LEVEL_ADDITIONS: dict[str, frozenset[str]] = {
    "src/stanstock/data/asof.py": frozenset(
        {
            "math",
            "Decimal",
            "InvalidOperation",
            "RefreshVerificationError",
            "VerifiedPriceFields",
            "_finite_positive_close",
            "_bound_volume",
            "raw_price_asset_for",
            "verified_price_fields",
        }
    ),
    "src/stanstock/data/assets.py": frozenset(
        {
            "RefreshVerificationError",
            "AssetRef",
            "asset_ref_for",
            "open_asset_store",
            "read_checksummed_bytes",
            "resolve_asset_ref",
            "verify_catalog_refs",
        }
    ),
    "src/stanstock/data/models.py": frozenset(),
    "src/stanstock/research/models.py": frozenset(),
}


class BaseOutcomesUnavailableError(RuntimeError):
    """The base `outcomes.py` bytes are not readable from the object database."""


class PureInsertionViolationError(ValueError):
    """A dependency's base top-level binding was changed or removed in head.

    Distinguished from an ordinary `AssertionError` so a caller can tell
    "the proof ran and failed" apart from "the proof could not even be
    evaluated" (e.g. a genuinely duplicate/unresolvable top-level binding,
    which is also raised as this same explicit error rather than silently
    comparing only one of the ambiguous candidates).
    """


@dataclass(frozen=True, slots=True)
class BaseOutcomes:
    base_sha: str
    source_sha256: str
    outcomes: ModuleType


def base_outcomes_available() -> bool:
    try:
        _read_base_source(OUTCOMES_PATH)
    except BaseOutcomesUnavailableError:
        return False
    return True


def base_outcomes_checksum() -> str:
    return hashlib.sha256(_read_base_source(OUTCOMES_PATH)).hexdigest()


def read_base_bytes(relative_path: str) -> bytes:
    """Return the exact base-revision bytes for a repo-root-relative path."""
    return _read_base_source(relative_path)


def require_pure_insertion_since_base(relative_path: str) -> None:
    """Raise `PureInsertionViolationError` unless every top-level statement in
    the base revision of `relative_path` is still present, unchanged, and in
    the same relative order in the current working-tree file, and every new
    head-only statement is an explicitly allowlisted, name-bound addition.

    Used by `test_dependency_modules_are_pure_insertions_since_base` to
    mechanically prove the single-module binding assumption stated in this
    module's docstring, rather than resting on the docstring alone.
    """
    base_text = read_base_bytes(relative_path).decode("utf-8")
    head_text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    _require_pure_insertion(
        base_text,
        head_text,
        relative_path,
        allowed_additions=SAFE_TOP_LEVEL_ADDITIONS.get(relative_path, frozenset()),
    )


#: A top-level statement's identity for matching across base/head: `None`
#: for anything with no single nameable binding (wildcard import, compound/
#: tuple assignment, bare expression, `if`/`try`/`with`/loop, ...) -- these
#: can never be treated as a freely admitted new addition, only as an
#: existing statement that must already match one in base exactly.
_NodeKey = tuple[str, tuple[str, ...]] | None


@dataclass(frozen=True, slots=True)
class _TopLevelNode:
    key: _NodeKey
    names: frozenset[str]
    dump: str
    lineno: int


def _require_pure_insertion(
    base_text: str,
    head_text: str,
    label: str,
    *,
    allowed_additions: frozenset[str] = frozenset(),
) -> None:
    base_nodes = _parse_top_level(base_text)
    head_nodes = _parse_top_level(head_text)

    base_by_key: dict[_NodeKey, _TopLevelNode] = {}
    base_bound_names: set[str] = set()
    for node in base_nodes:
        base_bound_names |= node.names
        if node.key is None:
            continue
        if node.key in base_by_key:
            raise PureInsertionViolationError(
                f"{label}: duplicate top-level binding {node.key!r} in base"
            )
        base_by_key[node.key] = node

    head_by_key: dict[_NodeKey, _TopLevelNode] = {}
    for node in head_nodes:
        if node.key is None:
            continue
        if node.key in head_by_key:
            raise PureInsertionViolationError(
                f"{label}: duplicate top-level binding {node.key!r} in head"
            )
        head_by_key[node.key] = node

    # Every base statement must still exist in head, unchanged, and in
    # non-decreasing relative order -- neither removed, edited, nor
    # reordered. Positional identity is tracked via each head node's own
    # object identity (`id`), never by re-deriving a position from `dump`
    # text, since two distinct nodes could otherwise share a dump.
    head_positions = {id(node): index for index, node in enumerate(head_nodes)}
    consumed_unbound_ids: set[int] = set()
    matched_head_ids: set[int] = set()
    last_position = -1
    for base_node in base_nodes:
        if base_node.key is not None:
            head_node = head_by_key.get(base_node.key)
            if head_node is None:
                raise PureInsertionViolationError(
                    f"{label}: top-level binding {base_node.key!r} present in base "
                    "is missing from head"
                )
            if head_node.dump != base_node.dump:
                raise PureInsertionViolationError(
                    f"{label}: top-level binding {base_node.key!r} changed since base "
                    "(not a pure insertion)"
                )
        else:
            head_node = next(
                (
                    candidate
                    for candidate in head_nodes
                    if candidate.key is None
                    and id(candidate) not in consumed_unbound_ids
                    and head_positions[id(candidate)] > last_position
                    and candidate.dump == base_node.dump
                ),
                None,
            )
            if head_node is None:
                raise PureInsertionViolationError(
                    f"{label}: an unbound top-level statement present in base at line "
                    f"{base_node.lineno} is missing, reordered, or changed in head"
                )
            consumed_unbound_ids.add(id(head_node))
        position = head_positions[id(head_node)]
        if position <= last_position:
            raise PureInsertionViolationError(
                f"{label}: top-level statement {base_node.key or '<unbound>'!r} was "
                "reordered relative to base"
            )
        last_position = position
        matched_head_ids.add(id(head_node))

    # Every head statement not matched above is new. It may only be
    # admitted if it binds solely fresh, non-rebinding, explicitly
    # allowlisted names -- never an unbound statement.
    for node in head_nodes:
        if id(node) in matched_head_ids:
            continue
        if node.key is None:
            raise PureInsertionViolationError(
                f"{label}: head has a new unbound top-level statement at line "
                f"{node.lineno}, which is never a permitted addition"
            )
        if node.names & base_bound_names:
            raise PureInsertionViolationError(
                f"{label}: head rebinds an existing base name via {node.key!r}"
            )
        if not node.names <= allowed_additions:
            raise PureInsertionViolationError(
                f"{label}: head adds top-level binding {node.key!r}, which is not on the "
                "explicit safe-addition allowlist"
            )


def _parse_top_level(source: str) -> list[_TopLevelNode]:
    tree = ast.parse(source)
    return [
        _TopLevelNode(
            key=_node_key(node),
            names=_all_bound_names(node),
            dump=ast.dump(node, include_attributes=False),
            lineno=getattr(node, "lineno", -1),
        )
        for node in tree.body
    ]


def _node_key(node: ast.stmt) -> _NodeKey:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return ("def", (node.name,))
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return ("assign", (target.id,))
        return None  # compound/tuple/list assignment target
    if isinstance(node, ast.Assign):
        return None  # `a = b = 1`-style multi-target assignment
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
            return ("assign", (node.target.id,))
        return None
    if isinstance(node, ast.Import | ast.ImportFrom):
        if any(alias.name == "*" for alias in node.names):
            return None  # wildcard import binds no single knowable name
        return ("import", tuple(alias.asname or alias.name for alias in node.names))
    return None


def _all_bound_names(node: ast.stmt) -> frozenset[str]:
    """Every name this statement binds, regardless of whether `_node_key`
    treats it as individually matchable -- used only to detect a *new*
    head statement rebinding a name base already used under a different
    statement shape (e.g. base `import os`, head `def os(): ...`)."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return frozenset({node.name})
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            names |= _assign_target_names(target)
        return frozenset(names)
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return frozenset({node.target.id})
    if isinstance(node, ast.Import | ast.ImportFrom):
        return frozenset(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return frozenset()


def _assign_target_names(target: ast.expr) -> frozenset[str]:
    if isinstance(target, ast.Name):
        return frozenset({target.id})
    if isinstance(target, ast.Tuple | ast.List):
        names: set[str] = set()
        for element in target.elts:
            names |= _assign_target_names(element)
        return frozenset(names)
    raise PureInsertionViolationError(
        f"unsupported top-level assignment target {ast.dump(target)!r}"
    )


def _read_base_source(relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{BASE_SHA}:{relative_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise BaseOutcomesUnavailableError(
            f"Base revision object {BASE_SHA}:{relative_path} is not readable from "
            "the local git object database"
        ) from error


@contextmanager
def base_research_outcomes() -> Iterator[BaseOutcomes]:
    """Temporarily import base `research.outcomes`; restore on exit.

    Only `stanstock.research.outcomes` itself is swapped in `sys.modules`.
    Its own `import` statements resolve against whatever is currently bound
    for `stanstock.data.asof`/`stanstock.data.assets`/`stanstock.data.models`/
    `stanstock.research.models` -- ordinarily the live (head) modules, which
    are provably backward-compatible supersets of base (see module
    docstring and `test_dependency_modules_are_pure_insertions_since_base`).
    """
    original = sys.modules.get(OUTCOMES_MODULE)
    source = _read_base_source(OUTCOMES_PATH)
    try:
        spec = importlib.util.spec_from_loader(
            OUTCOMES_MODULE, loader=None, origin=f"{BASE_SHA}:{OUTCOMES_PATH}"
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(REPO_ROOT / OUTCOMES_PATH)
        sys.modules[OUTCOMES_MODULE] = module
        exec(compile(source, f"<{BASE_SHA}:{OUTCOMES_PATH}>", "exec"), module.__dict__)
        yield BaseOutcomes(
            base_sha=BASE_SHA,
            source_sha256=hashlib.sha256(source).hexdigest(),
            outcomes=module,
        )
    finally:
        if original is None:
            sys.modules.pop(OUTCOMES_MODULE, None)
        else:
            sys.modules[OUTCOMES_MODULE] = original
