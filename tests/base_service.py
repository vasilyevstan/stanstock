"""Execute the exact pre-slice source of `stanstock.research.service`.

The Under-$10 shadow slice adds one nested `data_quality` key inside
`_compute_listing_from_asof`. Proving that nothing else moved needs the
*actual* base revision's service module, not the head module with the
feature switched off, so this reads the base bytes straight out of the git
object database and executes them in a throwaway module namespace.

**Exact bound set and why, stated precisely rather than as a round claim:**

- ``data/asof.py``, ``research/indicators.py``, and
  ``research/medium_forecasts.py`` -- base `research/service.py` directly
  imports all three, and this slice changed all three. Binding `service.py`
  without binding them would let those import statements resolve against
  whatever the *live*, working-tree modules currently are: a deliberate or
  accidental head-only mutation to as-of clipping/normalization, an existing
  indicator output, or medium-panel asset selection would then silently leak
  into the "base" run too, and the differential comparison could never detect
  it. In particular, head's medium-panel builder uses the new atomic
  ``price_frame_with_diagnostics`` API while the pinned base ``AsOfData`` does
  not expose that API, so those two base modules must be bound together.
- ``research/affordability.py`` and ``data/provider_policy.py`` -- base
  `research/service.py`'s own frozen source does **not** import either
  module at all (verified: neither name appears anywhere in the base file);
  both imports are new in head's `research/service.py`, added for the
  Under-$10 shadow feature itself. They are bound here defensively, at zero
  behavioral cost, in case a later correction to this same base-differential
  surface needs it -- not because base's actual source resolves them today.
- ``research/long_forecasts.py`` -- base `research/service.py` directly
  imports it (`LongForecast`, `build_long_forecasts`), and the
  `refresh-output-verification` slice renamed four of its private helpers
  to public names for reuse by `research.refresh_validation` (a pure
  rename; neither imported name nor any behavior changed). Its bytes now
  differ from base for that reason alone, so it must be bound here too --
  otherwise base `service.py`'s own import of it would resolve against the
  live, working-tree module instead of base's own bytes.
- ``research/service.py`` itself is, of course, always bound: it is the
  module under comparison.

`test_dependency_binding_covers_every_changed_module_base_service_imports` in
`tests/test_research_under10_pipeline.py` proves the *first* bullet's claim
mechanically -- by parsing base `research/service.py`'s actual `import`
statements and diffing each imported module's base bytes against the current
working tree -- rather than resting on this paragraph alone. It does not, and
cannot, prove a defensive binding is "needed"; that is a design choice, not a
fact about base's source, and is stated as such above.

Every module this slice changed that base `research/service.py` does not
import at all through any of the paths above (e.g. `research/under10.py`,
which does not exist at base; `web/views.py`) needs no binding, because base
`service.py` never resolves it in the first place.

Every other module -- the scoring engine, the models, and the long forecast
builder -- resolves to the working tree, which is exactly the comparison the
slice needs: this slice's own additive helpers must not have changed any
behavior the base service depends on, but nothing here re-derives modules this
slice never touched.

Nothing on disk is modified and no base file is copied into the working tree.
A shallow or object-pruned checkout simply has no base objects, in which case
`base_service_available()` is ``False`` and the caller skips rather than
silently comparing the working tree against itself.

This harness is deliberately separate from `tests/frozen_base.py`: that one
belongs to the `us-sec-long-v1`/`v2` golden at base
``38df10f014caf5ed81a57ac020b31f0dd634fd37`` and is neither reused nor
regenerated here.
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

#: The revision `feat/under10-shadow-controls` branched from.
BASE_SHA = "65314f87fe0eb8adbb05d35d3874c22c736ae54c"

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Bound in dependency order: `data.assets` before `data.asof` (which now
#: imports it), then `research.medium_forecasts` (which imports `data.asof`),
#: independent leaves next, and `research.service` last. Executing a
#: module's base source registers it in `sys.modules` *before* any later
#: module in this list is executed, so imports inside the base medium
#: builder and service resolve against the base modules this context
#: manager just bound -- never the live ones.
DEPENDENCY_MODULES: tuple[tuple[str, str], ...] = (
    ("stanstock.data.assets", "src/stanstock/data/assets.py"),
    ("stanstock.data.asof", "src/stanstock/data/asof.py"),
    ("stanstock.research.medium_forecasts", "src/stanstock/research/medium_forecasts.py"),
    ("stanstock.research.indicators", "src/stanstock/research/indicators.py"),
    ("stanstock.research.affordability", "src/stanstock/research/affordability.py"),
    ("stanstock.data.provider_policy", "src/stanstock/data/provider_policy.py"),
    ("stanstock.research.long_forecasts", "src/stanstock/research/long_forecasts.py"),
    ("stanstock.research.service", "src/stanstock/research/service.py"),
)

#: Backward-compatible name for the module/path this harness centers on.
SERVICE_PATH = "src/stanstock/research/service.py"


class BaseServiceUnavailableError(RuntimeError):
    """The base service bytes are not readable from the object database."""


@dataclass(frozen=True, slots=True)
class BaseService:
    base_sha: str
    #: SHA-256 over the concatenation of every bound dependency's base
    #: bytes, in `DEPENDENCY_MODULES` order -- not just `service.py` -- so a
    #: base revision missing any one of the bound files is distinguishable
    #: from one where all dependencies are genuinely present.
    source_sha256: str
    service: ModuleType


def base_service_available() -> bool:
    try:
        for _name, relative_path in DEPENDENCY_MODULES:
            _read_base_source(relative_path)
    except BaseServiceUnavailableError:
        return False
    return True


def base_service_checksum() -> str:
    """SHA-256 over the concatenated bound base sources, in bound order."""
    combined = b"".join(_read_base_source(path) for _name, path in DEPENDENCY_MODULES)
    return hashlib.sha256(combined).hexdigest()


def _read_base_source(relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{BASE_SHA}:{relative_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise BaseServiceUnavailableError(
            f"Base revision object {BASE_SHA}:{relative_path} is not readable from "
            "the local git object database"
        ) from error


def base_service_first_party_import_names() -> frozenset[str]:
    """Every ``stanstock.*`` module base `research/service.py`'s own source imports.

    Parsed directly from the actual base bytes via `ast`, not a hand-maintained
    list, so a regression test comparing this against `DEPENDENCY_MODULES`
    stays accurate as base service's own import list changes -- it is never a
    hardcoded tautology of the current binding set. Only names imported
    *directly* by `research/service.py` itself are returned; a module that a
    bound leaf imports transitively is that leaf's own concern, not this
    harness's, unless base `service.py` also imports it directly.
    """
    source = _read_base_source(SERVICE_PATH)
    tree = ast.parse(source, filename=SERVICE_PATH)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("stanstock.")
        ):
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("stanstock."):
                    modules.add(alias.name)
    return frozenset(modules)


def module_relative_path(module_name: str) -> str:
    """The `src/**.py` path a dotted ``stanstock.*`` module name resolves to."""
    return "src/" + module_name.replace(".", "/") + ".py"


@contextmanager
def base_research_service() -> Iterator[BaseService]:
    """Temporarily import the base revision's bound research modules.

    Every name in `DEPENDENCY_MODULES` is swapped for the duration of this
    context manager and restored (to whatever it was before -- live module,
    or absent) on exit, regardless of how the caller's block completes.

    Every original binding is snapshotted *before* any loading begins, not
    incrementally as each module is processed: if reading or executing a
    later module's base source fails partway through (e.g. `research.
    indicators` is unreadable), the modules after it in `DEPENDENCY_MODULES`
    were never reached and never swapped, but the `finally` block below still
    iterates over the complete list. Snapshotting incrementally would leave
    those never-reached names with no recorded original at all, and the
    restore loop would then treat "no snapshot" as "was absent" and
    incorrectly pop an untouched, perfectly live module out of `sys.modules`.
    """
    originals: dict[str, ModuleType | None] = {
        name: sys.modules.get(name) for name, _relative_path in DEPENDENCY_MODULES
    }
    sources: dict[str, bytes] = {}
    try:
        for name, relative_path in DEPENDENCY_MODULES:
            source = _read_base_source(relative_path)
            sources[name] = source
            spec = importlib.util.spec_from_loader(
                name, loader=None, origin=f"{BASE_SHA}:{relative_path}"
            )
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(REPO_ROOT / relative_path)
            # Registered before execution: `@dataclass` resolves annotations
            # through ``sys.modules[cls.__module__]`` while the class body
            # runs, and a later module in this list importing `name` must
            # see this bound module rather than whatever was live before.
            sys.modules[name] = module
            exec(compile(source, f"<{BASE_SHA}:{relative_path}>", "exec"), module.__dict__)
        combined_source = b"".join(sources[name] for name, _path in DEPENDENCY_MODULES)
        yield BaseService(
            base_sha=BASE_SHA,
            source_sha256=hashlib.sha256(combined_source).hexdigest(),
            service=sys.modules["stanstock.research.service"],
        )
    finally:
        for name, _relative_path in DEPENDENCY_MODULES:
            original = originals[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
