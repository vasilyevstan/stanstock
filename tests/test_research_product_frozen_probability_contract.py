"""Executable proof that `us-price-fhs-v1` is byte-for-byte frozen at b0be050.

The probability-first research slice adds a *separately registered*, additive
frequency `DataAsset` and a sibling scheduled child.  Its binding material
contract requires that everything it was declared not to touch -- the
`us-price-fhs-v1` config bytes and effective hash, the RNG/seed/8,192-path/
chunked simulation, the complete successful *and* genuinely withheld
calculation, projection and scenario payloads, the five immutable `Prediction`
rows per qualified listing, the registered output-manifest bytes, and the
original canonical scheduled verification and replay payload -- is identical
to the frozen base revision.

Proving that requires executing the base revision, not re-reading the
candidate.  Both revisions define the same dotted module names, so each is
executed in its own subprocess by the shared
`tests/research_product_frozen_probability_probe.py`:

* the **base** process runs against a read-only ``git archive`` export of
  :data:`FROZEN_BASE_SHA` into a disposable directory;
* the **candidate** process runs against the current *working tree*, so
  uncommitted corrections are included -- an archive of ``HEAD`` would not
  prove anything about the code that will actually ship.

The probe controls every identity, clock, and revision identically in both
processes (see its module docstring), so the two payloads are compared whole.
No field is excluded, no value is normalized after the fact, and no
success-shaped placeholder is substituted for missing evidence.

Declared additive difference
----------------------------
Exactly one class of difference is admissible, and only as an *addition*:
the new frequency asset, the new `research_product_frequency_v1` child, and
the new `frequencies`/`frequency_verification` parent-detail blocks.  A
changed or removed value is never admissible, and the additive allowance is
expressed as an explicit path predicate rather than by filtering the payload
before comparing it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

#: The exact frozen base revision this slice branched from.
FROZEN_BASE_SHA = "b0be0503e193acf6be2562c40390970a918d9f72"

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).resolve().parent / "research_product_frozen_probability_probe.py"

#: Wall-clock ceiling for one probe process. A full frozen issuance runs four
#: 8,192-path simulations plus its manifest verification.
PROBE_TIMEOUT_SECONDS = 120

#: Path segments that identify the additive frequency evidence. A difference
#: is admissible only when it is an *addition* underneath one of these.
ADDITIVE_SEGMENTS = ("frequencies", "frequency_verification")


# --------------------------------------------------------------------------
# Payload comparison
# --------------------------------------------------------------------------


def _flatten(value: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Map every leaf of `value` to its full path; nothing is summarized."""
    if isinstance(value, dict):
        flat: dict[tuple[str, ...], Any] = {path: "<mapping>"}
        for key, item in value.items():
            flat.update(_flatten(item, (*path, str(key))))
        return flat
    if isinstance(value, list):
        flat = {path: "<sequence>"}
        for index, item in enumerate(value):
            flat.update(_flatten(item, (*path, f"[{index}]")))
        return flat
    return {path: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)}


def _label(path: tuple[str, ...]) -> str:
    return " -> ".join(path)


def _is_additive(path: tuple[str, ...], *, parent_job_key: str | None) -> bool:
    return (
        (
            len(path) >= 2
            and path[0] == "assets"
            and path[1].startswith("research_product_frequency_evidence|")
        )
        or (
            len(path) >= 3
            and path[:2] == ("scheduled", "job_runs")
            and path[2].startswith("research_product_frequency_v1|")
        )
        or (
            len(path) >= 4
            and path[:3] == ("scheduled", "parent_job_run", "details")
            and path[3] in ADDITIVE_SEGMENTS
        )
        or (
            len(path) >= 5
            and path[:2] == ("scheduled", "job_runs")
            and path[2] == parent_job_key
            and path[3] == "details"
            and path[4] in ADDITIVE_SEGMENTS
        )
    )


def _describe(items: list[tuple[str, ...]], limit: int = 12) -> str:
    shown = [_label(path) for path in sorted(items)[:limit]]
    suffix = "" if len(items) <= limit else f" ... and {len(items) - limit} more"
    return "\n  ".join(shown) + suffix


def assert_frozen_surface_unchanged(
    base: dict[str, Any],
    candidate: dict[str, Any],
    *,
    allow_additive: bool,
) -> None:
    """Compare two complete payloads without excluding a single field."""
    parent = base.get("scheduled", {}).get("parent_job_run", {})
    identity_fields = ("job_name", "region", "target_date", "attempt")
    parent_job_key = (
        "|".join(str(parent[field]) for field in identity_fields)
        if all(field in parent for field in identity_fields)
        else None
    )
    base_flat = _flatten(base)
    candidate_flat = _flatten(candidate)

    removed = sorted(set(base_flat) - set(candidate_flat))
    added = sorted(set(candidate_flat) - set(base_flat))
    changed = sorted(
        path
        for path in set(base_flat) & set(candidate_flat)
        if base_flat[path] != candidate_flat[path]
    )

    problems: list[str] = []
    if removed:
        problems.append(f"{len(removed)} frozen value(s) disappeared:\n  {_describe(removed)}")
    if changed:
        detail = "\n  ".join(
            f"{_label(path)}\n    base      = {base_flat[path]!r}"
            f"\n    candidate = {candidate_flat[path]!r}"
            for path in changed[:8]
        )
        problems.append(f"{len(changed)} frozen value(s) changed:\n  {detail}")
    if added:
        if not allow_additive:
            problems.append(f"{len(added)} unexpected value(s) appeared:\n  {_describe(added)}")
        else:
            undeclared = [
                path for path in added if not _is_additive(path, parent_job_key=parent_job_key)
            ]
            if undeclared:
                problems.append(
                    f"{len(undeclared)} addition(s) outside the declared additive frequency "
                    f"evidence:\n  {_describe(undeclared)}"
                )
    if problems:
        raise AssertionError(
            "The frozen us-price-fhs-v1 surface differs between "
            f"{FROZEN_BASE_SHA[:7]} and the current working tree.\n" + "\n".join(problems)
        )


# --------------------------------------------------------------------------
# Source trees
# --------------------------------------------------------------------------


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    """Run one strictly read-only Git query against the repository."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=check,
        capture_output=True,
    )


def _base_objects_available() -> bool:
    return _git("cat-file", "-e", f"{FROZEN_BASE_SHA}^{{commit}}", check=False).returncode == 0


@pytest.fixture(scope="session")
def frozen_base_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the frozen base revision's tracked tree into a disposable dir.

    ``git archive`` reads the object database only: it stages nothing, writes
    nothing into the repository, and never touches the working tree. Private
    or untracked files (``.env`` and friends) are not tracked and therefore
    cannot be exported.
    """
    if not _base_objects_available():
        pytest.skip(
            f"Frozen base revision {FROZEN_BASE_SHA} is not present in the local object "
            "database, so its own code cannot be executed for the differential."
        )
    root = tmp_path_factory.mktemp("frozen-base-tree")
    archive = root / "base.tar"
    archive.write_bytes(_git("archive", "--format=tar", FROZEN_BASE_SHA).stdout)
    with tarfile.open(archive) as bundle:
        bundle.extractall(root / "tree", filter="data")
    archive.unlink()
    source = root / "tree" / "src"
    if not (source / "stanstock" / "research" / "price_product.py").is_file():
        raise AssertionError(f"The exported {FROZEN_BASE_SHA[:7]} tree has no price product module")
    return source


@pytest.fixture(scope="session")
def candidate_root() -> Path:
    """The current working tree, including uncommitted corrections."""
    source = REPO_ROOT / "src"
    if not (source / "stanstock" / "research" / "price_product.py").is_file():
        raise AssertionError("The working tree has no price product module")
    return source


# --------------------------------------------------------------------------
# Probe execution
# --------------------------------------------------------------------------


class ProbeRun:
    """One completed probe process and its emitted payload (or failure)."""

    def __init__(
        self,
        *,
        returncode: int,
        payload: dict[str, Any] | None,
        stderr: str,
        workdir: Path,
    ) -> None:
        self.returncode = returncode
        self.payload = payload
        self.stderr = stderr
        #: The directory the process left behind. Its SQLite database and
        #: asset store are that revision's genuine retained output and can be
        #: handed to another revision's recovery run.
        self.workdir = workdir

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and self.payload is not None

    def require(self, label: str) -> dict[str, Any]:
        if self.payload is None:
            raise AssertionError(
                f"The {label} probe did not emit a payload (exit {self.returncode}).\n"
                f"{self.failure_detail()}"
            )
        return self.payload

    def failure_detail(self, lines: int = 25) -> str:
        interesting = [
            line
            for line in self.stderr.splitlines()
            if '"level":"INFO"' not in line and line.strip()
        ]
        return "\n".join(interesting[-lines:])


def _run_probe(*, source_root: Path, mode: str, cohort: str, workdir: Path) -> ProbeRun:
    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / "payload.json"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STANSTOCK_", "PYTEST_", "DJANGO_"))
    }
    environment["PYTHONPATH"] = str(source_root)
    environment["DJANGO_SETTINGS_MODULE"] = "stanstock.settings.test"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--mode",
            mode,
            "--cohort",
            cohort,
            "--workdir",
            str(workdir),
            "--output",
            str(output),
        ],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
        check=False,
    )
    payload = (
        json.loads(output.read_text(encoding="utf-8"))
        if completed.returncode == 0 and output.is_file()
        else None
    )
    return ProbeRun(
        returncode=completed.returncode,
        payload=payload,
        stderr=completed.stderr,
        workdir=workdir,
    )


@pytest.fixture(scope="session")
def probe(
    tmp_path_factory: pytest.TempPathFactory,
    frozen_base_root: Path,
    candidate_root: Path,
):
    """Execute (and memoize) one probe process per revision/mode/cohort."""
    roots = {"base": frozen_base_root, "candidate": candidate_root}
    cache: dict[tuple[str, str, str], ProbeRun] = {}
    workspace = tmp_path_factory.mktemp("frozen-probe-runs")

    def _probe(revision: str, mode: str, cohort: str) -> ProbeRun:
        key = (revision, mode, cohort)
        if key not in cache:
            cache[key] = _run_probe(
                source_root=roots[revision],
                mode=mode,
                cohort=cohort,
                workdir=workspace / "-".join(key),
            )
        return cache[key]

    return _probe


@pytest.fixture(scope="session")
def recover_retained_base_parent(
    tmp_path_factory: pytest.TempPathFactory,
    frozen_base_root: Path,
    candidate_root: Path,
    probe,
):
    """Replay a genuinely retained base-written scheduled state per revision.

    The retained state is the working directory a full base scheduled probe
    process actually left behind: a real `JobRun` parent with its real
    children, immutable rows, registered assets, and asset files, written by
    :data:`FROZEN_BASE_SHA`'s own code. Each revision gets a pristine copy,
    because recovery legitimately appends its own skipped attempt.
    """
    roots = {"base": frozen_base_root, "candidate": candidate_root}
    cache: dict[str, ProbeRun] = {}
    workspace = tmp_path_factory.mktemp("frozen-recovery-runs")

    def _recover(revision: str) -> ProbeRun:
        if revision not in cache:
            source = probe("base", "scheduled", "full")
            source.require(f"base {FROZEN_BASE_SHA[:7]} scheduled")
            retained = workspace / revision
            shutil.copytree(source.workdir, retained)
            (retained / "payload.json").unlink(missing_ok=True)
            cache[revision] = _run_probe(
                source_root=roots[revision],
                mode="scheduled_recovery",
                cohort="full",
                workdir=retained,
            )
        return cache[revision]

    return _recover


# --------------------------------------------------------------------------
# Frozen-contract proofs
# --------------------------------------------------------------------------


def test_comparison_rejects_changed_removed_and_undeclared_values() -> None:
    """The comparator itself must not be able to pass a real difference."""
    base = {"a": {"b": 1}, "c": [2, 3]}

    with pytest.raises(AssertionError, match="changed"):
        assert_frozen_surface_unchanged(base, {"a": {"b": 9}, "c": [2, 3]}, allow_additive=True)
    with pytest.raises(AssertionError, match="disappeared"):
        assert_frozen_surface_unchanged(base, {"c": [2, 3]}, allow_additive=True)
    with pytest.raises(AssertionError, match="outside the declared additive"):
        assert_frozen_surface_unchanged(
            base, {"a": {"b": 1, "d": 4}, "c": [2, 3]}, allow_additive=True
        )
    with pytest.raises(AssertionError, match="unexpected"):
        assert_frozen_surface_unchanged(
            base,
            {"a": {"b": 1}, "c": [2, 3], "frequencies": {"x": 1}},
            allow_additive=False,
        )
    with pytest.raises(AssertionError, match="outside the declared additive"):
        assert_frozen_surface_unchanged(
            base, {"a": {"b": 1, "frequencies": {"x": 1}}, "c": [2, 3]}, allow_additive=True
        )
    assert_frozen_surface_unchanged(
        {"assets": {}},
        {"assets": {"research_product_frequency_evidence|fixture": {"count": 1}}},
        allow_additive=True,
    )


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (1, True),
        (1, 1.0),
        (0.0, -0.0),
        ({"[0]": 1}, [1]),
        ({}, "<mapping>"),
    ),
)
def test_comparison_preserves_json_types_and_container_shapes(before, after) -> None:
    with pytest.raises(AssertionError, match="changed"):
        assert_frozen_surface_unchanged({"value": before}, {"value": after}, allow_additive=True)


@pytest.fixture(scope="session")
def perturbed_base_root(tmp_path_factory: pytest.TempPathFactory, frozen_base_root: Path) -> Path:
    """A disposable base copy with exactly one frozen constant changed.

    Only the seed-identity domain is altered. It loads and verifies exactly
    like the frozen original, so the resulting payload difference comes from
    the frozen math itself rather than from an import or validation failure.
    """
    root = tmp_path_factory.mktemp("frozen-base-perturbed") / "tree"
    shutil.copytree(frozen_base_root.parent, root)
    module = root / "src" / "stanstock" / "research" / "price_product.py"
    source = module.read_text(encoding="utf-8")
    original = '_SEED_DOMAIN = "stanstock-research-product-seed-v1"'
    assert original in source, "The frozen seed identity constant moved"
    module.write_text(source.replace(original, f'{original[:-1]}-perturbed"'), encoding="utf-8")
    return root / "src"


def test_the_differential_detects_a_single_frozen_constant_change(
    probe, perturbed_base_root: Path, tmp_path: Path
) -> None:
    """A passing comparison must mean something: prove the harness can fail.

    Changing one constant inside the frozen operator must be caught by the
    same whole-payload comparison the contract proofs rely on. Without this,
    "no difference" could just mean the probe never reached the math.
    """
    base = probe("base", "daily", "full").require(f"base {FROZEN_BASE_SHA[:7]} daily")
    perturbed = _run_probe(
        source_root=perturbed_base_root,
        mode="daily",
        cohort="full",
        workdir=tmp_path / "perturbed",
    ).require("perturbed base")

    with pytest.raises(AssertionError, match="frozen value\\(s\\) changed") as failure:
        assert_frozen_surface_unchanged(base, perturbed, allow_additive=True)
    message = str(failure.value)
    assert "deterministic_seeds" in message or "base_return" in message or "calculation" in message


def test_probe_fixture_reaches_both_frozen_states_on_the_base_revision(probe) -> None:
    """The differential is only meaningful if the fixture exercises both states."""
    payload = probe("base", "daily", "full").require(f"base {FROZEN_BASE_SHA[:7]} daily")
    output = payload["output"]

    assert sorted(output["stock_analyses"]) == ["AAPL", "CHEAP", "FLAT", "MSFT"]
    assert output["prediction_count"] == 20
    withheld = {
        key: row["insufficiency_reason"]
        for key, row in output["predictions"].items()
        if key.startswith("FLAT|") and row["method_version"] == "us-price-fhs-v1"
    }
    assert sorted(withheld) == [
        "FLAT|advisory|12m|us-price-fhs-v1",
        "FLAT|advisory|3y|us-price-fhs-v1",
        "FLAT|advisory|5y|us-price-fhs-v1",
        "FLAT|advisory|6m|us-price-fhs-v1",
    ]
    assert set(withheld.values()) == {"filter_variance_degenerate"}
    assert all(
        row["base_return"] is None
        for key, row in output["predictions"].items()
        if key.startswith("FLAT|")
    )
    successful = [
        row
        for key, row in output["predictions"].items()
        if key.startswith("CHEAP|") and row["method_version"] == "us-price-fhs-v1"
    ]
    assert len(successful) == 4
    assert all(row["insufficiency_reason"] == "" for row in successful)
    assert all(row["base_return"] is not None for row in successful)
    assert payload["config"]["fhs_method_version"] == "us-price-fhs-v1"
    assert payload["config"]["typed_config"]["simulation"]["production_paths"] == 8192


def test_frozen_daily_issuance_is_unchanged_for_successful_and_withheld_evidence(probe) -> None:
    """Complete base-versus-candidate equality with the additive work excluded.

    This is the core frozen-contract comparison: identical controlled
    identities, identical clocks, identical declared revision, and every
    emitted value -- config bytes and hashes, typed config, the complete
    calculation/projection/scenario payloads for the successful *and* the
    genuinely withheld listing, all five immutable rows per qualified
    listing, the membership admissions and reasons, the deterministic seeds,
    the registered manifest and calculation-artifact bytes and documents --
    compared whole.
    """
    base = probe("base", "daily", "full").require(f"base {FROZEN_BASE_SHA[:7]} daily")
    candidate = probe("candidate", "daily", "full").require("candidate daily")

    assert_frozen_surface_unchanged(base, candidate, allow_additive=False)


def test_frozen_scheduled_verification_and_replay_are_unchanged(probe) -> None:
    """The original canonical scheduled payload survives the sibling stage.

    The candidate's own additive parent-detail blocks and sibling child are
    permitted only as additions and only under their declared paths; the
    recorded, recomputed, and replayed canonical verification dictionaries
    are compared in full, with no key removed from either side.
    """
    base = probe("base", "scheduled", "successful").require(f"base {FROZEN_BASE_SHA[:7]} scheduled")
    candidate = probe("candidate", "scheduled", "successful").require("candidate scheduled")

    assert (
        base["scheduled"]["recorded_verification"] == base["scheduled"]["recomputed_verification"]
    )
    assert (
        candidate["scheduled"]["recorded_verification"]
        == candidate["scheduled"]["replayed_verification"]
    )
    for block in ("recorded_verification", "recomputed_verification", "replayed_verification"):
        assert candidate["scheduled"][block] == base["scheduled"][block], block
    assert (
        candidate["scheduled"]["parent_job_run"]["details"]["stages"]
        == (base["scheduled"]["parent_job_run"]["details"]["stages"])
    )

    assert_frozen_surface_unchanged(base, candidate, allow_additive=True)


def test_candidate_replays_a_genuinely_retained_base_written_parent(
    recover_retained_base_parent,
) -> None:
    """Candidate code must replay an old parent it did not write itself.

    Every other proof here compares two independently *produced* states. This
    one does not: a full base scheduled probe writes a real database and asset
    store, and the candidate is then pointed at that untouched retained state
    and made to recover it. The parent, its children, its immutable rows, its
    manifests, and its registered assets are therefore genuinely the frozen
    revision's own bytes, not a hand-assembled stand-in such as a bare
    ``JobRun(details={"stages": {}})``.

    The cohort is the full one, so the retained parent includes the
    genuinely withheld listing as well as the successful ones.
    """
    base = recover_retained_base_parent("base").require(
        f"base {FROZEN_BASE_SHA[:7]} recovery of its own retained parent"
    )
    candidate = recover_retained_base_parent("candidate").require(
        "candidate recovery of the retained base parent"
    )

    for label, payload in (("base", base), ("candidate", candidate)):
        retained = payload["scheduled"]
        # The probe refuses a parent that already carries the additive
        # bindings, so reaching here at all means the recovered parent
        # genuinely predates them; assert it explicitly as well.
        before = retained["retained_parent_details_before_recovery"]
        assert "frequencies" not in before, label
        assert "frequency_verification" not in before, label
        assert retained["recovery_job_run"]["status"] == "skipped", label
        # Recovering must not rewrite the retained parent in place.
        assert retained["parent_job_run"]["details"] == before, label
        assert retained["recorded_verification"] == retained["replayed_verification"], label
        assert retained["recorded_verification"] == retained["recomputed_verification"], label

    # Nothing may be added when recovering a frozen old-source parent: the
    # additive stage lives inside the parent task, which a skip never runs.
    assert_frozen_surface_unchanged(base, candidate, allow_additive=False)


def test_retained_base_parent_recovery_covers_the_withheld_listing(
    recover_retained_base_parent,
) -> None:
    """The recovered retained state is the meaningful, not the empty, one."""
    candidate = recover_retained_base_parent("candidate").require(
        "candidate recovery of the retained base parent"
    )

    listings = candidate["output"]["predictions"]
    withheld = sorted(key for key, value in listings.items() if value.get("insufficiency_reason"))
    successful = sorted(
        key for key, value in listings.items() if not value.get("insufficiency_reason")
    )
    assert withheld, "the retained parent must carry a genuinely withheld forecast"
    assert successful, "the retained parent must carry successful forecasts too"
    assert {listings[key]["insufficiency_reason"] for key in withheld} == {
        "filter_variance_degenerate"
    }
    assert candidate["scheduled"]["prediction_count"] == len(listings)
    assert len(candidate["output"]["stock_analyses"]) * 5 == len(listings)
    assert candidate["scheduled"]["recorded_verification"]["prediction_count"] == len(listings)


def test_additive_frequency_evidence_only_adds_a_separately_registered_asset(probe) -> None:
    """Running the additive derivation may only add its own evidence."""
    base = probe("base", "daily", "successful").require(f"base {FROZEN_BASE_SHA[:7]} daily")
    candidate = probe("candidate", "daily_derived", "successful").require(
        "candidate daily with frequency derivation"
    )

    assert_frozen_surface_unchanged(base, candidate, allow_additive=True)
    added = sorted(set(_flatten(candidate)) - set(_flatten(base)))
    assert added, "The additive derivation registered no frequency evidence at all"
    assert {path[0] for path in added} == {"assets"}
    assert {path[1].split("|", 1)[0] for path in added} == {"research_product_frequency_evidence"}


# --------------------------------------------------------------------------
# Withheld-evidence regression in the additive derivation
# --------------------------------------------------------------------------
#
# Both proofs below are the standing regression for a production defect these
# tests first reproduced in code this module does not own:
# `product_frequency_evidence._derive_listing` called
# `filter_historical_returns` without handling `PriceProductInputError`, so a
# qualified listing whose frozen FHS forecast is *genuinely withheld* took
# down the surrounding issuance while the frozen base revision completed the
# identical fixture. The owning agent has since guarded that call; these
# assertions stay ordinary assertions, not xfail, because the contract states
# the frequency report is additive and a withheld forecast must never fail
# the frozen writer or the scheduled parent again.


def test_additive_frequency_derivation_preserves_the_frozen_daily_issuance(probe) -> None:
    """A withheld frozen forecast must not fail the frozen daily writer."""
    run = probe("candidate", "daily_derived", "full")

    assert run.succeeded, (
        "Enabling the additive frequency derivation failed the frozen daily research job "
        "for a qualified listing whose us-price-fhs-v1 forecast is genuinely withheld, "
        f"while base {FROZEN_BASE_SHA[:7]} completes the identical fixture.\n"
        f"{run.failure_detail()}"
    )
    base = probe("base", "daily", "full").require(f"base {FROZEN_BASE_SHA[:7]} daily")
    assert_frozen_surface_unchanged(base, run.require("candidate"), allow_additive=True)


def test_additive_frequency_stage_preserves_the_frozen_scheduled_parent(probe) -> None:
    """A withheld frozen forecast must not fail the scheduled parent."""
    run = probe("candidate", "scheduled", "full")

    assert run.succeeded, (
        "The additive frequency stage failed the scheduled research parent for a "
        "qualified listing whose us-price-fhs-v1 forecast is genuinely withheld, while "
        f"base {FROZEN_BASE_SHA[:7]} completes the identical fixture.\n"
        f"{run.failure_detail()}"
    )
    base = probe("base", "scheduled", "full").require(f"base {FROZEN_BASE_SHA[:7]} scheduled")
    assert_frozen_surface_unchanged(base, run.require("candidate"), allow_additive=True)
