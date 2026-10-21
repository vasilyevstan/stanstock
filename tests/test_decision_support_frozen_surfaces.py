"""C2 supplement: exact f800 full-payload comparison via the unchanged probe.

This covers the unchanged probe's compatible daily/frequency/scheduled modes.
Retained parents with frequency bindings need a separate recovery supplement:
the old recovery mode deliberately refuses them, and is not claimed here.
The historical b0be comparison and all literal platform/parser pins are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

F800 = "f8008c584c4847776c1489965a22c67e25d61fda"
ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tests/research_product_frozen_probability_probe.py"


@pytest.fixture(scope="module")
def c2_f800_export(tmp_path_factory):
    export = tmp_path_factory.mktemp("c2-f800-read-only-export")
    archive = export / "source.tar"
    completed = subprocess.run(
        ["git", "--no-pager", "-C", str(ROOT), "archive", "--format=tar", F800],
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, "The explicitly required f800 Git object must be accessible"
    archive.write_bytes(completed.stdout)
    with tarfile.open(archive) as bundle:
        bundle.extractall(export / "tree", filter="data")
    archive.unlink()
    return export / "tree/src"


def _probe_environment(source):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STANSTOCK_", "PYTEST_", "DJANGO_"))
        and not any(
            marker in key.upper() for marker in ("API_KEY", "TOKEN", "CREDENTIAL", "PASSWORD")
        )
    }
    environment.update(
        PYTHONPATH=str(source),
        DJANGO_SETTINGS_MODULE="stanstock.settings.test",
        STANSTOCK_DISABLE_KEYCHAIN="1",
    )
    return environment


def _payload(source, workdir, mode):
    workdir.mkdir()
    output = workdir / "payload.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--mode",
            mode,
            "--cohort",
            "full",
            "--workdir",
            str(workdir),
            "--output",
            str(output),
        ],
        cwd=workdir,
        env=_probe_environment(source),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, f"{mode} synthetic probe failed:\n{completed.stderr[-6000:]}"
    document = json.loads(output.read_bytes())
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert document["output"]["prediction_count"] == 20
    withheld = [
        row
        for row in document["output"]["predictions"].values()
        if row["method_version"] == "us-price-fhs-v1"
        and row["insufficiency_reason"] == "filter_variance_degenerate"
    ]
    assert len(withheld) == 4, "A real withheld listing, not an omitted branch, must execute"
    return encoded


@pytest.mark.parametrize("mode", ["daily", "daily_derived", "scheduled"])
def test_c2_preserves_complete_f800_default_v1_payloads(c2_f800_export, tmp_path, mode):
    base = _payload(c2_f800_export, tmp_path / "base", mode)
    candidate = _payload(ROOT / "src", tmp_path / "candidate", mode)
    assert candidate == base, (
        f"Complete {mode} payload changed: base={hashlib.sha256(base).hexdigest()} "
        f"candidate={hashlib.sha256(candidate).hexdigest()}"
    )
    print(f"C2 compatible f800 {mode} full-payload sha256={hashlib.sha256(candidate).hexdigest()}")


def test_c2_recovers_genuine_f800_frequency_bound_parent_without_rewriting_any_proof(
    c2_f800_export, tmp_path
):
    seed = tmp_path / "f800-producer"
    _payload(c2_f800_export, seed, "scheduled")
    payloads = []
    for source, label in ((c2_f800_export, "base"), (ROOT / "src", "candidate")):
        workdir = tmp_path / label
        shutil.copytree(seed, workdir)
        output = workdir / "recovery.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests/decision_support_frozen_probe.py"),
                "--workdir",
                str(workdir),
                "--output",
                str(output),
            ],
            cwd=workdir,
            env=_probe_environment(source),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, (
            f"{label} retained-frequency recovery failed:\n{completed.stderr[-6000:]}"
        )
        document = json.loads(output.read_bytes())
        assert document["output"]["prediction_count"] == 20
        assert document["scheduled"]["credential_and_fetch_calls"] == {
            "credentials": 0,
            "fetches": 0,
        }
        assert document["scheduled"]["dirty_checkout_reads"] == 0
        payloads.append(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
    assert payloads[1] == payloads[0], (
        "Complete retained-frequency recovery changed: "
        f"base={hashlib.sha256(payloads[0]).hexdigest()} "
        f"candidate={hashlib.sha256(payloads[1]).hexdigest()}"
    )
    print(f"C2 f800 retained-frequency recovery sha256={hashlib.sha256(payloads[1]).hexdigest()}")


@pytest.mark.parametrize("mode", ["declaration", "demo"])
def test_c2_preserves_full_f800_legacy_default_declaration_demo_and_rejection_payloads(
    c2_f800_export, tmp_path, mode
):
    payloads = []
    for source, label in ((c2_f800_export, "base"), (ROOT / "src", "candidate")):
        workdir = tmp_path / label
        workdir.mkdir()
        output = workdir / "supplement.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests/decision_support_frozen_probe.py"),
                "--mode",
                mode,
                "--workdir",
                str(workdir),
                "--output",
                str(output),
            ],
            cwd=workdir,
            env=_probe_environment(source),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, (
            f"{label} {mode} supplement failed:\n{completed.stderr[-6000:]}"
        )
        document = json.loads(output.read_bytes())
        assert document["output"]["prediction_count"] > 0
        payloads.append(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
    assert payloads[1] == payloads[0], (
        f"Complete legacy {mode} payload changed: base={hashlib.sha256(payloads[0]).hexdigest()} "
        f"candidate={hashlib.sha256(payloads[1]).hexdigest()}"
    )
    print(f"C2 f800 {mode} full-payload sha256={hashlib.sha256(payloads[1]).hexdigest()}")
