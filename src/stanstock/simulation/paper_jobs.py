"""One no-fetch paper capture adapter for explicit CLI and opted-in refresh."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import DecimalException

import polars as pl
from django.db import transaction
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job, target_job_lock
from stanstock.core.models import JobRun
from stanstock.data.asof import PriceFrameChecksumMismatchError
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset
from stanstock.data.provider_policy import ProviderError
from stanstock.research.forward_scorecard import PROTOCOL_EPOCH
from stanstock.research.product_reader import ProductUser
from stanstock.simulation.paper import _calendar
from stanstock.simulation.paper_v2 import (
    ACTIVATION_KIND,
    PaperArtifact,
    PaperV2Error,
    _capture_for_job,
    _capture_time,
    _one,
    _verify_origin,
    activation_subject,
    capture_identity,
    capture_job_name,
    read_activation,
    read_state,
)


@dataclass(frozen=True, slots=True)
class PaperCaptureResult:
    child: JobRun | None
    state: str
    reason_code: str | None = None
    capture: PaperArtifact | None = None

    @property
    def capture_count(self) -> int:
        return int(self.capture is not None)


class _RecoveryOnly(Exception):
    """Short-circuit before the generic runner creates a retry/skip row."""

    def __init__(self, result: PaperCaptureResult) -> None:
        self.result = result


def _safe_error(exc: BaseException) -> PaperV2Error:
    return exc if isinstance(exc, PaperV2Error) else PaperV2Error()


def execute_paper_capture_job(
    *,
    owner: ProductUser,
    session: date,
    store: AssetStore,
    activation_asset: DataAsset | None = None,
    recovery_only: bool = False,
) -> PaperCaptureResult:
    """Recover before any fresh work; child target -> owner -> durable commit.

    ``execute_target_job`` owns the target lock and normal status lifecycle.
    The before-attempt short circuit deliberately writes no skipped/retry row:
    the *original* job and assets, not a new status, are the recovery evidence.
    """
    child: JobRun | None = None
    job_name: str | None = None
    try:
        activation = read_activation(user=owner, store=store, asset=activation_asset)
        if type(session) is not date or session.isoformat() not in activation.document["calendar"]:
            return PaperCaptureResult(None, "outside_horizon", "paper_v2_outside_horizon")
        now = timezone.now()
        if timezone.is_naive(now):
            raise PaperV2Error("paper_v2_deadline_missed")
        calendar = _calendar(session)
        if now <= calendar.session_close(session).to_pydatetime():
            return PaperCaptureResult(None, "not_due")
        identity = capture_identity(activation)
        job_name = capture_job_name(identity)

        def before_attempt() -> None:
            with (
                target_job_lock(
                    job_name=activation_subject(str(owner.pk)),
                    region="us",
                    target_date=PROTOCOL_EPOCH,
                ),
                transaction.atomic(durable=True),
            ):
                # Re-read bytes under both locks. No request-spanning proof cache.
                state = read_state(user=owner, store=store, activation_asset=activation.asset)
                existing = next(
                    (
                        row
                        for row in state.captures
                        if row.document["session"] == session.isoformat()
                    ),
                    None,
                )
                if existing is not None:
                    assert state.execution is not None
                    origin = _verify_origin(state.activation, state.execution, existing)
                    raise _RecoveryOnly(PaperCaptureResult(origin, "recovered", capture=existing))
                # A success alone, or a surviving full binding without its
                # registered assets, is not proof and must never be repaired.
                prior = JobRun.objects.filter(job_name=job_name, region="us", target_date=session)
                if any(
                    row.status == JobRun.Status.SUCCESS
                    or (isinstance(row.details, dict) and "paper_v2_origin" in row.details)
                    for row in prior
                ):
                    raise PaperV2Error()
                if recovery_only:
                    raise _RecoveryOnly(PaperCaptureResult(None, "awaiting_capture"))

        def task(run: JobRun) -> JobExecutionResult:
            try:
                _capture_for_job(
                    user=owner,
                    session=session,
                    activation_asset=activation.asset,
                    job=run,
                    identity=identity,
                    store=store,
                )
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                DecimalException,
                PriceFrameChecksumMismatchError,
                ProviderError,
                pl.exceptions.PolarsError,
            ) as exc:
                # Generic runner diagnostics/logs must not inherit prices,
                # holdings, raw parser text, paths, URLs, or credential text.
                raise _safe_error(exc) from None
            run.refresh_from_db()
            # Preserve the atomic binding in the runner's normal final write.
            return JobExecutionResult(details=run.details)

        child = execute_target_job(
            job_name=job_name,
            region="us",
            target_date=session,
            task=task,
            before_attempt=before_attempt,
        )
        state = read_state(user=owner, store=store, activation_asset=activation.asset)
        capture = next(
            row for row in state.captures if row.document["session"] == session.isoformat()
        )
        assert state.execution is not None
        _verify_origin(state.activation, state.execution, capture)
        return PaperCaptureResult(child, "captured", capture=capture)
    except _RecoveryOnly as recovered:
        return recovered.result
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        DecimalException,
        PriceFrameChecksumMismatchError,
        ProviderError,
        pl.exceptions.PolarsError,
    ) as exc:
        if job_name is not None:
            child = (
                JobRun.objects.filter(job_name=job_name, region="us", target_date=session)
                .order_by("-attempt")
                .first()
            )
        return PaperCaptureResult(child, "failed", _safe_error(exc).reason_code)


@dataclass
class PaperCaptureInvocation:
    """One invocation disposition, not a cache of source proofs or summaries."""

    owner: ProductUser
    session: date
    store: AssetStore
    activation_asset: DataAsset | None
    result: PaperCaptureResult

    def recover(self) -> None:
        if self.activation_asset is not None and self.result.state == "awaiting_capture":
            self.result = execute_paper_capture_job(
                owner=self.owner,
                session=self.session,
                store=self.store,
                activation_asset=self.activation_asset,
                recovery_only=True,
            )

    def after_market(self, *, attempted: bool) -> PaperCaptureResult:
        if (
            attempted
            and self.activation_asset is not None
            and self.result.state == "awaiting_capture"
        ):
            # No second attempt after committed recovery or a proof failure.
            try:
                _capture_time(self.session, timezone.now())
            except (TypeError, ValueError):
                self.result = PaperCaptureResult(None, "withheld", "paper_v2_deadline_missed")
            else:
                self.result = execute_paper_capture_job(
                    owner=self.owner,
                    session=self.session,
                    store=self.store,
                    activation_asset=self.activation_asset,
                )
        return self.result


def prepare_paper_capture(
    *, owner: ProductUser, session: date, store: AssetStore
) -> PaperCaptureInvocation:
    # Exactly ONE existence lookup for an unactivated opted-in invocation.
    # No authority/source proof, paper lock, revision check, file I/O, or job.
    # An activation appearing later deliberately waits for the next invocation.
    asset = _one(ACTIVATION_KIND, activation_subject(str(owner.pk)))
    return PaperCaptureInvocation(
        owner,
        session,
        store,
        asset,
        PaperCaptureResult(None, "unactivated" if asset is None else "awaiting_capture"),
    )
