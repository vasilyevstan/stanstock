"""Lazy predicates for reportable/canonical aggregate performance reporting.

``reportable_prediction_filter`` -- is this prediction observed, on-time,
provider-backed evidence at all?

``canonical_reportable_prediction_filter`` -- is it also the earliest
reportable prediction for its exact observation key -- ``(listing,
target_date, horizon, evidence_role, method_version, config_hash,
price_provider)``, deliberately excluding ``model_version``/run identity so
a later on-time reissue of the same observation does not get double-counted?

Both return lazy ``Q`` objects usable directly on a ``Prediction`` queryset
(``prefix=""``) or traversed from ``PredictionOutcome`` via its
``prediction`` relation (``prefix="prediction__"``). Neither executes a
query; the canonical helper compiles its "no earlier sibling" check to a
correlated SQL ``NOT EXISTS`` via ``Exists``/``OuterRef``.
"""

from __future__ import annotations

from typing import cast

from django.db.models import Exists, OuterRef, Q

from stanstock.data.models import UniverseSnapshot
from stanstock.research.models import Prediction

# Deliberately excludes `model_version` and any run/analysis identifier:
# those distinguish *versions* of the same observation, not distinct ones.
_OBSERVATION_KEY_FIELDS: tuple[str, ...] = (
    "listing_id",
    "target_date",
    "horizon",
    "evidence_role",
    "method_version",
    "config_hash",
    "price_provider",
)


def reportable_prediction_filter(prefix: str = "") -> Q:
    """Observed, on-time, provider-backed prediction evidence: prediction and
    parent-run ``issued_on_time=True``, ``evidence_grade=OBSERVED``,
    ``source_mode=PROVIDER``, non-empty ``price_provider``."""
    return Q(
        **{
            f"{prefix}evidence_grade": UniverseSnapshot.Grade.OBSERVED,
            f"{prefix}source_mode": Prediction.SourceMode.PROVIDER,
            f"{prefix}issued_on_time": True,
            f"{prefix}analysis__run__issued_on_time": True,
        }
    ) & ~Q(**{f"{prefix}price_provider": ""})


def _no_earlier_reportable_sibling(prefix: str) -> Exists:
    """``NOT EXISTS`` a strictly-earlier (``generated_at`` then UUID ``id``
    ascending) reportable ``Prediction`` sharing this row's observation key.
    Repeats only reportability -- no outcome/status predicate -- so an
    earlier reportable-but-unresolved/corporate-event/withheld row is never
    displaced by a later matured sibling, while an earlier non-reportable row
    never suppresses a later reportable one."""
    key_filter = {field: OuterRef(f"{prefix}{field}") for field in _OBSERVATION_KEY_FIELDS}
    earlier_siblings = (
        Prediction.objects.filter(**key_filter)
        .filter(reportable_prediction_filter())
        .filter(
            Q(generated_at__lt=OuterRef(f"{prefix}generated_at"))
            | Q(
                generated_at=OuterRef(f"{prefix}generated_at"),
                id__lt=OuterRef(f"{prefix}id"),
            )
        )
        .order_by()
    )
    return Exists(earlier_siblings)


def canonical_reportable_prediction_filter(prefix: str = "") -> Q:
    """``reportable AND NOT EXISTS(earlier reportable sibling)``: the earliest
    reportable prediction for its exact observation key. Later valid
    reportable reissues stay in the immutable ledger, evaluated per version,
    but are excluded here so one observation is counted once."""
    return cast(Q, reportable_prediction_filter(prefix) & ~_no_earlier_reportable_sibling(prefix))
