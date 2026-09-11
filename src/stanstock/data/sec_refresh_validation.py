"""SEC data-stage scheduled-refresh verification.

Binds the reviewed SEC mapping/CIK/fundamentals config identity, then
proves the target's own child run's declared asset references are exactly
the canonical SEC evidence for the full CIK config: one submissions asset,
one companyfacts asset, and one history asset per filename the submissions
payload requires, for every mapped CIK, and nothing else. Does not import
`stanstock.data.sec_ingestion` or `stanstock.data.live_us`, both of which
transitively import `stanstock.research.service`. Semantic fact/filing
closure for long-horizon predictions is a research-domain concern.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from stanstock.core.models import JobRun
from stanstock.core.verification_types import (
    AssetRef,
    RefreshVerificationError,
    StageVerificationResult,
)
from stanstock.data.assets import (
    asset_ref_for,
    open_asset_store,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset
from stanstock.data.providers import sec
from stanstock.data.sec_config import load_sec_cik_config, load_sec_fundamentals_config
from stanstock.data.sec_evidence import (
    COMPANYFACTS_KIND,
    HISTORY_FILENAME_METADATA_KEY,
    MAPPING_KIND,
    MAPPING_SUBJECT,
    SUBMISSIONS_HISTORY_KIND,
    SUBMISSIONS_KIND,
    SecEvidencePayloadError,
    historical_submission_filenames,
)

_SEC_ASSET_KINDS = frozenset(
    {MAPPING_KIND, SUBMISSIONS_KIND, SUBMISSIONS_HISTORY_KIND, COMPANYFACTS_KIND}
)


def verify_sec_stage(sec_details: Mapping[str, Any], *, sec_run: JobRun) -> StageVerificationResult:
    """Bind the SEC mapping asset to the reviewed CIK/fundamentals config,
    then resolve the run's exact canonical SEC asset closure."""
    raw_id = sec_details.get("mapping_asset_id")
    sha256 = sec_details.get("mapping_sha256")
    if not raw_id or not sha256:
        raise RefreshVerificationError(
            "sec_mapping_identity_missing", "SEC stage details have no mapping asset identity"
        )
    try:
        asset_id = uuid.UUID(str(raw_id))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "sec_mapping_identity_malformed", "SEC stage mapping_asset_id is not a valid identifier"
        ) from exc
    asset = DataAsset.objects.filter(pk=asset_id, sha256=sha256).first()
    if asset is None:
        raise RefreshVerificationError(
            "sec_mapping_asset_missing",
            "SEC stage mapping asset could not be resolved with the recorded checksum",
        )
    if (
        asset.provider != sec.PROVIDER
        or asset.kind != MAPPING_KIND
        or asset.subject != MAPPING_SUBJECT
    ):
        raise RefreshVerificationError(
            "sec_mapping_asset_identity_mismatch",
            "SEC stage mapping asset is not the reviewed SEC ticker/exchange mapping",
        )
    cik_config = load_sec_cik_config()
    if asset.sha256 != cik_config.source_sha256:
        raise RefreshVerificationError(
            "sec_mapping_asset_not_reviewed",
            "SEC stage mapping asset is not the reviewed SEC CIK config's pinned source",
        )
    if (
        sec_details.get("cik_config_version") != cik_config.config_version
        or sec_details.get("cik_config_hash") != cik_config.config_hash
    ):
        raise RefreshVerificationError(
            "sec_cik_config_mismatch",
            "SEC stage does not record the reviewed CIK config's exact version/hash",
        )
    fundamentals_config = load_sec_fundamentals_config()
    if (
        sec_details.get("config_version") != fundamentals_config.config_version
        or sec_details.get("config_hash") != fundamentals_config.config_hash
    ):
        raise RefreshVerificationError(
            "sec_fundamentals_config_mismatch",
            "SEC stage does not record the reviewed fundamentals config's exact version/hash",
        )
    sec_boundary = sec_run.finished_at
    if (
        sec_boundary is None
        or asset.available_at > sec_boundary
        or asset.retrieved_at > sec_boundary
    ):
        raise RefreshVerificationError(
            "sec_mapping_asset_after_cutoff",
            "SEC stage mapping asset was admitted after the verified SEC stage's own execution",
        )

    raw_refs = sec_details.get("asset_refs")
    if not isinstance(raw_refs, list) or not raw_refs:
        raise RefreshVerificationError(
            "sec_asset_refs_missing", "SEC stage details have no recorded exact asset references"
        )
    try:
        parsed_refs = tuple(AssetRef.from_json(raw) for raw in raw_refs)
    except RefreshVerificationError as exc:
        raise RefreshVerificationError("sec_asset_ref_malformed", str(exc)) from exc
    if len(set(parsed_refs)) != len(parsed_refs):
        raise RefreshVerificationError(
            "sec_asset_ref_duplicate", "SEC stage exact asset references contain a duplicate entry"
        )
    mapping_ref = asset_ref_for(asset)
    if mapping_ref not in parsed_refs:
        raise RefreshVerificationError(
            "sec_mapping_ref_not_unique",
            "SEC stage exact asset references do not include the reviewed mapping asset "
            "exactly once",
        )

    store = open_asset_store()
    ordered_ciks = tuple(mapping.cik for mapping in cik_config.mappings.values())
    expected_ciks = frozenset(ordered_ciks)
    resolved_refs: list[AssetRef] = [mapping_ref]
    by_kind_cik: dict[tuple[str, str], list[DataAsset]] = defaultdict(list)
    for ref in parsed_refs:
        if ref == mapping_ref:
            continue
        if ref.provider != sec.PROVIDER or ref.kind not in _SEC_ASSET_KINDS:
            raise RefreshVerificationError(
                "sec_asset_ref_identity_mismatch",
                "A recorded SEC asset reference is not an eligible SEC asset kind",
            )
        if ref.subject not in expected_ciks:
            raise RefreshVerificationError(
                "sec_asset_ref_unexpected_cik",
                "SEC stage exact asset references include a CIK outside the reviewed CIK config",
            )
        resolved = resolve_asset_ref(ref, cutoff=sec_boundary)
        by_kind_cik[(ref.kind, ref.subject)].append(resolved)
        resolved_refs.append(ref)

    canonical_body: list[AssetRef] = []
    for cik in ordered_ciks:
        submissions = by_kind_cik.get((SUBMISSIONS_KIND, cik), [])
        if len(submissions) != 1:
            raise RefreshVerificationError(
                "sec_submissions_ref_not_unique",
                "SEC stage does not record exactly one submissions asset for a reviewed CIK",
            )
        companyfacts = by_kind_cik.get((COMPANYFACTS_KIND, cik), [])
        if len(companyfacts) != 1:
            raise RefreshVerificationError(
                "sec_companyfacts_ref_not_unique",
                "SEC stage does not record exactly one companyfacts asset for a reviewed CIK",
            )
        try:
            required_filenames = historical_submission_filenames(
                read_checksummed_bytes(store, submissions[0])
            )
        except SecEvidencePayloadError:
            raise RefreshVerificationError(
                "sec_submissions_evidence_malformed",
                "SEC stage submissions evidence does not have the required shape",
            ) from None
        history_by_filename: dict[str, list[DataAsset]] = defaultdict(list)
        for history_asset in by_kind_cik.get((SUBMISSIONS_HISTORY_KIND, cik), []):
            filename = (
                history_asset.metadata.get(HISTORY_FILENAME_METADATA_KEY)
                if isinstance(history_asset.metadata, dict)
                else None
            )
            if not isinstance(filename, str) or not filename:
                raise RefreshVerificationError(
                    "sec_history_ref_filename_missing",
                    "SEC stage history asset has no identifiable source filename",
                )
            history_by_filename[filename].append(history_asset)
        if set(history_by_filename) - set(required_filenames):
            raise RefreshVerificationError(
                "sec_history_ref_unexpected",
                "SEC stage records a history asset for a filename its own submissions payload "
                "does not require",
            )
        for filename in required_filenames:
            if len(history_by_filename.get(filename, [])) != 1:
                raise RefreshVerificationError(
                    "sec_history_ref_not_unique",
                    "SEC stage does not record exactly one history asset for a required filename",
                )
        canonical_body.append(asset_ref_for(submissions[0]))
        for filename in sorted(required_filenames):
            canonical_body.append(asset_ref_for(history_by_filename[filename][0]))
        canonical_body.append(asset_ref_for(companyfacts[0]))

    canonical_refs = (mapping_ref, *canonical_body)
    if parsed_refs != canonical_refs:
        raise RefreshVerificationError(
            "sec_asset_ref_order_mismatch",
            "SEC stage exact asset references are not in the canonical mapping/CIK/filename order",
        )

    return StageVerificationResult(
        summary={
            "job_run_id": str(sec_run.pk),
            "mapping_asset_id": str(asset.id),
            "cik_count": len(expected_ciks),
            "asset_count": len(resolved_refs),
        },
        asset_refs=canonical_refs,
    )
