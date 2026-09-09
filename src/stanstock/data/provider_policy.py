from __future__ import annotations

from django.contrib.auth import get_user_model

from stanstock.data.models import ProviderRecord
from stanstock.data.providers import sec, twelve_data
from stanstock.data.providers.exceptions import ProviderConfigurationError

#: Canonical SEC provider identifier, re-exported at the provider-policy
#: boundary so research code can qualify SEC evidence without importing the
#: raw provider client or duplicating its literal.
SEC_PROVIDER = sec.PROVIDER

#: Canonical Twelve Data provider identifier, re-exported here so callers
#: outside the `data.providers` package -- which must not import raw provider
#: modules -- can still name the provider without a duplicated literal.
TWELVE_DATA_PROVIDER = twelve_data.PROVIDER

PRIVATE_USAGE_SCOPE = "personal_internal_display_authorized"
BASIC_USAGE_SCOPE = "personal_single_user_noncommercial"
BASIC_PLAN = "basic"
DISPLAY_PLANS = frozenset({"grow", "pro", "ultra", "custom"})

#: Named provider capability for split and reverse-split event evidence.
SPLIT_EVENT_CAPABILITY = "corporate_actions_splits"

#: The only capability state this version can return. No reviewed
#: corporate-actions source is integrated anywhere in the codebase, so a
#: verified or available branch would be a claim nothing can support.
CAPABILITY_UNAVAILABLE = "unavailable"

#: The recorded Twelve Data plan is Basic, which is not entitled to a
#: corporate-actions feed. Upgrading the plan alone would not supply a
#: reviewed source; the source capability and licensing review still gates it.
CAPABILITY_PLAN_NOT_ENTITLED = "provider_plan_not_entitled"

#: No reviewed corporate-actions source exists for this provider at all --
#: including when no plan was recorded, the recorded plan is malformed, or the
#: provider is not Twelve Data.
CAPABILITY_NO_REVIEWED_SOURCE = "no_reviewed_corporate_actions_source"


def normalized_provider_plan(plan: str | None) -> str | None:
    """Lower-cased, whitespace-stripped plan label, or ``None`` when absent.

    An empty or whitespace-only label is not a recorded plan.
    """
    if plan is None:
        return None
    normalized = plan.strip().lower()
    return normalized or None


def split_event_capability(provider: str, plan: str | None = None) -> tuple[str, str]:
    """Whether verified split/reverse-split evidence can be obtained.

    Pure: it performs no database query, no network call, and no inference
    from adjusted prices, share-count discontinuities, or SEC facts. It
    reports the recorded entitlement position and nothing else, and it can
    never return an available or verified state.
    """
    if (
        provider.strip().lower() == TWELVE_DATA_PROVIDER
        and normalized_provider_plan(plan) == BASIC_PLAN
    ):
        return (CAPABILITY_UNAVAILABLE, CAPABILITY_PLAN_NOT_ENTITLED)
    return (CAPABILITY_UNAVAILABLE, CAPABILITY_NO_REVIEWED_SOURCE)


def validate_provider_usage(record: ProviderRecord) -> None:
    plan = str(record.metadata.get("plan") or "").lower()
    if plan == BASIC_PLAN:
        if (
            record.usage_scope != BASIC_USAGE_SCOPE
            or record.metadata.get("personal_noncommercial_confirmed") is not True
        ):
            raise ProviderConfigurationError(
                "Twelve Data Basic activation has no recorded personal, "
                "single-user, non-commercial authorization."
            )
        licensed_user_id = str(record.metadata.get("licensed_user_id") or "")
        active_user_ids = [
            str(value)
            for value in get_user_model()
            .objects.filter(is_active=True)
            .order_by("pk")
            .values_list("pk", flat=True)[:2]
        ]
        if active_user_ids != [licensed_user_id]:
            raise ProviderConfigurationError(
                "Twelve Data Basic access is restricted to the one licensed active "
                "StanStock user. Disable additional users or use a plan/agreement "
                "that covers the intended audience."
            )
        return
    if plan not in DISPLAY_PLANS:
        raise ProviderConfigurationError(
            "Twelve Data ProviderRecord metadata has no supported usage plan"
        )
    if (
        record.usage_scope != PRIVATE_USAGE_SCOPE
        or record.metadata.get("internal_display_rights_confirmed") is not True
    ):
        raise ProviderConfigurationError(
            "Twelve Data activation has no recorded internal-display entitlement. "
            "Re-run configure_twelve_data --enable with the required plan and "
            "rights confirmation."
        )
