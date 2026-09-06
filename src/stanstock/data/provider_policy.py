from __future__ import annotations

from django.contrib.auth import get_user_model

from stanstock.data.models import ProviderRecord
from stanstock.data.providers.exceptions import ProviderConfigurationError

PRIVATE_USAGE_SCOPE = "personal_internal_display_authorized"
BASIC_USAGE_SCOPE = "personal_single_user_noncommercial"
DISPLAY_PLANS = frozenset({"grow", "pro", "ultra", "custom"})


def validate_provider_usage(record: ProviderRecord) -> None:
    plan = str(record.metadata.get("plan") or "").lower()
    if plan == "basic":
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
