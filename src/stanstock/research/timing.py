from __future__ import annotations

from datetime import date, datetime

from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.data.models import Region, UniverseSnapshot


def is_us_session_issuance_on_time(
    *,
    target_date: date,
    generated_at: datetime,
) -> bool:
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    if target_date > generated_at.date():
        return False
    if target_date == generated_at.date():
        return True
    calendar = get_calendar("XNYS")
    if not calendar.is_session(target_date.isoformat()):
        raise ValueError(f"US target date {target_date.isoformat()} is not an XNYS session")
    next_session = calendar.next_session(target_date.isoformat())
    next_session_open: datetime = calendar.session_open(next_session).to_pydatetime()
    return generated_at < next_session_open


def is_observed_issuance_on_time(
    snapshot: UniverseSnapshot,
    *,
    target_date: date,
    generated_at: datetime,
) -> bool:
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    if target_date > generated_at.date():
        return False
    if target_date == generated_at.date():
        return True

    regions = set(
        snapshot.memberships.filter(eligible=True).values_list(
            "listing__region",
            flat=True,
        )
    )
    if regions != {Region.US}:
        raise ValueError(
            "Overnight on-time issuance requires one supported regional market calendar"
        )
    return is_us_session_issuance_on_time(
        target_date=target_date,
        generated_at=generated_at,
    )
