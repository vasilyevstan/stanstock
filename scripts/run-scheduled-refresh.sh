#!/bin/sh
set -eu

umask 077
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_FILE="${STANSTOCK_ENV_FILE:-$PROJECT_ROOT/.env}"
SCHEDULE_TIMEZONE="${STANSTOCK_SCHEDULE_TIMEZONE:-}"

if [ ! -f "$ENV_FILE" ]; then
  echo "StanStock scheduled refresh requires $ENV_FILE" >&2
  exit 1
fi

ENV_MODE=$(stat -f '%Lp' "$ENV_FILE")
case "$ENV_MODE" in
  400|600) ;;
  *)
    echo "StanStock scheduled refresh requires owner-only permissions on $ENV_FILE" >&2
    exit 1
    ;;
esac

set -a
. "$ENV_FILE"
set +a

export STANSTOCK_DISABLE_KEYCHAIN=1
export STANSTOCK_SCHEDULE_TIMEZONE="$SCHEDULE_TIMEZONE"
export PYTHONUNBUFFERED=1

exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/manage.py" scheduled_refresh
