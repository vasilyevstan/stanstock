#!/bin/sh
set -eu

python manage.py migrate --noinput
python manage.py ensure_data_dir
python manage.py collectstatic --noinput

if [ -n "${STANSTOCK_OWNER_PASSWORD:-}" ]; then
  python manage.py bootstrap_owner
fi

if [ "${STANSTOCK_DEMO_MODE:-false}" = "true" ]; then
  python manage.py refresh_demo
fi

exec "$@"
