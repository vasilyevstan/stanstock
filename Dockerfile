FROM ghcr.io/astral-sh/uv:0.12.13 AS uv
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    && rm -rf /root/.cache/uv

COPY . .
RUN uv sync --frozen --no-dev \
    && rm -rf /root/.cache/uv

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME="/home/stanstock" \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system --gid 10001 stanstock \
    && useradd --system --uid 10001 --gid stanstock \
        --create-home --home-dir /home/stanstock stanstock \
    && apt-get update \
    && apt-get install --yes --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/var/backups /app/var/data /app/var/static \
    && chown -R stanstock:stanstock /app/var /home/stanstock

USER stanstock

HEALTHCHECK --interval=10s --timeout=4s --retries=12 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--worker-class", "gthread", "--access-logfile", "-", "stanstock.wsgi:application"]
