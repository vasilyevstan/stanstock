UV ?= uv

.PHONY: dev down logs migrate owner demo backup verify-assets test lint typecheck check

dev:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f web

migrate:
	$(UV) run python manage.py migrate

owner:
	$(UV) run python manage.py bootstrap_owner

demo:
	$(UV) run python manage.py refresh_demo

backup:
	$(UV) run python manage.py backup

verify-assets:
	$(UV) run python manage.py verify_assets

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy src

check: lint typecheck test
