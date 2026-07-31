# Convenience targets that wrap the commands documented in README.md and
# CONTRIBUTING.md. Run `make help` to list them. Recipes call the backend
# virtual environment directly (backend/.venv/bin/...) so targets work from
# a fresh shell without first sourcing an activation script.

.PHONY: help venv install-backend install-frontend ingest ingest-fast \
	run-backend run-frontend lint test build check-links verify

help:
	@echo "make venv               create backend/.venv"
	@echo "make install-backend    install backend dependencies into .venv"
	@echo "make install-frontend   npm ci in frontend/"
	@echo "make ingest             download Flickr8k, build the SQLite DB and embeddings"
	@echo "make ingest-fast        ingest --skip-embeddings (keyword search only)"
	@echo "make run-backend        start FastAPI on :8000"
	@echo "make run-frontend       start the Vite dev server on :5173"
	@echo "make lint               ruff check app tests"
	@echo "make test               pytest -q"
	@echo "make build              tsc && vite build"
	@echo "make check-links        verify relative Markdown links"
	@echo "make verify             lint + test + build + check-links (what CI runs)"

venv:
	cd backend && python3 -m venv .venv

install-backend: venv
	backend/.venv/bin/python -m pip install -r backend/requirements.txt

install-frontend:
	cd frontend && npm ci

ingest:
	cd backend && .venv/bin/python -m app.ingest

ingest-fast:
	cd backend && .venv/bin/python -m app.ingest --skip-embeddings

run-backend:
	cd backend && .venv/bin/python -m uvicorn app.main:app --port 8000

run-frontend:
	cd frontend && npm run dev -- --port 5173

lint:
	cd backend && .venv/bin/ruff check app tests

test:
	cd backend && .venv/bin/python -m pytest -q

build:
	cd frontend && npm run build

check-links:
	backend/.venv/bin/python scripts/check_links.py

verify: lint test build check-links
