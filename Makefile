.PHONY: up down build migrate test test-v logs clean shell

# Docker management
up:
	docker-compose up -d
	docker-compose exec backend alembic -c alembic.ini upgrade head

down:
	docker-compose down

build:
	docker-compose build

migrate:
	cd backend && export ALEMBIC_DATABASE_URL=sqlite:///./nexus_fallback.db && venv/bin/alembic -c alembic.ini upgrade head

logs:
	docker-compose logs -f backend

# Testing
test:
	cd backend && export ALEMBIC_DATABASE_URL=sqlite:///./nexus_fallback.db && venv/bin/alembic -c alembic.ini upgrade head && export MLFLOW_TRACKING_URI=sqlite:///mlruns.db && PYTHONPATH=. venv/bin/pytest tests/test_api_endpoints.py

test-v:
	cd backend && export ALEMBIC_DATABASE_URL=sqlite:///./nexus_fallback.db && venv/bin/alembic -c alembic.ini upgrade head && export MLFLOW_TRACKING_URI=sqlite:///mlruns.db && PYTHONPATH=. venv/bin/pytest tests/test_api_endpoints.py -v

# Utility
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf backend/.pytest_cache
	rm -f backend/nexus_fallback.db
	rm -f backend/mlruns.db

shell:
	cd backend && source venv/bin/activate
