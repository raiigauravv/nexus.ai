.PHONY: up down build test test-v logs clean shell

# Docker management
up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f backend

# Testing
test:
	cd backend && export MLFLOW_TRACKING_URI=sqlite:///mlruns.db && PYTHONPATH=. venv/bin/pytest tests/test_api_endpoints.py

test-v:
	cd backend && export MLFLOW_TRACKING_URI=sqlite:///mlruns.db && PYTHONPATH=. venv/bin/pytest tests/test_api_endpoints.py -v

# Utility
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf backend/.pytest_cache
	rm -f backend/nexus_fallback.db
	rm -f backend/mlruns.db

shell:
	cd backend && source venv/bin/activate
