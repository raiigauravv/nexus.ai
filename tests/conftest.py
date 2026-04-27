import os

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
os.environ.setdefault("POSTGRES_PASSWORD", "test-postgres-password")


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_rate_limiter_state():
    from app.rate_limiter import rate_limiter

    rate_limiter.reset()
    yield
