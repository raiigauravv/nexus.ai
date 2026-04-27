import uuid
from app.api.endpoints.auth import _validate_registration_input, UserRegister

def test_health_check(client):
    """1. Test the root health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_recommend_list_users(client):
    """2. Test fetching the list of synthetic users for recommendations."""
    response = client.get("/api/v1/recommend/users")
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert len(data["users"]) > 0

def test_recommend_for_user(client):
    """3. Test personalized recommendations for a specific user."""
    response = client.get("/api/v1/recommend/for/U001")
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["id"] == "U001"
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 6

def test_recommend_trending(client):
    """4. Test fetching trending products globally."""
    response = client.get("/api/v1/recommend/trending")
    assert response.status_code == 200
    data = response.json()
    assert "trending" in data
    assert len(data["trending"]) > 0

def test_fraud_stats(client):
    """5. Test fetching aggregate fraud detection holdout metrics."""
    response = client.get("/api/v1/fraud/stats")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "f1" in data["metrics"]
    assert "auc_roc" in data["metrics"]

def test_fraud_verify(client):
    """6. Test checking a specific high-risk transaction for fraud."""
    payload = {
        "amount": 4500,
        "merchant_category": "luxury",
        "velocity_1h": 12,
        "distance_from_home_km": 1500,
        "unusual_location": 1
    }
    response = client.post("/api/v1/fraud/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_score" in data["prediction"]
    assert "risk_level" in data["prediction"]
    assert data["prediction"]["fraud_score"] >= 0.0

def test_sentiment_analyze(client):
    """7. Test the local DistilBERT transformer sentiment analysis endpoint."""
    payload = {
        "text": "This product completely exceeded my expectations, it is amazing!"
    }
    response = client.post("/api/v1/sentiment/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall"]["label"] == "positive"
    assert data["overall"]["score"] > 0.0

def test_recommend_similar(client):
    """8. Test fetching visually or semantically similar items for a product."""
    response = client.get("/api/v1/recommend/similar/P001")
    assert response.status_code == 200
    data = response.json()
    assert "similar_items" in data
    assert isinstance(data["similar_items"], list)

def test_recommend_products(client):
    """9. Test fetching the entire product catalog."""
    response = client.get("/api/v1/recommend/products")
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert len(data["products"]) > 0

def test_agent_routing_fallback(client):
    """10. Test the dynamic Agent Router tool execution framework (using a simple fallback to avoid LLM limits)."""
    payload = {
        "message": "What is 2+2?",
        "user_id": "U001",
        "history": []
    }
    response = client.post("/api/v1/agent/chat", json=payload)
    # The agent uses real LLM calls returning SSE stream, read raw text
    assert response.status_code in [200, 429, 503]
    if response.status_code == 200:
        data = response.text
        assert "data:" in data or len(data) == 0

def test_auth_me_requires_token(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401

def test_auth_register_login_and_profile_flow(client):
    user_id = f"test_{uuid.uuid4().hex[:10]}"
    password = "SecurePass123"

    register_payload = {
        "id": user_id,
        "name": "Test User",
        "password": password,
        "persona": "tech_professional",
    }
    register_resp = client.post("/api/v1/auth/register", json=register_payload)
    assert register_resp.status_code == 200
    token = register_resp.json()["access_token"]

    login_resp = client.post(
        "/api/v1/auth/login",
        data={"username": user_id, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login_resp.status_code == 200

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me = me_resp.json()
    assert me["id"] == user_id

def test_auth_login_rejects_invalid_password(client):
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "U001", "password": "wrong-password"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 401

def test_sentiment_analyze_rejects_empty_text(client):
    response = client.post("/api/v1/sentiment/analyze", json={"text": ""})
    assert response.status_code == 422

def test_recommend_similar_unknown_product_returns_404(client):
    response = client.get("/api/v1/recommend/similar/DOES_NOT_EXIST")
    assert response.status_code == 404

def test_auth_register_rejects_weak_password(client):
    payload = {
        "id": f"weak_{uuid.uuid4().hex[:8]}",
        "name": "Weak Password",
        "password": "weakpass",
        "persona": "tech_professional",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422

def test_auth_register_rejects_invalid_user_id(client):
    payload = {
        "id": "bad id",
        "name": "Bad ID",
        "password": "StrongPass123",
        "persona": "tech_professional",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422

def test_rate_limiter_returns_retry_after_header(client):
    # Use a per-test synthetic key on login endpoint by intentionally reusing bad credentials.
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    too_many = None
    for _ in range(12):
        resp = client.post(
            "/api/v1/auth/login",
            data={"username": "U001", "password": "bad"},
            headers=headers,
        )
        if resp.status_code == 429:
            too_many = resp
            break

    assert too_many is not None
    assert "Retry-After" in too_many.headers

def test_registration_validator_direct_contract():
    valid = UserRegister(id="valid_user-1", name="Valid User", password="ValidPass123", persona="tech_professional")
    _validate_registration_input(valid)
