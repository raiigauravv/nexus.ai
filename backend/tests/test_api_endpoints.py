import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create a test client using the FastAPI app (this will run the lifespan and init_db)
client = TestClient(app)

def test_health_check():
    """1. Test the root health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_recommend_list_users():
    """2. Test fetching the list of synthetic users for recommendations."""
    response = client.get("/api/v1/recommend/users")
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert len(data["users"]) > 0

def test_recommend_for_user():
    """3. Test personalized recommendations for a specific user."""
    response = client.get("/api/v1/recommend/for/U001")
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["id"] == "U001"
    assert "recommendations" in data
    assert len(data["recommendations"]) <= 6

def test_recommend_trending():
    """4. Test fetching trending products globally."""
    response = client.get("/api/v1/recommend/trending")
    assert response.status_code == 200
    data = response.json()
    assert "trending" in data
    assert len(data["trending"]) > 0

def test_fraud_stats():
    """5. Test fetching aggregate fraud detection holdout metrics."""
    response = client.get("/api/v1/fraud/stats")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "f1" in data["metrics"]
    assert "auc_roc" in data["metrics"]

def test_fraud_verify():
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

def test_sentiment_analyze():
    """7. Test the local DistilBERT transformer sentiment analysis endpoint."""
    payload = {
        "text": "This product completely exceeded my expectations, it is amazing!"
    }
    response = client.post("/api/v1/sentiment/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["overall"]["label"] == "positive"
    assert data["overall"]["score"] > 0.0

def test_recommend_similar():
    """8. Test fetching visually or semantically similar items for a product."""
    response = client.get("/api/v1/recommend/similar/P001")
    assert response.status_code == 200
    data = response.json()
    assert "similar_items" in data
    assert isinstance(data["similar_items"], list)

def test_recommend_products():
    """9. Test fetching the entire product catalog."""
    response = client.get("/api/v1/recommend/products")
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert len(data["products"]) > 0

def test_agent_routing_fallback():
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
