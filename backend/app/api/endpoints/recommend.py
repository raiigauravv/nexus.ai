"""
Recommendation Engine API Endpoints
GET /api/v1/recommend/users                     — list all synthetic users
GET /api/v1/recommend/products                  — list all products
GET /api/v1/recommend/for/{user_id}             — personalized recommendations (cross-module enhanced)
GET /api/v1/recommend/similar/{product_id}      — similar items
GET /api/v1/recommend/trending                  — trending products
GET /api/v1/recommend/product-sentiment/{id}    — sentiment health for a single product
GET /api/v1/recommend/category-complaints/{cat} — cross-module complaint analysis
"""
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.ml.recommender import (
    get_recommendations,
    get_similar_items,
    get_trending,
    get_recommender,
    USERS,
    PRODUCTS,
)
from app.ml.cross_module import (
    get_sentiment_adjusted_recommendations,
    get_product_sentiment_health,
    explain_complaints_for_category,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Pre-warm model on import
try:
    get_recommender()
    logger.info("Recommendation model ready.")
except Exception as e:
    logger.warning(f"Could not pre-warm recommender: {e}")


@router.get("/recommend/users")
async def list_users():
    """Return all synthetic users."""
    return {"users": USERS}


@router.get("/recommend/products")
async def list_products():
    """Return all products."""
    return {"products": PRODUCTS}


@router.get("/recommend/for/{user_id}")
async def recommend_for_user(
    user_id: str,
    top_n: int = 6,
    fraud_risk: Optional[str] = Query(None, description="LOW | MEDIUM | HIGH"),
    cross_module: bool = Query(True, description="Enable sentiment + fraud cross-module adjustments"),
):
    """
    Personalized recommendations with optional cross-module intelligence:
    - cross_module=true: blends in product sentiment health + fraud risk adjustment
    - fraud_risk=HIGH: de-ranks luxury items, adds fraud flag
    """
    user = next((u for u in USERS if u["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    if cross_module:
        recs = get_sentiment_adjusted_recommendations(
            user_id, top_n=top_n, fraud_risk=fraud_risk
        )
        algorithm = "Hybrid SVD (60%) + Content-Based (40%) + Sentiment Health (20%) + Fraud Adjustment"
    else:
        recs = get_recommendations(user_id, top_n=top_n)
        algorithm = "Hybrid SVD (60%) + Content-Based (40%)"

    return {
        "user": user,
        "recommendations": recs,
        "algorithm": algorithm,
        "cross_module_active": cross_module,
        "fraud_risk_applied": fraud_risk,
    }


@router.get("/recommend/similar/{product_id}")
async def similar_items(product_id: str, top_n: int = 4):
    """Return items similar to the given product."""
    items = get_similar_items(product_id, top_n=top_n)
    if not items:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    return {"seed_product": product, "similar_items": items}


@router.get("/recommend/trending")
async def trending(top_n: int = 8):
    """Return trending products."""
    items = get_trending(top_n=top_n)
    return {"trending": items}


@router.get("/recommend/product-sentiment/{product_id}")
async def product_sentiment_health(product_id: str):
    """Get the sentiment health score for a specific product based on its reviews."""
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")
    health = get_product_sentiment_health(product_id)
    label = "Loved" if health >= 0.70 else "Mixed" if health >= 0.50 else "Poor reviews"
    return {
        "product": product,
        "sentiment_health": health,
        "sentiment_health_pct": round(health * 100, 1),
        "label": label,
    }


@router.get("/recommend/category-complaints/{category}")
async def category_complaints(category: str):
    """Cross-module: analyze review sentiment for all products in a category."""
    try:
        analysis = explain_complaints_for_category(category)
        return {"category": category, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
