#!/usr/bin/env python3
"""
pytest tests for NEXUS-AI core prediction endpoints.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np


# ── Sentiment ──────────────────────────────────────────────────────────────────
class TestSentiment:
    def test_positive_text(self):
        from app.ml.sentiment import analyze
        result = analyze("I absolutely love this product! It works amazingly well.")
        assert result["overall"]["label"] == "positive"
        assert result["overall"]["score"] > 0

    def test_negative_text(self):
        from app.ml.sentiment import analyze
        result = analyze("This is terrible. Worst purchase I've ever made. Broken in 2 days.")
        assert result["overall"]["label"] == "negative"
        assert result["overall"]["score"] < 0

    def test_result_structure(self):
        from app.ml.sentiment import analyze
        result = analyze("The product is okay.")
        assert "overall" in result
        assert "emotions" in result
        assert "aspects" in result
        assert "metadata" in result
        assert "model_info" in result
        assert "score" in result["overall"]
        assert "label" in result["overall"]

    def test_confidence_range(self):
        from app.ml.sentiment import analyze
        result = analyze("Pretty good product overall.")
        conf = result["overall"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_sample_reviews_export(self):
        from app.ml.sentiment import SAMPLE_REVIEWS
        assert len(SAMPLE_REVIEWS) >= 5
        assert all("id" in r and "text" in r for r in SAMPLE_REVIEWS)


# ── Fraud ──────────────────────────────────────────────────────────────────────
class TestFraud:
    def test_high_risk_transaction(self):
        from app.ml.fraud_model import predict_fraud
        result = predict_fraud({
            "amount": 4500, "merchant_category": "atm",
            "velocity_1h": 10, "distance_from_home_km": 800, "unusual_location": 1,
        })
        assert result["risk_level"] == "HIGH"
        assert result["fraud_score"] > 0.5

    def test_low_risk_transaction(self):
        from app.ml.fraud_model import predict_fraud
        result = predict_fraud({
            "amount": 35, "merchant_category": "grocery",
            "velocity_1h": 1, "distance_from_home_km": 2, "unusual_location": 0,
        })
        assert result["risk_level"] == "LOW"
        assert result["fraud_score"] < 0.5

    def test_result_structure(self):
        from app.ml.fraud_model import predict_fraud
        result = predict_fraud({"amount": 100, "merchant_category": "restaurant"})
        assert "fraud_score" in result
        assert "is_fraud" in result
        assert "risk_level" in result
        assert "reasons" in result
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_score_range(self):
        from app.ml.fraud_model import predict_fraud
        result = predict_fraud({"amount": 500, "merchant_category": "online_retail"})
        assert 0.0 <= result["fraud_score"] <= 1.0


# ── Recommender ────────────────────────────────────────────────────────────────
class TestRecommender:
    def test_returns_recommendations(self):
        from app.ml.recommender import get_recommendations
        recs = get_recommendations("U001", top_n=5)
        assert len(recs) > 0
        assert len(recs) <= 5

    def test_recommendation_structure(self):
        from app.ml.recommender import get_recommendations
        recs = get_recommendations("U001", top_n=3)
        for r in recs:
            assert "id" in r
            assert "name" in r
            assert "recommendation_score" in r
            assert "match_reason" in r

    def test_score_range(self):
        from app.ml.recommender import get_recommendations
        recs = get_recommendations("U002", top_n=6)
        for r in recs:
            assert 0.0 <= r["recommendation_score"] <= 1.0

    def test_different_users_different_results(self):
        from app.ml.recommender import get_recommendations
        recs_tech  = get_recommendations("U001", top_n=3)  # tech professional
        recs_gamer = get_recommendations("U003", top_n=3)  # gamer
        ids_tech  = {r["id"] for r in recs_tech}
        ids_gamer = {r["id"] for r in recs_gamer}
        # Results should differ for different personas
        assert ids_tech != ids_gamer

    def test_trending_returns_items(self):
        from app.ml.recommender import get_trending
        trending = get_trending(top_n=5)
        assert len(trending) == 5
        for item in trending:
            assert "trending_score" in item

    def test_training_matrix_size(self):
        from app.ml.recommender import get_recommender_stats
        stats = get_recommender_stats()
        assert stats["n_users_trained"] >= 500
        assert stats["n_products"] >= 80    # 80 products in expanded catalog
        assert stats["n_interactions"] >= 1000


# ── Visual Search ──────────────────────────────────────────────────────────────
class TestVisualSearch:
    def test_search_returns_results(self):
        from app.ml.visual_search import search_by_description
        results = search_by_description("wireless headphones", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_result_structure(self):
        from app.ml.visual_search import search_by_description
        results = search_by_description("gaming mouse", top_k=2)
        for r in results:
            assert "name" in r
            assert "similarity" in r
            assert "category" in r

    def test_similarity_range(self):
        from app.ml.visual_search import search_by_description
        results = search_by_description("laptop computer", top_k=3)
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0
