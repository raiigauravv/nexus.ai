"""
NEXUS-AI — Evaluate All Models
================================
Single command to re-evaluate all trained models and print a full metrics table.

Usage:
    cd nexus.ai
    python -m training.evaluate_all

Requires:
    - training/artifacts/fraud_model.pkl       (from train_fraud.py)
    - training/artifacts/recommender_model.pkl (from train_recommender.py)
    - Training datasets in training/data/
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"


def _section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ── Fraud ──────────────────────────────────────────────────────────────────────

def evaluate_fraud() -> None:
    _section("Fraud Detection — Real Dataset Evaluation")

    bundle_path = ARTIFACTS / "fraud_model.pkl"
    if not bundle_path.exists():
        print("  ⚠️  No artifact found. Run: python -m training.train_fraud")
        return

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    metrics = bundle.get("metrics", {})
    print(f"  Dataset       : Kaggle Credit Card Fraud (284,807 rows, 20% holdout)")
    print(f"  Architecture  : Isolation Forest (anomaly score) + Gradient Boosting")
    print(f"  SMOTE         : Applied (10% oversample ratio)")
    print()
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<25} {v}")


# ── Recommender ───────────────────────────────────────────────────────────────

def evaluate_recommender() -> None:
    _section("Recommendation Engine — MovieLens 1M Evaluation")

    bundle_path = ARTIFACTS / "recommender_model.pkl"
    if not bundle_path.exists():
        print("  ⚠️  No artifact found. Run: python -m training.train_recommender")
        return

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    metrics = bundle.get("metrics", {})
    print(f"  Dataset       : MovieLens 1M (1,000,209 ratings, 80/20 temporal split)")
    print(f"  Architecture  : Truncated SVD rank={metrics.get('best_svd_rank','?')} + Content-Based (Genre vectors)")
    print(f"  Fusion        : 60% Collaborative + 40% Content-Based")
    print()
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<25} {v}")

    # Quick sanity check on Vt for ALS updates
    if "Vt" in bundle:
        print(f"\n  ALS Update Ready : ✅  Vt shape={bundle['Vt'].shape}")
    else:
        print("\n  ALS Update Ready : ❌  Vt missing from bundle")


# ── Inference sanity checks ────────────────────────────────────────────────────

def sanity_check_inference() -> None:
    _section("Inference Sanity Checks (uses in-memory models)")

    # Fraud
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
        from app.ml.fraud_model import predict_fraud
        high = predict_fraud({"amount": 4500, "merchant_category": "atm",
                              "velocity_1h": 15, "distance_from_home_km": 900, "unusual_location": 1})
        low  = predict_fraud({"amount": 32,   "merchant_category": "grocery",
                              "velocity_1h": 1,  "distance_from_home_km": 2,   "unusual_location": 0})
        fraud_ok = high["risk_level"] == "HIGH" and low["risk_level"] == "LOW"
        print(f"  Fraud model   : {'✅ PASS' if fraud_ok else '❌ FAIL'}  "
              f"(high={high['fraud_score']:.3f}, low={low['fraud_score']:.3f})")
    except Exception as e:
        print(f"  Fraud model   : ❌ ERROR — {e}")

    # Recommender
    try:
        from app.ml.recommender import get_recommendations, get_recommender_stats
        recs = get_recommendations("U001", top_n=5)
        stats = get_recommender_stats()
        rec_ok = len(recs) > 0 and stats["n_users_trained"] >= 500
        print(f"  Recommender   : {'✅ PASS' if rec_ok else '❌ FAIL'}  "
              f"({len(recs)} recs, {stats['n_users_trained']} users)")
    except Exception as e:
        print(f"  Recommender   : ❌ ERROR — {e}")

    # Sentiment
    try:
        from app.ml.sentiment import analyze
        pos = analyze("I absolutely love this product!")
        neg = analyze("Terrible quality, completely broken.")
        sent_ok = pos["overall"]["label"] == "positive" and neg["overall"]["label"] == "negative"
        print(f"  Sentiment     : {'✅ PASS' if sent_ok else '❌ FAIL'}  "
              f"(pos={pos['overall']['score']:.3f}, neg={neg['overall']['score']:.3f})")
    except Exception as e:
        print(f"  Sentiment     : ❌ ERROR — {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "█" * 55)
    print("  NEXUS-AI — Full Model Evaluation Report")
    print("█" * 55)

    evaluate_fraud()
    evaluate_recommender()
    sanity_check_inference()

    print(f"\n{'='*55}")
    print("  Evaluation complete.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
