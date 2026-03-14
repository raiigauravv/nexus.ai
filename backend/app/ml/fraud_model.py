"""
Fraud Detection ML Model
Ensemble of Isolation Forest (anomaly detection) + Gradient Boosting (classification)
Trained on synthetic transaction data generated at startup.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import random
import datetime
import logging

logger = logging.getLogger(__name__)

# ── Global singleton ───────────────────────────────────────────────────────────
_model_bundle: dict | None = None

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_retail", "travel",
    "electronics", "pharmacy", "entertainment", "atm", "luxury"
]

CATEGORY_RISK = {
    "grocery": 0.05,
    "gas_station": 0.15,
    "restaurant": 0.08,
    "online_retail": 0.25,
    "travel": 0.30,
    "electronics": 0.40,
    "pharmacy": 0.07,
    "entertainment": 0.12,
    "atm": 0.35,
    "luxury": 0.45,
}

def _generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate realistic synthetic transaction data with labeled fraud."""
    random.seed(42)
    np.random.seed(42)

    records = []
    for i in range(n_samples):
        is_fraud = random.random() < 0.08  # 8% fraud rate

        category = random.choice(MERCHANT_CATEGORIES)
        hour = random.randint(0, 23)
        day_of_week = random.randint(0, 6)
        unusual_location = 0
        
        if is_fraud:
            # Most fraud is suspicious, but introduce "stealthy" fraud (noise)
            if random.random() < 0.20: # 20% stealthy fraud
                amount = random.uniform(20, 100)
                hour = random.randint(9, 17) # normal hours
                velocity = random.randint(1, 4)
                distance_from_home = random.uniform(0, 30)
                unusual_location = 0
            else:
                amount = random.choice([
                    random.uniform(500, 5000),   # large unusual amount
                    random.uniform(0.01, 2.0),   # micro-test transactions
                ])
                hour = random.choice(list(range(0, 6)) + list(range(23, 24)))  # odd hours
                category = random.choice(["atm", "luxury", "electronics", "online_retail"])
                velocity = random.randint(5, 20)         # many transactions in short window
                distance_from_home = random.uniform(100, 5000)
                unusual_location = 1
        else:
            # Most legit is normal, but introduce "suspicious-looking" legit (noise)
            if random.random() < 0.05: # 5% suspicious-looking legitimate
                amount = random.uniform(600, 1200)
                hour = 3 # midnight
                velocity = 5
                distance_from_home = 300
                unusual_location = 1
                category = "luxury"
            else:
                amount = abs(np.random.lognormal(mean=3.5, sigma=1.0))  # ~$33 median
                amount = min(amount, 800)
                velocity = random.randint(0, 4)
                distance_from_home = random.uniform(0, 50)
                unusual_location = 0

        records.append({
            "amount": amount,
            "hour": hour,
            "day_of_week": day_of_week,
            "category_risk": CATEGORY_RISK.get(category, 0.1),
            "velocity_1h": velocity,
            "distance_from_home_km": distance_from_home,
            "unusual_location": unusual_location,
            "is_weekend": int(day_of_week >= 5),
            "is_night": int(hour < 6 or hour >= 22),
            "is_fraud": int(is_fraud),
        })

    return pd.DataFrame(records)

def train_model() -> dict:
    """Train the fraud detection ensemble and return model bundle."""
    logger.info("Training fraud detection model on synthetic data...")

    df = _generate_synthetic_data(5000)
    
    feature_cols = [
        "amount", "hour", "day_of_week", "category_risk",
        "velocity_1h", "distance_from_home_km", "unusual_location",
        "is_weekend", "is_night",
    ]
    
    X = df[feature_cols].values
    y = df["is_fraud"].values

    # Stratified split to ensure fraud representation in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. Isolation Forest for unsupervised anomaly feature engineering
    iso_forest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    iso_forest.fit(X_train)
    
    # Generate IF scores for both train and test
    iso_score_train = iso_forest.score_samples(X_train).reshape(-1, 1)
    iso_score_test = iso_forest.score_samples(X_test).reshape(-1, 1)

    # 2. Gradient Boosting for supervised classification (with IF score as feature)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Append the IF score as the final feature
    X_train_final = np.hstack([X_train_scaled, iso_score_train])
    X_test_final = np.hstack([X_test_scaled, iso_score_test])
    
    gb_clf = GradientBoostingClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42
    )
    gb_clf.fit(X_train_final, y_train)

    # Evaluate the ensemble on STRICTLY HELD-OUT data
    gb_proba = gb_clf.predict_proba(X_test_final)[:, 1]
    y_pred = (gb_proba > 0.50).astype(int)

    metrics = {
        "f1": float(round(f1_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred), 4)),
        "recall": float(round(recall_score(y_test, y_pred), 4)),
        "auc_roc": float(round(roc_auc_score(y_test, gb_proba), 4)),
    }

    bundle = {
        "iso_forest": iso_forest,
        "gb_clf": gb_clf,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }

    # --- MLFLOW LOGGING ---
    try:
        import mlflow
        import os
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("Nexus_Fraud_Detection")
        with mlflow.start_run(run_name="feature_ensemble_train"):
            mlflow.set_tag("architecture", "IF_Score_as_Feature_GBM")
            mlflow.log_params({
                "iso_contaimination": 0.08,
                "gb_estimators": 120,
                "gb_max_depth": 5,
                "test_size": 0.2
            })
            mlflow.log_metrics(metrics)
            logger.info("Successfully logged professional ensemble metrics to MLflow.")
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

    print(f"--- FRAUD MODEL METRICS ---")
    print(f"F1: {metrics['f1']}, AUC-ROC: {metrics['auc_roc']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}")
    return bundle


def get_model() -> dict:
    """Get or lazily initialize the model bundle."""
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = train_model()
    return _model_bundle


def extract_features(transaction: dict) -> np.ndarray:
    """Extract feature vector from a transaction dict."""
    category = transaction.get("merchant_category", "grocery")
    amount = float(transaction.get("amount", 0))
    timestamp = transaction.get("timestamp", datetime.datetime.now().isoformat())
    
    try:
        dt = datetime.datetime.fromisoformat(timestamp)
    except Exception:
        dt = datetime.datetime.now()

    hour = dt.hour
    day_of_week = dt.weekday()

    features = np.array([[
        amount,
        hour,
        day_of_week,
        CATEGORY_RISK.get(category, 0.1),
        float(transaction.get("velocity_1h", 1)),
        float(transaction.get("distance_from_home_km", 10)),
        int(transaction.get("unusual_location", 0)),
        int(day_of_week >= 5),
        int(hour < 6 or hour >= 22),
    ]])
    return features


def predict_fraud(transaction: dict) -> dict:
    """
    Run fraud prediction on a transaction.
    Returns fraud_score (0-1), is_fraud (bool), confidence, and explanation.
    """
    bundle = get_model()
    features = extract_features(transaction)
    
    # 1. Isolation Forest feature engineering (unsupervised)
    iso_score = float(bundle["iso_forest"].score_samples(features)[0])

    # 2. Gradient Boosting classification (with IF score as feature)
    features_scaled = bundle["scaler"].transform(features)
    features_final = np.hstack([features_scaled, [[iso_score]]])
    
    fraud_score = float(bundle["gb_clf"].predict_proba(features_final)[0, 1])
    is_fraud = fraud_score > 0.50

    # Build human-readable explanation
    reasons = []
    amount = float(transaction.get("amount", 0))
    velocity = int(transaction.get("velocity_1h", 1))
    distance = float(transaction.get("distance_from_home_km", 10))
    category = transaction.get("merchant_category", "grocery")

    if amount > 1000:
        reasons.append(f"Unusually high amount (${amount:,.2f})")
    if amount < 2.0:
        reasons.append(f"Suspicious micro-transaction (${amount:.2f})")
    if velocity >= 5:
        reasons.append(f"High velocity ({velocity} txns/hr)")
    if distance > 200:
        reasons.append(f"Far from home ({distance:.0f} km)")
    if transaction.get("unusual_location", 0):
        reasons.append("Unusual geographic location")
    if CATEGORY_RISK.get(category, 0) > 0.35:
        reasons.append(f"High-risk category ({category})")
    if transaction.get("is_night", False):
        reasons.append("Unusual transaction hour")
    
    if not reasons:
        reasons = ["Transaction pattern within normal range"]

    return {
        "fraud_score": fraud_score,
        "is_fraud": is_fraud,
        "confidence": round(abs(fraud_score - 0.5) * 2, 4),
        "isolation_forest_score": round(iso_score, 4),
        "gradient_boosting_score": round(fraud_score, 4),
        "risk_level": "HIGH" if fraud_score > 0.7 else "MEDIUM" if fraud_score > 0.4 else "LOW",
        "reasons": reasons,
    }
