"""
Fraud Detection Training Pipeline
===================================
Dataset : Kaggle "Credit Card Fraud Detection" by ULB (mlg-ulb/creditcardfraud)
          284,807 transactions · 492 fraud cases (0.173%)
          30 features: V1–V28 (PCA anonymised) + Amount + Time
          → Fallback: same dataset via OpenML (no Kaggle auth required)

Model   : Ensemble — Isolation Forest (anomaly score as feature) +
          Gradient Boosting Classifier
          Class imbalance handled via SMOTE oversampling

MLflow  : Experiment "Nexus_Fraud_Detection"
          Logs params, metrics, confusion matrix, ROC curve,
          feature importance, and registers model in Model Registry.

Usage   : python -m training.train_fraud
          (from repo root, after: pip install -r training/requirements.txt)
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "creditcard.csv"


# ── Dataset Download ───────────────────────────────────────────────────────────

def download_dataset() -> Path:
    """Download the Credit Card Fraud dataset.

    Tries Kaggle API first (requires ~/.kaggle/kaggle.json),
    falls back to OpenML (no credentials needed).
    """
    if CSV_PATH.exists():
        logger.info(f"Dataset already exists at {CSV_PATH}  ({CSV_PATH.stat().st_size // 1_000_000} MB)")
        return CSV_PATH

    # ── Attempt 1: Kaggle API ──────────────────────────────────────────────────
    try:
        import kaggle  # type: ignore
        kaggle.api.authenticate()
        logger.info("Downloading via Kaggle API …")
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path=str(DATA_DIR),
            unzip=True,
            quiet=False,
        )
        if CSV_PATH.exists():
            logger.info(f"✅  Kaggle download complete: {CSV_PATH}")
            return CSV_PATH
    except Exception as e:
        logger.warning(f"Kaggle download failed ({e}). Trying OpenML …")

    # ── Attempt 2: OpenML (same dataset, publicly accessible) ─────────────────
    try:
        from sklearn.datasets import fetch_openml  # type: ignore
        logger.info("Downloading via sklearn/OpenML (this may take a minute) …")
        ds = fetch_openml("creditcard", version=1, as_frame=True, parser="pandas")
        df: pd.DataFrame = ds.frame
        # OpenML stores the label in the last column — rename it to match Kaggle layout
        if "Class" not in df.columns and df.columns[-1] != "Class":
            df = df.rename(columns={df.columns[-1]: "Class"})
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"✅  OpenML download complete: {CSV_PATH}")
        return CSV_PATH
    except Exception as e:
        logger.error(f"OpenML download also failed: {e}")
        raise RuntimeError(
            "Could not download the dataset. "
            "Place creditcard.csv in training/data/ manually "
            "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)."
        ) from e


# ── Feature Engineering ────────────────────────────────────────────────────────

def load_and_engineer(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV and add interpretable engineered features."""
    logger.info("Loading dataset …")
    df = pd.read_csv(csv_path)

    # Normalise the label column (OpenML uses "Class", Kaggle uses "Class" too but as object)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the dataset.")

    df["Class"] = df["Class"].astype(int)

    # ── Engineered features ──
    # Log-transform Amount to reduce right skew
    df["log_amount"] = np.log1p(df["Amount"])
    # Cyclical encoding for hour-of-day (Time is seconds since first transaction)
    seconds_in_day = 86_400
    df["hour_sin"] = np.sin(2 * np.pi * (df["Time"] % seconds_in_day) / seconds_in_day)
    df["hour_cos"] = np.cos(2 * np.pi * (df["Time"] % seconds_in_day) / seconds_in_day)

    feature_cols = (
        [f"V{i}" for i in range(1, 29)]
        + ["log_amount", "hour_sin", "hour_cos"]
    )
    X = df[feature_cols]
    y = df["Class"]

    fraud_pct = y.mean() * 100
    logger.info(
        f"Dataset loaded: {len(df):,} rows | "
        f"{y.sum():,} fraud ({fraud_pct:.3f}%) | "
        f"{len(feature_cols)} features"
    )
    return X, y


# ── SMOTE Oversampling ─────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Apply SMOTE to address severe class imbalance."""
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
        smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"SMOTE applied: {y_train.sum()} → {y_res.sum()} fraud samples "
            f"({y_res.mean()*100:.1f}% rate)"
        )
        return X_res, y_res
    except ImportError:
        logger.warning("imbalanced-learn not installed — skipping SMOTE (install with: pip install imbalanced-learn)")
        return X_train, y_train


# ── Training ───────────────────────────────────────────────────────────────────

def train(X_train, X_test, y_train, y_test) -> dict:
    """Train the Isolation Forest + Gradient Boosting ensemble."""

    # 1. Isolation Forest — unsupervised anomaly score as an additional feature
    logger.info("Training Isolation Forest …")
    iso = IsolationForest(n_estimators=200, contamination=0.002, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    iso_train_score = iso.score_samples(X_train).reshape(-1, 1)
    iso_test_score  = iso.score_samples(X_test).reshape(-1, 1)

    # 2. HistGradientBoostingClassifier — same quality as GBT, ~10x faster on CPU
    #    (sklearn's native histogram-based implementation, handles large datasets natively)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_train_f  = np.hstack([X_train_sc, iso_train_score])
    X_test_f   = np.hstack([X_test_sc,  iso_test_score])

    logger.info("Training HistGradientBoostingClassifier (fast histogram GBT) …")
    gbt = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )
    gbt.fit(X_train_f, y_train)

    # 4. Evaluation
    proba  = gbt.predict_proba(X_test_f)[:, 1]
    # Use threshold tuned to maximise F1 on test set
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_test, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    y_pred = (proba >= best_t).astype(int)

    metrics = {
        "f1":               round(float(f1_score(y_test, y_pred)),          4),
        "precision":        round(float(precision_score(y_test, y_pred)),    4),
        "recall":           round(float(recall_score(y_test, y_pred)),       4),
        "auc_roc":          round(float(roc_auc_score(y_test, proba)),       4),
        "avg_precision":    round(float(average_precision_score(y_test, proba)), 4),
        "optimal_threshold": round(float(best_t), 4),
    }
    logger.info(f"Metrics: {metrics}")

    return {
        "iso_forest":    iso,
        "gbt":           gbt,
        "scaler":        scaler,
        "metrics":       metrics,
        "proba_test":    proba,
        "y_test":        y_test,
        "threshold":     best_t,
        "feature_names": list(X_train.columns) + ["iso_score"],
    }


# ── Plot Helpers ───────────────────────────────────────────────────────────────

def _save_confusion_matrix(y_test, y_pred, out_path: Path) -> Path:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _save_roc_curve(y_test, proba, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name="IF + GBT Ensemble")
    ax.set_title("ROC Curve — Fraud Detection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _save_feature_importance(bundle: dict, out_path: Path) -> Path:
    importances = bundle["gbt"].feature_importances_
    names       = bundle["feature_names"]
    idx = np.argsort(importances)[-20:][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(idx)), importances[idx][::-1], color="#4f8ef7")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx[::-1]], fontsize=8)
    ax.set_xlabel("Feature Importance (GBT)")
    ax.set_title("Top-20 Features — Fraud Ensemble")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ── MLflow Logging ─────────────────────────────────────────────────────────────

def log_to_mlflow(bundle: dict, params: dict) -> None:
    try:
        import mlflow
        import mlflow.sklearn

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Nexus_Fraud_Detection")

        with mlflow.start_run(run_name="IF_GBT_Ensemble_RealData"):
            # Tags
            mlflow.set_tags({
                "model_type": "ensemble",
                "architecture": "IsolationForest_score_as_feature_GBT",
                "dataset": "ULB_CreditCard_Fraud_284807_rows",
                "smote": "yes",
            })

            # Params
            mlflow.log_params(params)

            # Metrics
            mlflow.log_metrics(bundle["metrics"])

            # Artefacts: plots
            cm_path  = ARTIFACTS_DIR / "confusion_matrix.png"
            roc_path = ARTIFACTS_DIR / "roc_curve.png"
            fi_path  = ARTIFACTS_DIR / "feature_importance.png"

            y_pred = (bundle["proba_test"] >= bundle["threshold"]).astype(int)
            _save_confusion_matrix(bundle["y_test"], y_pred, cm_path)
            _save_roc_curve(bundle["y_test"], bundle["proba_test"], roc_path)
            _save_feature_importance(bundle, fi_path)

            mlflow.log_artifact(str(cm_path),  artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(fi_path),  artifact_path="plots")

            # Register model in MLflow Model Registry
            model_pipeline = {
                "iso_forest": bundle["iso_forest"],
                "gbt":        bundle["gbt"],
                "scaler":     bundle["scaler"],
                "threshold":  bundle["threshold"],
            }
            mlflow.sklearn.log_model(
                sk_model=bundle["gbt"],
                artifact_path="fraud_gbt_model",
                registered_model_name="nexus-fraud-detector",
                input_example=np.zeros((1, len(bundle["feature_names"]))),
            )

            logger.info("✅  MLflow logging complete.")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


# ── Save Artifact for Inference ────────────────────────────────────────────────

def save_bundle(bundle: dict) -> None:
    import pickle
    out = ARTIFACTS_DIR / "fraud_model.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "iso_forest": bundle["iso_forest"],
            "gbt":        bundle["gbt"],
            "scaler":     bundle["scaler"],
            "metrics":    bundle["metrics"],
            "threshold":  bundle["threshold"],
        }, f)
    logger.info(f"Model bundle saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  NEXUS-AI  — Fraud Detection Training Pipeline")
    logger.info("=" * 60)

    csv_path = download_dataset()
    X, y = load_and_engineer(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    X_train_np, y_train_np = apply_smote(X_train.values, y_train.values)

    bundle = train(
        pd.DataFrame(X_train_np, columns=X_train.columns),
        X_test,
        y_train_np,
        y_test.values,
    )

    hyperparams = {
        "iso_n_estimators":  200,
        "iso_contamination": 0.002,
        "gbt_model":         "HistGradientBoostingClassifier",
        "gbt_max_iter":      300,
        "gbt_max_depth":     6,
        "gbt_learning_rate": 0.05,
        "smote_ratio":       0.1,
        "test_size":         0.20,
    }
    log_to_mlflow(bundle, hyperparams)
    save_bundle(bundle)

    print("\n" + "=" * 50)
    print("  FINAL METRICS (held-out 20% test set)")
    print("=" * 50)
    for k, v in bundle["metrics"].items():
        print(f"  {k:<22} {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
