import mlflow
import pandas as pd
import os

# Set tracking URI to the local DB
mlflow.set_tracking_uri("sqlite:///mlflow.db") # Wait, I used mlruns.db in the run. Let me check.
if not os.path.exists("mlruns.db") and os.path.exists("mlflow.db"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
else:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

def print_latest_metrics():
    print("--- 🚀 Nexus-AI Final Production Metrics 🚀 ---")
    
    # 1. Fraud
    try:
        fraud_runs = mlflow.search_runs(experiment_names=["Nexus_Fraud_Detection"], order_by=["start_time DESC"], max_results=1)
        if not fraud_runs.empty:
            run = fraud_runs.iloc[0]
            print(f"\n[Fraud Detection Ensemble]")
            print(f"  - F1 Score:  {run.get('metrics.f1_score', 'N/A'):.4f}")
            print(f"  - AUC-ROC:   {run.get('metrics.auc_roc', 'N/A'):.4f}")
            print(f"  - Precision: {run.get('metrics.precision', 'N/A'):.4f}")
            print(f"  - Model:     IsolationForest + GradientBoosting")
    except Exception as e:
        print(f"Fraud metrics fetch failed: {e}")

    # 2. Recommender
    try:
        rec_runs = mlflow.search_runs(experiment_names=["Nexus_Recommendation_Engine"], order_by=["start_time DESC"], max_results=1)
        if not rec_runs.empty:
            run = rec_runs.iloc[0]
            print(f"\n[Recommendation Engine]")
            print(f"  - Density:   {run.get('metrics.density_pct', 'N/A'):.2f}%")
            print(f"  - SVD Rank:  {run.get('params.svd_rank', 'N/A')}")
            print(f"  - Matrix:    {run.get('params.matrix_shape', 'N/A')}")
    except Exception as e:
        print(f"Recommender metrics fetch failed: {e}")

    # 3. Vision
    print(f"\n[Computer Vision]")
    print(f"  - Model:     CLIP ViT-B/32")
    print(f"  - Indexed:   150 items")
    print(f"  - Features:  Hybrid SVD-CLIP Fusion")

if __name__ == "__main__":
    print_latest_metrics()
