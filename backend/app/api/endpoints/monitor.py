"""
NEXUS-AI — Model Monitoring & Drift Detection API
==================================================
GET /api/v1/monitor/drift    — PSI drift report across all 8 ML models
GET /api/v1/monitor/health   — Quick system health summary
"""
from fastapi import APIRouter
from app.monitoring.drift import get_drift_report

router = APIRouter(prefix="/api/v1/monitor", tags=["monitoring"])


@router.get("/drift")
def drift_report():
    """
    Return PSI-based drift report for all 8 NEXUS ML models.

    PSI thresholds:
      < 0.10  → stable
      0.10–0.25 → elevated (monitor)
      > 0.25  → drift_detected (consider retraining)
    """
    return get_drift_report()


@router.get("/health")
def model_health():
    """Quick health summary — overall status + per-model status string."""
    report = get_drift_report()
    return {
        "status":         report["overall_health"],
        "n_models":       report["n_models"],
        "n_stable":       report["n_stable"],
        "n_elevated":     report["n_elevated"],
        "n_drift":        report["n_drift_detected"],
        "generated_at":   report["generated_at"],
    }
