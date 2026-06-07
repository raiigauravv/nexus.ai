"""
NEXUS-AI — ML Model Drift Detection
======================================
Tracks prediction-score drift across all 8 ML models using a
rolling window + Population Stability Index (PSI) approach.

PSI thresholds (industry standard):
  < 0.10  — stable (no significant drift)
  0.10–0.25 — moderate drift (monitor closely)
  > 0.25  — significant drift (consider retraining)

Usage:
    from app.monitoring.drift import record_prediction, get_drift_report
    record_prediction("fraud", 0.87)
    report = get_drift_report()
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Deque

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
WINDOW_SIZE = 500          # rolling window per model
PSI_BINS    = 10           # PSI bucket count
PSI_WARN    = 0.10         # yellow alert threshold
PSI_ALERT   = 0.25         # red alert threshold

# Baseline score distributions (from training/holdout evaluation)
_BASELINES: dict[str, dict] = {
    "fraud":          {"mean": 0.062, "std": 0.18,  "p50": 0.021, "p95": 0.71},
    "sentiment_pos":  {"mean": 0.74,  "std": 0.25,  "p50": 0.82,  "p95": 0.97},
    "sentiment_neg":  {"mean": 0.14,  "std": 0.19,  "p50": 0.07,  "p95": 0.61},
    "recommender":    {"mean": 0.68,  "std": 0.17,  "p50": 0.71,  "p95": 0.93},
    "visual_search":  {"mean": 0.61,  "std": 0.22,  "p50": 0.63,  "p95": 0.88},
    "rag_relevance":  {"mean": 0.82,  "std": 0.14,  "p50": 0.86,  "p95": 0.97},
    "topic_coherence":{"mean": 0.57,  "std": 0.21,  "p50": 0.59,  "p95": 0.87},
    "cross_module":   {"mean": 0.72,  "std": 0.16,  "p50": 0.74,  "p95": 0.95},
}


@dataclass
class ModelDriftState:
    model_name: str
    window:     Deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    n_total:    int = 0
    last_psi:   float = 0.0
    last_updated: str = ""

    def record(self, score: float) -> None:
        self.window.append(float(score))
        self.n_total += 1
        self.last_updated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def psi(self) -> float:
        """Compute PSI between current rolling window and baseline distribution."""
        if len(self.window) < 20:
            return 0.0

        baseline = _BASELINES.get(self.model_name, {"mean": 0.5, "std": 0.25})
        bl_mean, bl_std = baseline["mean"], max(baseline["std"], 1e-6)

        current = list(self.window)
        # Build uniform bins in [0, 1]
        edges = [i / PSI_BINS for i in range(PSI_BINS + 1)]
        edges[0]  = -1e-9
        edges[-1] = 1.0 + 1e-9

        def _hist(values: list[float]) -> list[float]:
            counts = [0] * PSI_BINS
            for v in values:
                for i in range(PSI_BINS):
                    if edges[i] < v <= edges[i + 1]:
                        counts[i] += 1
                        break
            total = max(sum(counts), 1)
            return [max(c / total, 1e-6) for c in counts]  # avoid log(0)

        # Generate synthetic baseline sample from N(mean, std) clipped to [0,1]
        import random
        rng = random.Random(42)
        baseline_sample = [
            min(1.0, max(0.0, rng.gauss(bl_mean, bl_std)))
            for _ in range(len(current))
        ]
        p_curr     = _hist(current)
        p_baseline = _hist(baseline_sample)

        psi_val = sum(
            (p_curr[i] - p_baseline[i]) * math.log(p_curr[i] / p_baseline[i])
            for i in range(PSI_BINS)
        )
        self.last_psi = round(psi_val, 4)
        return self.last_psi

    def status(self) -> str:
        psi = self.psi()
        if psi < PSI_WARN:
            return "stable"
        elif psi < PSI_ALERT:
            return "elevated"
        return "drift_detected"

    def summary(self) -> dict:
        w = list(self.window)
        if not w:
            return {
                "model": self.model_name,
                "status": "no_data",
                "n_total": 0,
                "psi": 0.0,
                "mean_score": None,
                "p95_score": None,
                "baseline_mean": _BASELINES.get(self.model_name, {}).get("mean"),
                "last_updated": self.last_updated,
            }
        n = len(w)
        sorted_w = sorted(w)
        return {
            "model":          self.model_name,
            "status":         self.status(),
            "n_window":       n,
            "n_total":        self.n_total,
            "psi":            self.psi(),
            "mean_score":     round(sum(w) / n, 4),
            "p50_score":      round(sorted_w[n // 2], 4),
            "p95_score":      round(sorted_w[int(n * 0.95)], 4),
            "baseline_mean":  _BASELINES.get(self.model_name, {}).get("mean"),
            "last_updated":   self.last_updated,
        }


# ── Global state (in-process, resets on restart) ──────────────────────────────
_STATES: dict[str, ModelDriftState] = {
    name: ModelDriftState(model_name=name)
    for name in _BASELINES
}


def record_prediction(model_name: str, score: float) -> None:
    """
    Record an inference score for drift tracking.
    Call this from each ML endpoint after running inference.

    Args:
        model_name:  One of the 8 NEXUS model keys (e.g. "fraud", "sentiment_pos")
        score:       Prediction probability / confidence score in [0, 1]
    """
    if model_name not in _STATES:
        _STATES[model_name] = ModelDriftState(model_name=model_name)
    _STATES[model_name].record(score)


def get_drift_report() -> dict:
    """
    Return a full drift report across all 8 ML models.
    Exposed at GET /api/v1/monitor/drift
    """
    models = [state.summary() for state in _STATES.values()]
    n_drifting = sum(1 for m in models if m["status"] == "drift_detected")
    n_elevated = sum(1 for m in models if m["status"] == "elevated")

    return {
        "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_models":        len(models),
        "n_stable":        len(models) - n_drifting - n_elevated,
        "n_elevated":      n_elevated,
        "n_drift_detected": n_drifting,
        "overall_health":  "healthy" if n_drifting == 0 and n_elevated <= 1 else
                           "degraded" if n_drifting == 0 else "critical",
        "psi_thresholds":  {"warn": PSI_WARN, "alert": PSI_ALERT},
        "models":          models,
    }
