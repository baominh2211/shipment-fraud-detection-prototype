from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score



def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_percent: float) -> float:
    top_n = max(1, int(np.ceil(len(scores) * k_percent / 100.0)))
    top_indices = np.argsort(scores)[::-1][:top_n]
    return float(np.mean(y_true[top_indices]))



def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k_percent: float) -> float:
    positives = float(np.sum(y_true))
    if positives == 0:
        return 0.0
    top_n = max(1, int(np.ceil(len(scores) * k_percent / 100.0)))
    top_indices = np.argsort(scores)[::-1][:top_n]
    return float(np.sum(y_true[top_indices]) / positives)



def revenue_at_k(values: np.ndarray, scores: np.ndarray, k_percent: float) -> float:
    top_n = max(1, int(np.ceil(len(scores) * k_percent / 100.0)))
    top_indices = np.argsort(scores)[::-1][:top_n]
    return float(np.sum(values[top_indices]))



def ranking_metrics(y_true: Iterable[int], scores: Iterable[float]) -> dict[str, float | None]:
    """Compute ranking-based metrics at multiple review budgets.

    These top-k metrics are the primary evaluation for the triage paradigm:
    an analyst only reviews the top-k highest-risk records.
    """
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    metrics: dict[str, float | None] = {
        "average_precision": float(average_precision_score(y, s)),
        # Precision at multiple review budgets (what fraction of reviewed records are fraudulent?)
        "precision_at_1pct": precision_at_k(y, s, 1.0),
        "precision_at_3pct": precision_at_k(y, s, 3.0),
        "precision_at_5pct": precision_at_k(y, s, 5.0),
        "precision_at_10pct": precision_at_k(y, s, 10.0),
        # Recall at multiple review budgets (what fraction of all fraud is caught?)
        "recall_at_5pct": recall_at_k(y, s, 5.0),
        "recall_at_10pct": recall_at_k(y, s, 10.0),
        "recall_at_20pct": recall_at_k(y, s, 20.0),
    }
    metrics["roc_auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None
    return metrics



def threshold_metrics(y_true: Iterable[int], flagged: Iterable[int | bool]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(flagged), dtype=int)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "flag_rate": float(np.mean(p)),
    }



def explanation_coverage(scored: pd.DataFrame, flag_col: str = "flagged") -> dict[str, float]:
    flagged = scored[scored[flag_col]].copy()
    if flagged.empty:
        return {"flagged_records": 0, "coverage_pct": 100.0}
    has_expl = flagged["explanation"].fillna("").str.len().gt(10).mean() * 100.0
    return {"flagged_records": int(len(flagged)), "coverage_pct": float(has_expl)}



def synthetic_breakdown(scored: pd.DataFrame) -> dict[str, dict[str, float]]:
    if "synthetic_anomaly_type" not in scored.columns:
        return {}
    result: dict[str, dict[str, float]] = {}
    anomaly_rows = scored[scored["synthetic_is_anomaly"] == 1]
    for anomaly_type, grp in anomaly_rows.groupby("synthetic_anomaly_type"):
        result[str(anomaly_type)] = {
            "count": int(len(grp)),
            "recall": float(grp["flagged"].mean()) if len(grp) else 0.0,
            "avg_risk_score": float(grp["risk_score"].mean()) if len(grp) else 0.0,
            "avg_anomaly_score": float(grp["anomaly_score"].mean()) if len(grp) else 0.0,
        }
    return result



def find_best_f1_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    n_candidates: int = 500,
    max_flag_rate: float = 0.30,
) -> tuple[float, float]:
    """Find the threshold that maximises F1 on a validation set.

    To avoid the degenerate solution of flagging everything (which maximises
    recall and can give a misleadingly high F1 on imbalanced data), we enforce
    a maximum flag rate constraint.

    Returns (best_threshold, best_f1).
    """
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    # Search from high thresholds (few flags) to low (many flags)
    candidates = np.linspace(float(np.max(s)), float(np.min(s)), n_candidates)
    best_f1 = -1.0
    best_thr = float(np.quantile(s, 0.95))
    for thr in candidates:
        preds = (s >= thr).astype(int)
        flag_rate = float(np.mean(preds))
        if flag_rate > max_flag_rate:
            continue
        current_f1 = float(f1_score(y, preds, zero_division=0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thr = float(thr)
    return best_thr, best_f1


def tier_diagnostic_metrics(scored: pd.DataFrame, label_col: str) -> dict[str, dict[str, float | None]]:
    """Compute ranking metrics for each individual scoring tier.

    Helps diagnose which tier contributes most to detection.
    """
    tier_cols = {
        "rule_score": "tier1_rule",
        "peer_score": "tier2_peer",
        "isolation_score": "tier3_if",
        "supervised_score": "tier4_supervised",
    }
    diagnostics: dict[str, dict[str, float | None]] = {}
    y = scored[label_col].to_numpy(dtype=int)
    for col, name in tier_cols.items():
        if col not in scored.columns:
            continue
        s = scored[col].to_numpy(dtype=float)
        diagnostics[name] = ranking_metrics(y, s)
    return diagnostics


def _score_block(scored: pd.DataFrame, label_col: str, score_col: str, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_ranking": ranking_metrics(scored[label_col], scored[score_col]),
        f"{prefix}_threshold": threshold_metrics(scored[label_col], scored["flagged"]),
    }



def summarize_metrics(
    valid_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    synthetic_scored: pd.DataFrame,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    metrics.update(_score_block(valid_scored, "fraud", "risk_score", "validation_risk"))
    metrics.update(_score_block(valid_scored, "fraud", "anomaly_score", "validation_anomaly"))
    metrics.update(_score_block(test_scored, "fraud", "risk_score", "test_risk"))
    metrics.update(_score_block(test_scored, "fraud", "anomaly_score", "test_anomaly"))
    metrics.update(_score_block(synthetic_scored, "synthetic_is_anomaly", "risk_score", "synthetic_risk"))
    metrics.update(_score_block(synthetic_scored, "synthetic_is_anomaly", "anomaly_score", "synthetic_anomaly"))

    if "revenue_at_risk" in synthetic_scored.columns:
        metrics["synthetic_revenue_at_5pct"] = revenue_at_k(
            synthetic_scored["revenue_at_risk"].fillna(0.0).to_numpy(dtype=float),
            synthetic_scored["risk_score"].to_numpy(dtype=float),
            5.0,
        )
    metrics["synthetic_breakdown"] = synthetic_breakdown(synthetic_scored)
    metrics["explanation_quality"] = {
        "validation": explanation_coverage(valid_scored),
        "test": explanation_coverage(test_scored),
        "synthetic": explanation_coverage(synthetic_scored),
    }

    # Per-tier diagnostics
    metrics["tier_diagnostics"] = {
        "validation": tier_diagnostic_metrics(valid_scored, "fraud"),
        "test": tier_diagnostic_metrics(test_scored, "fraud"),
        "synthetic": tier_diagnostic_metrics(synthetic_scored, "synthetic_is_anomaly"),
    }
    return metrics
