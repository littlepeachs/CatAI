from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
    if len(set(y_true.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def best_threshold_accuracy(y_true, y_prob) -> dict[str, float]:
    best = {"best_accuracy": -1.0, "best_threshold": 0.5}
    for threshold in np.linspace(0.0, 1.0, 101):
        acc = accuracy_score(np.asarray(y_true, dtype=int), np.asarray(y_prob) >= threshold)
        if acc > best["best_accuracy"]:
            best = {"best_accuracy": float(acc), "best_threshold": float(threshold)}
    return best
