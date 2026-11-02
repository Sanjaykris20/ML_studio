"""Comprehensive Evaluation Metrics and Diagnostic Visual Data Generator.

Provides:
- Regression Metrics: R², MAE, MSE, RMSE, Actual vs. Predicted points, Residuals.
- Classification Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, ROC Curve points.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_plot_points: int = 150,
) -> Dict[str, Any]:
    """Calculate regression evaluation metrics and scatter plot coordinate data."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    residuals = (y_true - y_pred).tolist()

    # Subsample data points if too large for smooth JSON transmission to UI
    n_samples = len(y_true)
    if n_samples > max_plot_points:
        indices = np.linspace(0, n_samples - 1, max_plot_points, dtype=int)
    else:
        indices = np.arange(n_samples)

    actual_vs_predicted = [
        {
            "index": int(i),
            "actual": float(y_true[i]),
            "predicted": float(y_pred[i]),
            "residual": float(y_true[i] - y_pred[i]),
        }
        for i in indices
    ]

    return {
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "actual_vs_predicted": actual_vs_predicted,
        "residual_distribution": {
            "mean": round(float(np.mean(residuals)), 4),
            "std": round(float(np.std(residuals)), 4),
            "min": round(float(np.min(residuals)), 4),
            "max": round(float(np.max(residuals)), 4),
        },
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Calculate classification evaluation metrics, confusion matrix, and ROC curve."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if classes is None:
        classes = sorted(list(set(y_true).union(set(y_pred))))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_data = {
        "labels": [str(c) for c in classes],
        "matrix": cm.tolist(),
    }

    # ROC AUC & ROC Curve (when probability distribution is provided)
    roc_auc: Optional[float] = None
    roc_points: List[Dict[str, float]] = []

    if y_prob is not None:
        try:
            if len(classes) == 2:
                # Binary classification
                # Use prob of positive class
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
                roc_auc = float(roc_auc_score(y_true, prob_pos))
                fpr, tpr, thresholds = roc_curve(y_true, prob_pos)

                # Subsample ROC points if dense
                step = max(1, len(fpr) // 50)
                roc_points = [
                    {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
                    for i in range(0, len(fpr), step)
                ]
                # Ensure (1, 1) is present
                if roc_points and (roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0):
                    roc_points.append({"fpr": 1.0, "tpr": 1.0})
            elif len(classes) > 2 and y_prob.ndim == 2 and y_prob.shape[1] == len(classes):
                # Multiclass ROC AUC (One-vs-Rest)
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc = None
            roc_points = []

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm_data,
        "roc_curve": roc_points,
    }
