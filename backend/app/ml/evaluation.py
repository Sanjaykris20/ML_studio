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


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract, normalize, and rank feature importance for supported models.

    Note:
        Feature importance is computed from the learned coefficients of the
        preprocessed features (after scaling/encoding). Higher absolute coefficient
        magnitude (|w|) indicates greater influence on model predictions.

    Args:
        model: Fitted model instance (Scratch or Scikit-learn wrapper).
        feature_names: List of preprocessed feature names corresponding to model inputs.
        algorithm: Optional algorithm identifier ("linear_regression", "logistic_regression", "mlp").
            If omitted, the function inspects the model class name and attributes.

    Returns:
        Dict[str, Any] with:
            - supported (bool): Whether the model architecture supports direct coefficient importance.
            - method (str): Description of the methodology used (e.g. "Normalized Absolute Coefficients (|w|)").
            - explanation (Optional[str]): Informative explanation, especially when unsupported.
            - rankings (List[Dict[str, Any]]): Ranked list of features, from highest to lowest influence.
                Each element contains:
                - feature (str): Name of the preprocessed feature.
                - importance (float): Normalized importance percentage (0.0 to 100.0).
                - rank (int): 1-based rank (1 is highest influence).
                - raw_coefficient (float): Raw learned weight/coefficient value.
            - scores (Dict[str, float]): Map of feature name to importance percentage.
            - raw_coefficients (Dict[str, float]): Map of feature name to raw learned weight.

    Raises:
        ValueError: If feature_names is empty, or coefficient dimensions do not match
            feature_names, or coefficients contain invalid numerical values (NaN, Inf).
    """
    algo_str = (algorithm or "").lower().strip()
    class_name = model.__class__.__name__.lower() if model is not None else ""

    # Keep MLP unsupported: multi-layer non-linear models do not expose direct linear coefficients
    if algo_str == "mlp" or "mlp" in class_name:
        return {
            "supported": False,
            "method": "Not applicable",
            "explanation": (
                "Multilayer Perceptron (MLP) consists of non-linear hidden layers "
                "without direct linear coefficients. Direct feature importance is "
                "supported for Linear Regression and Logistic Regression models."
            ),
            "rankings": [],
            "scores": {},
            "raw_coefficients": {},
        }

    # Validate feature names list
    if not feature_names:
        raise ValueError("feature_names list cannot be empty.")

    # Extract learned weights / coefficients
    raw_weights = None
    if hasattr(model, "weights") and model.weights is not None:
        # Scratch LinearRegression or LogisticRegression
        raw_weights = model.weights
    elif hasattr(model, "model") and hasattr(model.model, "coef_") and model.model.coef_ is not None:
        # SklearnLinearRegressionWrapper or SklearnLogisticRegressionWrapper
        raw_weights = model.model.coef_
    elif hasattr(model, "coef_") and model.coef_ is not None:
        # Direct Scikit-learn LinearRegression or LogisticRegression estimator
        raw_weights = model.coef_
    else:
        raise ValueError(
            "Model has no learned weights or coefficients. Ensure the model is fitted before extracting feature importance."
        )

    # Convert to 1D float64 numpy array
    weights_arr = np.asarray(raw_weights, dtype=np.float64)

    # Handle 2D coefficient shapes from Scikit-learn (e.g. (1, n_features) for binary logistic regression)
    if weights_arr.ndim == 2:
        if weights_arr.shape[0] == 1:
            weights_arr = weights_arr[0]
        elif weights_arr.shape[0] > 1:
            # Multi-class: average absolute coefficient across classes
            weights_arr = np.mean(weights_arr, axis=0)

    # Validate dimensions match feature_names
    if weights_arr.ndim != 1 or len(weights_arr) != len(feature_names):
        raise ValueError(
            f"Feature names count ({len(feature_names)}) does not match coefficient count ({len(weights_arr)})."
        )

    # Validate numerical integrity
    if np.isnan(weights_arr).any() or np.isinf(weights_arr).any():
        raise ValueError("Model coefficients contain NaN or Infinite values.")

    # Calculate absolute magnitude
    abs_weights = np.abs(weights_arr)
    total_abs = float(np.sum(abs_weights))

    # Safe handling of zero coefficients
    if total_abs == 0.0:
        importances = np.zeros(len(feature_names), dtype=np.float64)
    else:
        importances = (abs_weights / total_abs) * 100.0

    # Stable descending sort by absolute magnitude
    sort_order = np.argsort(-abs_weights, kind="mergesort")

    rankings = [
        {
            "feature": feature_names[idx],
            "importance": round(float(importances[idx]), 2),
            "rank": rank + 1,
            "raw_coefficient": round(float(weights_arr[idx]), 6),
        }
        for rank, idx in enumerate(sort_order)
    ]

    scores = {feature_names[i]: round(float(importances[i]), 2) for i in range(len(feature_names))}
    raw_coefficients = {feature_names[i]: round(float(weights_arr[i]), 6) for i in range(len(feature_names))}

    return {
        "supported": True,
        "method": "Normalized Absolute Coefficients (|w|)",
        "explanation": None,
        "rankings": rankings,
        "scores": scores,
        "raw_coefficients": raw_coefficients,
    }

