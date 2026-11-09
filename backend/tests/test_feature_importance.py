"""Unit Tests for Feature Importance Extraction (Module 1).

Covers:
- Scratch and Scikit-learn Linear Regression
- Scratch and Scikit-learn Logistic Regression
- Normalized absolute coefficient percentage calculations
- Correct feature rankings and raw coefficient retention
- Safe handling of all-zero coefficients
- Validation: dimension mismatch, empty feature list, NaN/Inf coefficients
- Unsupported MLP handling (Scratch PyTorch MLP, Sklearn MLP, algorithm="mlp")
"""

import numpy as np
import pytest

# On Windows, PyTorch must be imported before Scikit-learn/MKL to avoid OpenMP DLL conflict
try:
    import torch
except Exception:
    pass

from sklearn.linear_model import LinearRegression, LogisticRegression

from app.ml.evaluation import extract_feature_importance
from app.ml.scratch.linear_regression import ScratchLinearRegression
from app.ml.scratch.logistic_regression import ScratchLogisticRegression
from app.ml.scratch.mlp import ScratchPyTorchMLP
from app.ml.library.sklearn_models import (
    SklearnLinearRegressionWrapper,
    SklearnLogisticRegressionWrapper,
    SklearnMLPWrapper,
)


def test_scratch_linear_regression_known_rankings():
    """Verify that Scratch Linear Regression ranks the most influential feature #1."""
    np.random.seed(42)
    # Target: y = 10.0 * x1 - 3.0 * x2 + 0.5 * x3 + 1.0
    # Expected ranking by magnitude: x1 (10.0), x2 (|-3.0|=3.0), x3 (0.5)
    X = np.random.randn(200, 3)
    y = 10.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 * X[:, 2] + 1.0 + np.random.normal(0, 0.05, size=200)
    feature_names = ["feat_primary", "feat_secondary", "feat_minor"]

    model = ScratchLinearRegression(learning_rate=0.05, epochs=400, random_state=42)
    model.fit(X, y)

    res = extract_feature_importance(model, feature_names, algorithm="linear_regression")

    assert res["supported"] is True
    assert res["method"] == "Normalized Absolute Coefficients (|w|)"
    assert res["explanation"] is None
    assert len(res["rankings"]) == 3

    # Ranking assertions
    rankings = res["rankings"]
    assert rankings[0]["feature"] == "feat_primary"
    assert rankings[0]["rank"] == 1
    assert rankings[1]["feature"] == "feat_secondary"
    assert rankings[1]["rank"] == 2
    assert rankings[2]["feature"] == "feat_minor"
    assert rankings[2]["rank"] == 3

    # Importance percentages should be proportional and sum to ~100%
    assert rankings[0]["importance"] > rankings[1]["importance"] > rankings[2]["importance"]
    total_importance = sum(r["importance"] for r in rankings)
    assert pytest.approx(total_importance, abs=0.5) == 100.0

    # Scores dict check
    assert res["scores"]["feat_primary"] == rankings[0]["importance"]
    assert res["raw_coefficients"]["feat_primary"] == rankings[0]["raw_coefficient"]


def test_sklearn_linear_regression_wrapper():
    """Verify Scikit-learn Linear Regression wrapper importance extraction."""
    np.random.seed(42)
    X = np.random.randn(150, 4)
    # y = -8.0 * x0 + 4.0 * x1 + 2.0 * x2 - 0.1 * x3
    y = -8.0 * X[:, 0] + 4.0 * X[:, 1] + 2.0 * X[:, 2] - 0.1 * X[:, 3] + 2.0
    feature_names = ["alpha", "beta", "gamma", "delta"]

    model = SklearnLinearRegressionWrapper(fit_intercept=True)
    model.fit(X, y)

    res = extract_feature_importance(model, feature_names)

    assert res["supported"] is True
    rankings = res["rankings"]
    assert len(rankings) == 4
    # Expected order: alpha (|-8|=8), beta (4), gamma (2), delta (0.1)
    assert [r["feature"] for r in rankings] == ["alpha", "beta", "gamma", "delta"]
    assert [r["rank"] for r in rankings] == [1, 2, 3, 4]
    assert rankings[0]["raw_coefficient"] < 0  # alpha has negative raw coefficient
    assert rankings[0]["importance"] > 0       # but positive importance percentage


def test_scratch_logistic_regression():
    """Verify Scratch Logistic Regression feature importance extraction."""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    # Probability strongly driven by x0
    z = 5.0 * X[:, 0] + 0.2 * X[:, 1] - 0.5
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob >= 0.5).astype(int)
    feature_names = ["strong_signal", "weak_signal"]

    model = ScratchLogisticRegression(learning_rate=0.05, epochs=300, random_state=42)
    model.fit(X, y)

    res = extract_feature_importance(model, feature_names, algorithm="logistic_regression")

    assert res["supported"] is True
    rankings = res["rankings"]
    assert rankings[0]["feature"] == "strong_signal"
    assert rankings[0]["rank"] == 1
    assert rankings[1]["feature"] == "weak_signal"
    assert rankings[1]["rank"] == 2
    assert rankings[0]["importance"] > 80.0


def test_sklearn_logistic_regression_wrapper_2d_coef():
    """Verify Scikit-learn Logistic Regression wrapper with 2D coef_ shape (1, n_features)."""
    np.random.seed(42)
    X = np.random.randn(150, 3)
    z = -4.0 * X[:, 0] + 2.5 * X[:, 1] - 0.5 * X[:, 2]
    y = (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)
    feature_names = ["f0", "f1", "f2"]

    model = SklearnLogisticRegressionWrapper(epochs=300, random_state=42)
    model.fit(X, y)

    assert model.model.coef_.ndim == 2  # shape (1, 3)

    res = extract_feature_importance(model, feature_names)

    assert res["supported"] is True
    rankings = res["rankings"]
    assert [r["feature"] for r in rankings] == ["f0", "f1", "f2"]
    assert rankings[0]["rank"] == 1


def test_zero_coefficients_safe_handling():
    """Verify that models with all-zero weights do not cause ZeroDivisionError and return 0%."""
    class DummyZeroModel:
        weights = np.array([0.0, 0.0, 0.0])

    feature_names = ["c1", "c2", "c3"]
    res = extract_feature_importance(DummyZeroModel(), feature_names)

    assert res["supported"] is True
    assert all(r["importance"] == 0.0 for r in res["rankings"])
    assert all(r["raw_coefficient"] == 0.0 for r in res["rankings"])
    assert list(res["scores"].values()) == [0.0, 0.0, 0.0]


def test_dimension_mismatch_raises_error():
    """Verify that dimension mismatch between feature_names and weights raises ValueError."""
    class DummyModel:
        weights = np.array([1.5, -2.0])  # 2 weights

    with pytest.raises(ValueError, match="Feature names count \\(3\\) does not match coefficient count \\(2\\)"):
        extract_feature_importance(DummyModel(), ["f1", "f2", "f3"])


def test_empty_feature_names_raises_error():
    """Verify that empty feature_names raises ValueError."""
    class DummyModel:
        weights = np.array([1.0])

    with pytest.raises(ValueError, match="feature_names list cannot be empty"):
        extract_feature_importance(DummyModel(), [])


def test_invalid_nan_inf_coefficients_raises_error():
    """Verify that NaN or Inf coefficients raise ValueError."""
    class NanModel:
        weights = np.array([1.0, np.nan])

    with pytest.raises(ValueError, match="Model coefficients contain NaN or Infinite values"):
        extract_feature_importance(NanModel(), ["f1", "f2"])

    class InfModel:
        weights = np.array([1.0, np.inf])

    with pytest.raises(ValueError, match="Model coefficients contain NaN or Infinite values"):
        extract_feature_importance(InfModel(), ["f1", "f2"])


def test_unfitted_model_raises_error():
    """Verify that an unfitted model without weights raises ValueError."""
    model = ScratchLinearRegression()  # Unfitted, weights is None

    with pytest.raises(ValueError, match="Model has no learned weights or coefficients"):
        extract_feature_importance(model, ["f1", "f2"])


def test_unsupported_mlp_models():
    """Verify that Scratch MLP, Sklearn MLP, and algorithm='mlp' return supported: False with clear explanation."""
    feature_names = ["feat1", "feat2", "feat3"]

    # 1. Scratch PyTorch MLP
    scratch_mlp = ScratchPyTorchMLP(epochs=2)
    res_scratch = extract_feature_importance(scratch_mlp, feature_names)
    assert res_scratch["supported"] is False
    assert res_scratch["method"] == "Not applicable"
    assert "Multilayer Perceptron (MLP) consists of non-linear hidden layers" in res_scratch["explanation"]
    assert res_scratch["rankings"] == []
    assert res_scratch["scores"] == {}

    # 2. Sklearn MLP Wrapper
    sklearn_mlp = SklearnMLPWrapper(epochs=2)
    res_sklearn = extract_feature_importance(sklearn_mlp, feature_names)
    assert res_sklearn["supported"] is False
    assert res_sklearn["method"] == "Not applicable"
    assert res_sklearn["rankings"] == []

    # 3. Explicit algorithm="mlp" flag
    class AnyModel:
        weights = np.array([1.0, 2.0, 3.0])

    res_algo = extract_feature_importance(AnyModel(), feature_names, algorithm="mlp")
    assert res_algo["supported"] is False
    assert res_algo["method"] == "Not applicable"
    assert res_algo["rankings"] == []
