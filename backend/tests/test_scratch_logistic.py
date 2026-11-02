"""Tests for Scratch NumPy Logistic Regression:
- Sigmoid activation output bounded in [0, 1] and numerically stable
- Binary cross-entropy cost computation
- Analytical gradient check against central finite differences
- Convergence on separable binary classification problem
"""

import numpy as np
import pytest
from app.ml.scratch.logistic_regression import ScratchLogisticRegression


def test_sigmoid_numerical_stability():
    # Test extreme negative and positive values
    extreme_z = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    s = ScratchLogisticRegression.sigmoid(extreme_z)

    assert not np.any(np.isnan(s))
    assert not np.any(np.isinf(s))
    assert pytest.approx(s[0], abs=1e-5) == 0.0
    assert pytest.approx(s[2], abs=1e-5) == 0.5
    assert pytest.approx(s[4], abs=1e-5) == 1.0


def test_bce_cost_calculation():
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([0.9999, 0.0001])
    loss_low = ScratchLogisticRegression.compute_cost(y_true, y_prob)
    assert loss_low < 0.01

    y_prob_wrong = np.array([0.0001, 0.9999])
    loss_high = ScratchLogisticRegression.compute_cost(y_true, y_prob_wrong)
    assert loss_high > 5.0


def test_logistic_regression_numerical_gradient_check():
    """Verify analytical gradients against finite differences."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20).astype(float)

    model = ScratchLogisticRegression(learning_rate=0.01, epochs=10)
    model._init_params(3)
    z = np.dot(X, model.weights) + model.bias
    p = model.sigmoid(z)

    analytical_dw, analytical_db = model.compute_gradients(X, y, p)

    eps = 1e-6
    numerical_dw = np.zeros_like(model.weights)
    for j in range(len(model.weights)):
        w_plus = model.weights.copy()
        w_plus[j] += eps
        loss_plus = model.compute_cost(y, model.sigmoid(np.dot(X, w_plus) + model.bias))

        w_minus = model.weights.copy()
        w_minus[j] -= eps
        loss_minus = model.compute_cost(y, model.sigmoid(np.dot(X, w_minus) + model.bias))

        numerical_dw[j] = (loss_plus - loss_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytical_dw, numerical_dw, rtol=1e-4, atol=1e-4)

    # Bias gradient check
    loss_b_plus = model.compute_cost(y, model.sigmoid(np.dot(X, model.weights) + (model.bias + eps)))
    loss_b_minus = model.compute_cost(y, model.sigmoid(np.dot(X, model.weights) + (model.bias - eps)))
    numerical_db = (loss_b_plus - loss_b_minus) / (2.0 * eps)

    assert pytest.approx(analytical_db, rel=1e-4) == numerical_db


def test_logistic_regression_convergence():
    """Test convergence and classification on linearly separable data."""
    np.random.seed(42)
    # Class 0 centered at (-2, -2), Class 1 centered at (2, 2)
    X0 = np.random.randn(50, 2) - 2.0
    X1 = np.random.randn(50, 2) + 2.0
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    model = ScratchLogisticRegression(learning_rate=0.1, epochs=300, random_state=42)
    model.fit(X, y)

    # Loss must decrease
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.loss_history[-1] < 0.2

    # Accuracy must be >= 95%
    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    assert accuracy >= 0.95

    # Check probabilities shape and range
    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(100), atol=1e-5)
