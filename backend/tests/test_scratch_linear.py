"""Tests for Scratch NumPy Linear Regression:
- Mathematical correctness of MSE cost function
- Analytical gradient calculation verified against numerical gradient
- Weight and bias updates via Gradient Descent
- Loss monotonically decreases on standard data
- Prediction accuracy on synthetic linear relationship
"""

import numpy as np
import pytest
from app.ml.scratch.linear_regression import ScratchLinearRegression


def test_linear_regression_cost_calculation():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert ScratchLinearRegression.compute_cost(y_true, y_pred) == 0.0

    y_pred_off = np.array([2.0, 3.0, 4.0])
    # MSE = ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = 3 / 3 = 1.0
    assert pytest.approx(ScratchLinearRegression.compute_cost(y_true, y_pred_off)) == 1.0


def test_linear_regression_numerical_gradient_check():
    """Verify analytical gradients match numerical gradients via central finite differences."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = np.random.randn(20)

    model = ScratchLinearRegression(learning_rate=0.01, epochs=10)
    model._init_params(3)
    y_pred = np.dot(X, model.weights) + model.bias

    analytical_dw, analytical_db = model.compute_gradients(X, y, y_pred)

    eps = 1e-6
    # Check dw numerical
    numerical_dw = np.zeros_like(model.weights)
    for j in range(len(model.weights)):
        w_plus = model.weights.copy()
        w_plus[j] += eps
        loss_plus = model.compute_cost(y, np.dot(X, w_plus) + model.bias)

        w_minus = model.weights.copy()
        w_minus[j] -= eps
        loss_minus = model.compute_cost(y, np.dot(X, w_minus) + model.bias)

        numerical_dw[j] = (loss_plus - loss_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytical_dw, numerical_dw, rtol=1e-4, atol=1e-4)

    # Check db numerical
    loss_b_plus = model.compute_cost(y, np.dot(X, model.weights) + (model.bias + eps))
    loss_b_minus = model.compute_cost(y, np.dot(X, model.weights) + (model.bias - eps))
    numerical_db = (loss_b_plus - loss_b_minus) / (2.0 * eps)

    assert pytest.approx(analytical_db, rel=1e-4) == numerical_db


def test_linear_regression_convergence_and_loss_decrease():
    """Verify that Gradient Descent decreases the training loss."""
    np.random.seed(42)
    # Synthetic target: y = 2.5 * x1 - 1.5 * x2 + 3.0
    X = np.random.randn(100, 2)
    y = 2.5 * X[:, 0] - 1.5 * X[:, 1] + 3.0 + np.random.normal(0, 0.05, size=100)

    model = ScratchLinearRegression(learning_rate=0.05, epochs=300, random_state=42)
    model.fit(X, y)

    # Loss must decrease
    assert len(model.loss_history) == 300
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.loss_history[-1] < 0.1  # Converged to low MSE

    # Weights should be close to true parameters
    assert pytest.approx(model.weights[0], abs=0.2) == 2.5
    assert pytest.approx(model.weights[1], abs=0.2) == -1.5
    assert pytest.approx(model.bias, abs=0.2) == 3.0

    # Predictions check
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
