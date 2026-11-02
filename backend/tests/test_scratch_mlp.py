"""Tests for PyTorch Multilayer Perceptron:
- Regression support and loss reduction
- Binary classification support and probability prediction
- Multiclass classification support (e.g. 3 classes)
- Forward propagation shape and backpropagation
"""

import numpy as np
import pytest
from app.ml.scratch.mlp import ScratchPyTorchMLP


def test_mlp_regression():
    np.random.seed(42)
    X = np.random.randn(80, 4)
    y = 2.0 * X[:, 0] - X[:, 1] + 1.0

    mlp = ScratchPyTorchMLP(
        hidden_layers=[32, 16],
        activation="relu",
        optimizer="adam",
        learning_rate=0.02,
        epochs=80,
        batch_size=16,
        task="regression",
        random_state=42,
    )
    mlp.fit(X, y)

    assert len(mlp.loss_history) == 80
    assert mlp.loss_history[-1] < mlp.loss_history[0]

    preds = mlp.predict(X[:10])
    assert preds.shape == (10,)
    assert not np.any(np.isnan(preds))


def test_mlp_binary_classification():
    np.random.seed(42)
    X0 = np.random.randn(40, 3) - 1.5
    X1 = np.random.randn(40, 3) + 1.5
    X = np.vstack([X0, X1])
    y = np.array([0] * 40 + [1] * 40)

    mlp = ScratchPyTorchMLP(
        hidden_layers=[16],
        activation="relu",
        optimizer="adam",
        learning_rate=0.03,
        epochs=60,
        batch_size=16,
        task="binary_classification",
        random_state=42,
    )
    mlp.fit(X, y)

    preds = mlp.predict(X)
    probs = mlp.predict_proba(X)

    assert preds.shape == (80,)
    assert probs.shape == (80, 2)
    accuracy = np.mean(preds == y)
    assert accuracy >= 0.90


def test_mlp_multiclass_classification():
    np.random.seed(42)
    # 3 distinct clusters
    X0 = np.random.randn(30, 2) + np.array([-3.0, 0.0])
    X1 = np.random.randn(30, 2) + np.array([3.0, 0.0])
    X2 = np.random.randn(30, 2) + np.array([0.0, 3.0])
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)

    mlp = ScratchPyTorchMLP(
        hidden_layers=[32, 16],
        activation="relu",
        optimizer="adam",
        learning_rate=0.02,
        epochs=60,
        batch_size=16,
        task="multiclass_classification",
        random_state=42,
    )
    mlp.fit(X, y)

    preds = mlp.predict(X)
    probs = mlp.predict_proba(X)

    assert probs.shape == (90, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(90), atol=1e-4)
    accuracy = np.mean(preds == y)
    assert accuracy >= 0.90
