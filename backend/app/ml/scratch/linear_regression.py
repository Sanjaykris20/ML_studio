"""Scratch Linear Regression Implementation using NumPy.

Features:
- Analytical prediction function
- Weight and bias initialization (configurable random state)
- Mean Squared Error (MSE) cost function
- Analytical gradient calculation
- Batch Gradient Descent optimizer
- Training loss history tracking
- Evaluation metrics calculation
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class ScratchLinearRegression:
    """Linear Regression implemented from scratch using NumPy.
    
    Attributes:
        learning_rate (float): Step size for gradient descent updates.
        epochs (int): Number of complete passes over the training dataset.
        random_state (Optional[int]): Seed for reproducible weight initialization.
        weights (np.ndarray): Learned feature coefficients (shape: (n_features,)).
        bias (float): Learned intercept term.
        loss_history (List[float]): MSE loss value recorded at each epoch.
        training_time (float): Execution duration in seconds.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        random_state: Optional[int] = 42,
    ):
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"Epochs must be positive integer, got {epochs}")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.n_features_: Optional[int] = None

    def _init_params(self, n_features: int) -> None:
        """Initialize weights and bias."""
        self.n_features_ = n_features
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            # Initialize with small random normal values scaled by 1/sqrt(n_features)
            self.weights = rng.randn(n_features) * (1.0 / np.sqrt(max(1, n_features)))
            self.bias = float(rng.randn() * 0.01)
        else:
            self.weights = np.zeros(n_features, dtype=np.float64)
            self.bias = 0.0

    @staticmethod
    def compute_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error (MSE): (1 / m) * sum((y_pred - y_true)^2)."""
        m = len(y_true)
        if m == 0:
            return 0.0
        residuals = y_pred - y_true
        return float(np.mean(residuals ** 2))

    def compute_gradients(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute analytical gradients of MSE loss with respect to weights and bias.
        
        dJ/dw = (2 / m) * X.T @ (y_pred - y)
        dJ/db = (2 / m) * sum(y_pred - y)
        """
        m = X.shape[0]
        residuals = y_pred - y
        dw = (2.0 / m) * np.dot(X.T, residuals)
        db = (2.0 / m) * np.sum(residuals)
        return dw, float(db)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ScratchLinearRegression":
        """Train Linear Regression model using Batch Gradient Descent.
        
        Args:
            X: Training features array of shape (n_samples, n_features).
            y: Training target array of shape (n_samples,).
            X_val: Optional validation features array.
            y_val: Optional validation target array.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        n_samples, n_features = X.shape
        self._init_params(n_features)
        self.loss_history = []
        self.val_loss_history = []

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64).flatten()

        for epoch in range(self.epochs):
            # 1. Forward prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # 2. Compute cost and record history
            loss = self.compute_cost(y, y_pred)
            self.loss_history.append(float(loss))

            if has_val:
                val_pred = np.dot(X_val, self.weights) + self.bias
                val_loss = self.compute_cost(y_val, val_pred)
                self.val_loss_history.append(float(val_loss))

            # 3. Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)

            # Prevent gradient explosion
            grad_norm = np.linalg.norm(dw)
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                # Stop if numerically unstable
                break
            if grad_norm > 100.0:
                dw = dw * (100.0 / grad_norm)

            # 4. Update weights and bias via Gradient Descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values for given input features: y_pred = X @ w + b."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.weights) + self.bias

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters and learned parameters."""
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "random_state": self.random_state,
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": float(self.bias),
        }
