"""Scratch Logistic Regression Implementation using NumPy.

Features:
- Numerically stable Sigmoid activation function
- Linear combination (z = Xw + b)
- Binary Cross-Entropy (BCE) cost function
- Analytical gradient computation
- Gradient Descent optimization
- Probability prediction and thresholded binary class prediction
- Configurable classification threshold
- Training and validation loss history tracking
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class ScratchLogisticRegression:
    """Logistic Regression for binary classification implemented from scratch using NumPy.
    
    Attributes:
        learning_rate (float): Step size for gradient descent updates.
        epochs (int): Number of training iterations.
        threshold (float): Decision threshold for classifying probability to class 1.
        random_state (Optional[int]): Random seed for reproducible initialization.
        weights (np.ndarray): Learned feature weights.
        bias (float): Learned bias / intercept.
        loss_history (List[float]): BCE loss at each epoch.
        val_loss_history (List[float]): Validation BCE loss at each epoch.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        threshold: float = 0.5,
        random_state: Optional[int] = 42,
    ):
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        if epochs <= 0:
            raise ValueError(f"Epochs must be positive integer, got {epochs}")
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"Threshold must be strictly between 0 and 1, got {threshold}")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.threshold = float(threshold)
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable Sigmoid function: sigma(z) = 1 / (1 + exp(-z)).
        
        Clips z into [-250, 250] to prevent overflow in exp(-z).
        """
        z_clipped = np.clip(z, -250.0, 250.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def compute_cost(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
        """Compute Binary Cross-Entropy Loss:
        
        J = -(1/m) * sum[ y * ln(p + eps) + (1 - y) * ln(1 - p + eps) ]
        """
        m = len(y_true)
        if m == 0:
            return 0.0
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
        return float(loss)

    def compute_gradients(
        self, X: np.ndarray, y: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute analytical gradients of BCE loss with respect to weights and bias.
        
        dJ/dw = (1 / m) * X.T @ (y_prob - y)
        dJ/db = (1 / m) * sum(y_prob - y)
        """
        m = X.shape[0]
        residuals = y_prob - y
        dw = (1.0 / m) * np.dot(X.T, residuals)
        db = (1.0 / m) * np.sum(residuals)
        return dw, float(db)

    def _init_params(self, n_features: int) -> None:
        """Initialize weights and bias."""
        self.n_features_ = n_features
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            self.weights = rng.randn(n_features) * (1.0 / np.sqrt(max(1, n_features)))
            self.bias = 0.0
        else:
            self.weights = np.zeros(n_features, dtype=np.float64)
            self.bias = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ScratchLogisticRegression":
        """Fit Logistic Regression model using Gradient Descent.
        
        Args:
            X: Training features array of shape (n_samples, n_features).
            y: Binary target array of shape (n_samples,) with values in {0, 1}.
            X_val: Optional validation features.
            y_val: Optional validation target.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        unique_classes = np.unique(y)
        if len(unique_classes) > 2:
            raise ValueError(
                f"Binary Logistic Regression expects 2 distinct classes, found {len(unique_classes)}: {unique_classes}"
            )
        self.classes_ = np.array([0, 1])

        n_samples, n_features = X.shape
        self._init_params(n_features)
        self.loss_history = []
        self.val_loss_history = []

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64).flatten()

        for epoch in range(self.epochs):
            # 1. Linear combination: z = Xw + b
            z = np.dot(X, self.weights) + self.bias

            # 2. Sigmoid activation: p = sigma(z)
            y_prob = self.sigmoid(z)

            # 3. Compute loss
            loss = self.compute_cost(y, y_prob)
            self.loss_history.append(float(loss))

            if has_val:
                val_z = np.dot(X_val, self.weights) + self.bias
                val_prob = self.sigmoid(val_z)
                val_loss = self.compute_cost(y_val, val_prob)
                self.val_loss_history.append(float(val_loss))

            # 4. Compute gradients
            dw, db = self.compute_gradients(X, y, y_prob)

            # Gradient clipping to maintain stability
            grad_norm = np.linalg.norm(dw)
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                break
            if grad_norm > 50.0:
                dw = dw * (50.0 / grad_norm)

            # 5. Gradient descent update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: returns array of shape (n_samples, 2),
        where col 0 is P(Y=0) and col 1 is P(Y=1).
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        z = np.dot(X, self.weights) + self.bias
        p1 = self.sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 or 1) using the configured classification threshold."""
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= self.threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters and learned weights."""
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": float(self.bias),
        }
