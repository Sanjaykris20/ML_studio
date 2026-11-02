"""Scikit-Learn Library Model Wrappers.

Provides uniform interfaces matching the scratch implementations for fair benchmarking:
- SklearnLinearRegressionWrapper -> LinearRegression
- SklearnLogisticRegressionWrapper -> LogisticRegression
- SklearnMLPWrapper -> MLPRegressor / MLPClassifier
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier


class SklearnLinearRegressionWrapper:
    """Wrapper around scikit-learn LinearRegression."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.n_features_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SklearnLinearRegressionWrapper":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        self.n_features_ = X.shape[1]

        self.model.fit(X, y)

        # Scikit-learn LinearRegression uses closed-form OLS (no iterative loss curve),
        # but for visualization parity, record initial and final training & val MSE
        train_pred = self.model.predict(X)
        train_mse = float(np.mean((train_pred - y) ** 2))
        self.loss_history = [train_mse]

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = float(np.mean((val_pred - y_val) ** 2))
            self.val_loss_history = [val_mse]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "fit_intercept": self.fit_intercept,
            "weights": self.model.coef_.tolist() if hasattr(self.model, "coef_") else [],
            "bias": float(self.model.intercept_) if hasattr(self.model, "intercept_") else 0.0,
        }


class SklearnLogisticRegressionWrapper:
    """Wrapper around scikit-learn LogisticRegression."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        threshold: float = 0.5,
        random_state: Optional[int] = 42,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.random_state = random_state
        # Use saga or lbfgs solver for comparable gradient behavior
        self.model = LogisticRegression(
            max_iter=epochs,
            random_state=random_state,
            solver="lbfgs",
        )
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SklearnLogisticRegressionWrapper":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).flatten()
        self.n_features_ = X.shape[1]

        self.model.fit(X, y)
        self.classes_ = self.model.classes_

        # Record final train/val log loss for comparison
        from sklearn.metrics import log_loss
        train_probs = self.model.predict_proba(X)
        self.loss_history = [float(log_loss(y, train_probs))]

        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)
            self.val_loss_history = [float(log_loss(y_val, val_probs))]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def get_params(self) -> Dict[str, Any]:
        return {
            "max_iter": self.epochs,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "weights": self.model.coef_.tolist() if hasattr(self.model, "coef_") else [],
            "bias": self.model.intercept_.tolist() if hasattr(self.model, "intercept_") else [],
        }


class SklearnMLPWrapper:
    """Wrapper around scikit-learn MLPRegressor and MLPClassifier."""

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        task: str = "regression",
        random_state: Optional[int] = 42,
    ):
        self.hidden_layers = tuple(hidden_layers) if hidden_layers else (64, 32)
        # sklearn activations: identity, logistic, tanh, relu
        act_map = {"sigmoid": "logistic", "relu": "relu", "tanh": "tanh"}
        self.activation = act_map.get(activation.lower(), "relu")
        self.optimizer = "adam" if optimizer.lower() == "adam" else "sgd"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.task = task.lower()
        self.random_state = random_state

        if self.task == "regression":
            self.model: Union[MLPRegressor, MLPClassifier] = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                solver=self.optimizer,
                learning_rate_init=self.learning_rate,
                max_iter=self.epochs,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                solver=self.optimizer,
                learning_rate_init=self.learning_rate,
                max_iter=self.epochs,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )

        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SklearnMLPWrapper":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]

        self.model.fit(X, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_

        # Extract loss curve recorded by scikit-learn MLP
        if hasattr(self.model, "loss_curve_"):
            self.loss_history = [float(l) for l in self.model.loss_curve_]
        else:
            self.loss_history = [float(self.model.loss_)]

        if X_val is not None and y_val is not None:
            if self.task == "regression":
                val_pred = self.model.predict(X_val)
                self.val_loss_history = [float(np.mean((val_pred - y_val) ** 2))]
            else:
                from sklearn.metrics import log_loss
                val_probs = self.model.predict_proba(X_val)
                self.val_loss_history = [float(log_loss(y_val, val_probs))]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            raise ValueError("predict_proba is not available for regression.")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layers,
            "activation": self.activation,
            "solver": self.optimizer,
            "learning_rate": self.learning_rate,
            "max_iter": self.epochs,
            "batch_size": self.batch_size,
            "task": self.task,
        }
