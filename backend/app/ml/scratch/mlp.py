"""Multilayer Perceptron (MLP) Implementation using PyTorch.

Supports:
- Tasks: Regression, Binary Classification, Multiclass Classification
- Flexible architecture: custom hidden layers & neuron counts
- Activations: ReLU, Sigmoid, Tanh, GELU
- Optimizers: Adam, SGD
- Mini-batch gradient descent via PyTorch DataLoader
- Autograd backpropagation & parameter optimization
- Training and validation loss curves tracking
- Class probabilities extraction (Softmax / Sigmoid)
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



class PyTorchMLPModule(nn.Module):
    """Dynamically generated PyTorch Neural Network module."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "relu",
        task: str = "regression",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.task = task.lower()

        # Activation function mapping
        act_lower = activation.lower()
        if act_lower == "relu":
            act_cls = nn.ReLU
        elif act_lower == "sigmoid":
            act_cls = nn.Sigmoid
        elif act_lower == "tanh":
            act_cls = nn.Tanh
        elif act_lower == "gelu":
            act_cls = nn.GELU
        else:
            act_cls = nn.ReLU

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            prev_dim = hidden_dim

        # Final output layer (raw logits or linear output)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ScratchPyTorchMLP:
    """Multilayer Perceptron wrapper handling training, evaluation, and inference."""

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        task: str = "regression",  # "regression", "binary_classification", "multiclass_classification"
        random_state: Optional[int] = 42,
    ):
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32]
        self.activation = activation
        self.optimizer_name = optimizer.lower()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.task = task.lower()
        self.random_state = random_state

        self.model: Optional[PyTorchMLPModule] = None
        self.device = torch.device("cpu")  # CPU default for deterministic stability
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.output_dim_: int = 1

    def _determine_output_dim(self, y: np.ndarray) -> int:
        if self.task == "regression":
            self.classes_ = None
            return 1
        elif self.task == "binary_classification":
            self.classes_ = np.array([0, 1])
            return 2  # 2 output logits for clean cross entropy
        else:  # multiclass
            classes = np.unique(y)
            self.classes_ = classes
            return len(classes)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ScratchPyTorchMLP":
        """Train the PyTorch MLP on the provided dataset."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.output_dim_ = self._determine_output_dim(y)

        # Build PyTorch Network
        self.model = PyTorchMLPModule(
            input_dim=n_features,
            hidden_layers=self.hidden_layers,
            output_dim=self.output_dim_,
            activation=self.activation,
            task=self.task,
        ).to(self.device)

        # Loss function setup
        if self.task == "regression":
            criterion = nn.MSELoss()
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        else:
            criterion = nn.CrossEntropyLoss()
            y_tensor = torch.tensor(y, dtype=torch.long).view(-1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, n_samples),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )


        # Optional validation tensors
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(np.asarray(X_val, dtype=np.float32), device=self.device)
            if self.task == "regression":
                y_val_t = torch.tensor(np.asarray(y_val, dtype=np.float32), device=self.device).view(-1, 1)
            else:
                y_val_t = torch.tensor(np.asarray(y_val, dtype=np.int64), device=self.device).view(-1)

        # Optimizer selection
        if self.optimizer_name == "sgd":
            opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_history = []
        self.val_loss_history = []

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            total_batches = 0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                opt.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                opt.step()

                running_loss += loss.item()
                total_batches += 1

            epoch_train_loss = running_loss / max(1, total_batches)
            self.loss_history.append(float(epoch_train_loss))

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
                    self.val_loss_history.append(float(val_loss))
                self.model.train()

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions for regression or discrete class labels for classification."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        self.model.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            if self.task == "regression":
                return outputs.view(-1).cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probabilities for classification tasks."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        if self.task == "regression":
            raise ValueError("predict_proba is only available for classification tasks.")
        self.model.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs

    def get_params(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "task": self.task,
        }
