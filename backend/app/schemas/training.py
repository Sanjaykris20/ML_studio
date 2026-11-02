"""Pydantic Schemas for Model Training."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: str
    experiment_name: Optional[str] = None
    algorithm: str = Field(..., description="linear_regression, logistic_regression, mlp")
    implementation_type: str = Field("scratch", description="scratch or library")
    input_features: List[str]
    target_column: str
    task_type: Optional[str] = None  # Auto-detected if None
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    random_state: int = Field(42)

    # Preprocessing Config
    scaling: str = Field("standard", description="standard, minmax, none")
    imputation: str = Field("mean", description="mean, median, most_frequent")
    encoding: str = Field("onehot", description="onehot, ordinal")

    # Hyperparameters
    learning_rate: float = Field(0.01, gt=0.0)
    epochs: int = Field(500, ge=1, le=10000)
    threshold: float = Field(0.5, gt=0.0, lt=1.0)
    hidden_layers: List[int] = Field(default_factory=lambda: [64, 32])
    activation: str = Field("relu", description="relu, sigmoid, tanh")
    optimizer: str = Field("adam", description="adam, sgd")
    batch_size: int = Field(32, ge=1)


class TrainResponse(BaseModel):
    experiment_id: str
    name: str
    status: str
    algorithm: str
    implementation_type: str
    task_type: str
    metrics: Dict[str, Any]
    loss_history: List[float]
    val_loss_history: List[float]
    training_time_ms: float
    feature_names: List[str]
    target_column: str
    created_at: datetime
