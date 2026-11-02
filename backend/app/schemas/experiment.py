"""Pydantic Schemas for Experiment Tracking and Model Comparison."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ExperimentOut(BaseModel):
    id: str
    name: str
    dataset_id: str
    algorithm: str
    implementation_type: str
    task_type: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    loss_history: Optional[List[float]] = None
    val_loss_history: Optional[List[float]] = None
    feature_names: List[str]
    target_column: str
    training_time_ms: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ModelComparisonOut(BaseModel):
    task_type: str
    model_a: Dict[str, Any]  # Scratch model summary + metrics
    model_b: Dict[str, Any]  # Library model summary + metrics
    metric_differences: Dict[str, float]
    comparison_valid: bool
    notes: Optional[str] = None
