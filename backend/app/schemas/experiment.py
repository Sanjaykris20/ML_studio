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


class FeatureRankingItem(BaseModel):
    feature: str
    importance: float
    rank: int
    raw_coefficient: Optional[float] = None


class FeatureImportanceInfo(BaseModel):
    supported: bool
    method: str
    explanation: Optional[str] = None
    rankings: List[FeatureRankingItem] = []
    scores: Dict[str, float] = {}
    raw_coefficients: Dict[str, float] = {}


class FeatureImportanceComparisonOut(BaseModel):
    model_a: FeatureImportanceInfo
    model_b: FeatureImportanceInfo
    differences: Dict[str, float] = {}
    comparison_method: str = "Normalized Absolute Coefficients (|w|)"
    comparison_valid: bool = True
    explanation: Optional[str] = None


class ModelComparisonOut(BaseModel):
    task_type: str
    model_a: Dict[str, Any]  # Scratch model summary + metrics
    model_b: Dict[str, Any]  # Library model summary + metrics
    metric_differences: Dict[str, float]
    comparison_valid: bool
    notes: Optional[str] = None
    feature_importance_comparison: Optional[FeatureImportanceComparisonOut] = None
