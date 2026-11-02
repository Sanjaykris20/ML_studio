"""Pydantic Schemas for Prediction Studio."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class PredictRequest(BaseModel):
    experiment_id: Optional[str] = None
    inputs: Dict[str, Any]


class PredictResponse(BaseModel):
    experiment_id: str
    prediction: str
    raw_value: Any
    probabilities: Optional[Dict[str, float]] = None
    task_type: str
    timestamp: datetime


class PredictionHistoryOut(BaseModel):
    id: str
    experiment_id: str
    input_values: Dict[str, Any]
    predicted_value: str
    probabilities: Optional[Dict[str, float]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ModelFeatureInfo(BaseModel):
    experiment_id: str
    name: str
    algorithm: str
    implementation_type: str
    task_type: str
    feature_names: List[str]
    target_column: str
    feature_stats: Dict[str, Dict[str, Any]]
