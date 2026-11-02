"""Pydantic Schemas for Dataset Exploration and Upload."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetOut(BaseModel):
    id: str
    name: str
    filename: str
    task_type: str
    total_rows: int
    total_columns: int
    feature_names: List[str]
    target_column: Optional[str] = None
    is_builtin: bool
    description: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetPreviewOut(BaseModel):
    id: str
    name: str
    total_rows: int
    total_columns: int
    columns: List[str]
    head_rows: List[Dict[str, Any]]
    numerical_features: List[str]
    categorical_features: List[str]
    missing_values: Dict[str, int]
    duplicate_rows: int
    summary_statistics: Dict[str, Dict[str, Any]]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
