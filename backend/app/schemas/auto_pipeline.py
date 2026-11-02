"""Pydantic Schemas for Automated Data Cleaning and Auto-Train Pipeline."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.schemas.training import TrainResponse


class CleanDatasetRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    handle_duplicates: bool = True
    impute_missing: bool = True
    cap_outliers: bool = True
    clean_strings: bool = True
    save_as_new_dataset: bool = True
    new_dataset_name: Optional[str] = None


class CleanDatasetResponse(BaseModel):
    original_dataset_id: str
    cleaned_dataset_id: Optional[str] = None
    cleaned_dataset_name: Optional[str] = None
    audit_report: Dict[str, Any]
    preview_rows: List[Dict[str, Any]]
    columns: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    recommended_target: Optional[str] = None
    recommended_task: str


class AutoTrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    auto_clean_first: bool = True
    method: str = Field(
        "auto_recommend",
        description="auto_recommend, scratch_linear, library_linear, scratch_logistic, library_logistic, scratch_mlp, library_mlp, compare_both",
    )
    implementation_type: str = Field("scratch", description="scratch or library")
    test_size: float = 0.2
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None


class AutoTrainResponse(BaseModel):
    dataset_id: str
    cleaning_report: Optional[Dict[str, Any]] = None
    primary_model: TrainResponse
    comparison_model: Optional[TrainResponse] = None
    metric_differences: Optional[Dict[str, float]] = None
    sample_prediction: Optional[Dict[str, Any]] = None
    summary_message: str
