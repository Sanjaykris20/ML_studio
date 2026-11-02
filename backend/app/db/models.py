"""SQLAlchemy ORM Models for ML Studio.

Entities:
- DatasetModel: Stores dataset metadata, schema, and storage location.
- ExperimentModel: Stores model training experiments, metrics, loss curves, and artifact references.
- PredictionHistoryModel: Stores historical inference records made in Prediction Studio.
"""

from datetime import datetime
import uuid
from typing import List, Optional, Any
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(128), nullable=False)
    filename = Column(String(256), nullable=False)
    filepath = Column(String(512), nullable=False)
    task_type = Column(String(64), nullable=False, default="regression")
    total_rows = Column(Integer, nullable=False, default=0)
    total_columns = Column(Integer, nullable=False, default=0)
    feature_names = Column(JSON, nullable=False, default=list)
    target_column = Column(String(128), nullable=True)
    is_builtin = Column(Boolean, nullable=False, default=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    experiments = relationship("ExperimentModel", back_populates="dataset", cascade="all, delete-orphan")


class ExperimentModel(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(128), nullable=False)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    algorithm = Column(String(64), nullable=False)  # "linear_regression", "logistic_regression", "mlp"
    implementation_type = Column(String(32), nullable=False)  # "scratch", "library"
    task_type = Column(String(64), nullable=False)  # "regression", "binary_classification", "multiclass_classification"
    hyperparameters = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict)
    loss_history = Column(JSON, nullable=True, default=list)
    val_loss_history = Column(JSON, nullable=True, default=list)
    feature_names = Column(JSON, nullable=False, default=list)
    target_column = Column(String(128), nullable=False)
    artifact_path = Column(String(512), nullable=False)
    training_time_ms = Column(Float, nullable=False, default=0.0)
    status = Column(String(32), nullable=False, default="completed")  # "completed", "training", "failed"
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    dataset = relationship("DatasetModel", back_populates="experiments")
    predictions = relationship("PredictionHistoryModel", back_populates="experiment", cascade="all, delete-orphan")


class PredictionHistoryModel(Base):
    __tablename__ = "prediction_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    input_values = Column(JSON, nullable=False, default=dict)
    predicted_value = Column(String(128), nullable=False)
    probabilities = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    experiment = relationship("ExperimentModel", back_populates="predictions")
