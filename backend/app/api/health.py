"""Health, Diagnostics, and System Overview API Endpoints."""

import os
import sys
from typing import Dict, Any
import torch
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sqlalchemy import text
from app.core.config import settings
from app.db.session import get_db
from app.db.models import DatasetModel, ExperimentModel, PredictionHistoryModel

router = APIRouter(tags=["Health & System"])

# Pre-calculate hardware and system properties once to prevent repeated native probe overhead
TORCH_VERSION = torch.__version__
try:
    CUDA_AVAILABLE = bool(torch.cuda.is_available())
    DEVICE_NAME = str(torch.cuda.get_device_name(0)) if CUDA_AVAILABLE else "CPU"
except Exception:
    CUDA_AVAILABLE = False
    DEVICE_NAME = "CPU"
PYTHON_VERSION = sys.version.split()[0]


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Diagnostic health check verifying DB connectivity."""
    db_ok = False
    datasets_cnt = 0
    experiments_cnt = 0
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
        datasets_cnt = db.query(DatasetModel.id).count()
        experiments_cnt = db.query(ExperimentModel.id).count()
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "database_path": str(settings.DATABASE_PATH),
        "pytorch_version": TORCH_VERSION,
        "cuda_available": CUDA_AVAILABLE,
        "device_name": DEVICE_NAME,
        "python_version": PYTHON_VERSION,
        "datasets_count": datasets_cnt,
        "experiments_count": experiments_cnt,
    }


@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Aggregated statistics for the ML Studio overview dashboard."""
    total_datasets = db.query(DatasetModel).count()
    total_experiments = db.query(ExperimentModel).count()
    total_predictions = db.query(PredictionHistoryModel).count()

    scratch_count = (
        db.query(ExperimentModel)
        .filter(ExperimentModel.implementation_type == "scratch")
        .count()
    )
    library_count = (
        db.query(ExperimentModel)
        .filter(ExperimentModel.implementation_type == "library")
        .count()
    )

    recent_experiments = (
        db.query(ExperimentModel)
        .order_by(ExperimentModel.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "total_datasets": total_datasets,
        "total_experiments": total_experiments,
        "total_predictions": total_predictions,
        "scratch_experiments_count": scratch_count,
        "library_experiments_count": library_count,
        "recent_experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "algorithm": exp.algorithm,
                "implementation_type": exp.implementation_type,
                "task_type": exp.task_type,
                "training_time_ms": exp.training_time_ms,
                "metrics": exp.metrics,
                "created_at": exp.created_at,
            }
            for exp in recent_experiments
        ],
    }
