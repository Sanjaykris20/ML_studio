"""Experiment History and Lifecycle API Endpoints.

Handles:
- Querying experiment records
- Inspecting deep metrics, loss curves, and diagnostic chart data
- Deleting experiments and removing associated model artifacts from disk
"""

from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import ExperimentModel
from app.schemas.experiment import ExperimentOut

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=List[Dict[str, Any]])
def list_experiments(db: Session = Depends(get_db)):
    """List all recorded experiments with summary metrics and metadata."""
    experiments = (
        db.query(ExperimentModel)
        .order_by(ExperimentModel.created_at.desc())
        .all()
    )
    return [
        {
            "id": exp.id,
            "name": exp.name,
            "dataset_id": exp.dataset_id,
            "dataset_name": exp.dataset.name if exp.dataset else "Unknown",
            "algorithm": exp.algorithm,
            "implementation_type": exp.implementation_type,
            "task_type": exp.task_type,
            "metrics": exp.metrics,
            "training_time_ms": exp.training_time_ms,
            "status": exp.status,
            "created_at": exp.created_at,
        }
        for exp in experiments
    ]


@router.get("/{experiment_id}", response_model=ExperimentOut)
def get_experiment_details(experiment_id: str, db: Session = Depends(get_db)):
    """Retrieve full experiment details, hyperparameters, and diagnostic curve coordinates."""
    experiment = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_200_OK)
def delete_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Delete an experiment and its saved artifact bundle."""
    experiment = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    # Remove serialized artifact file
    if experiment.artifact_path:
        path = Path(experiment.artifact_path)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    db.delete(experiment)
    db.commit()
    return {"message": f"Experiment '{experiment.name}' deleted successfully."}
