"""Model & Preprocessing Pipeline Bundle Manager.

Ensures safe persistence and exact reconstruction of:
1. Fitted DataPreprocessor (preventing desynchronization / leakage)
2. Trained Model instance (NumPy Scratch, PyTorch MLP, or Scikit-learn)
3. Feature names, target mappings, and hyperparameters
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.ml.preprocessing import DataPreprocessor


class PipelineBundle:
    """Encapsulates a trained model and its associated preprocessor for inference."""

    def __init__(
        self,
        model: Any,
        preprocessor: DataPreprocessor,
        algorithm: str,
        implementation_type: str,
        task_type: str,
        feature_names: list,
        target_column: str,
        classes: Optional[list] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.algorithm = algorithm
        self.implementation_type = implementation_type
        self.task_type = task_type
        self.feature_names = feature_names
        self.target_column = target_column
        self.classes = classes
        self.hyperparameters = hyperparameters or {}

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for a single input dictionary provided by Prediction Studio."""
        # Preprocess single record
        X_trans = self.preprocessor.transform_single(input_dict)

        # Generate prediction
        pred = self.model.predict(X_trans)
        raw_val = pred[0]

        # Handle class probabilities if classification
        probabilities: Optional[Dict[str, float]] = None
        if hasattr(self.model, "predict_proba") and self.task_type != "regression":
            try:
                probs = self.model.predict_proba(X_trans)[0]
                if self.classes is not None and len(self.classes) == len(probs):
                    probabilities = {str(self.classes[i]): round(float(probs[i]), 4) for i in range(len(probs))}
                else:
                    probabilities = {f"Class {i}": round(float(p), 4) for i, p in enumerate(probs)}
            except Exception:
                probabilities = None

        # Format output value
        if self.task_type == "regression":
            formatted_val = f"{float(raw_val):.4f}"
        else:
            formatted_val = str(int(raw_val) if isinstance(raw_val, (int, np.integer)) else raw_val)

        return {
            "prediction": formatted_val,
            "raw_value": float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
            "probabilities": probabilities,
            "task_type": self.task_type,
        }


def save_pipeline_bundle(bundle: PipelineBundle, filename: str) -> str:
    """Save pipeline bundle to artifacts directory."""
    settings.ensure_directories()
    safe_filename = Path(filename).name  # Prevent path traversal
    full_path = settings.ARTIFACTS_DIR / safe_filename
    joblib.dump(bundle, full_path, compress=3)
    return str(full_path)


def load_pipeline_bundle(artifact_path: str) -> PipelineBundle:
    """Safely load pipeline bundle from disk."""
    path = Path(artifact_path)
    if not path.exists():
        # Try finding in settings.ARTIFACTS_DIR
        alt_path = settings.ARTIFACTS_DIR / path.name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, PipelineBundle):
        raise TypeError(f"Expected PipelineBundle instance, got {type(bundle)}")
    return bundle
