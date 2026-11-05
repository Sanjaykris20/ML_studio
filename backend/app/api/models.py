"""Model Training, Prediction, and Comparison API Endpoints.

Handles:
- Rigorous model training (Linear Regression, Logistic Regression, MLP)
- Scratch vs. Library execution with fair parity
- Dynamic input schema generation for Prediction Studio
- Instant real-time inference with exact saved preprocessor
- Side-by-side Model Comparison
"""

import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db.models import DatasetModel, ExperimentModel, PredictionHistoryModel
from app.schemas.training import TrainRequest, TrainResponse
from app.schemas.prediction import PredictRequest, PredictResponse, ModelFeatureInfo, PredictionHistoryOut
from app.schemas.experiment import ModelComparisonOut, FeatureImportanceComparisonOut
from app.ml.preprocessing import DataPreprocessor
from app.ml.evaluation import evaluate_regression, evaluate_classification, extract_feature_importance
from app.ml.pipeline import PipelineBundle, save_pipeline_bundle, load_pipeline_bundle
from app.ml.scratch.linear_regression import ScratchLinearRegression
from app.ml.scratch.logistic_regression import ScratchLogisticRegression
from app.ml.scratch.mlp import ScratchPyTorchMLP
from app.ml.library.sklearn_models import (
    SklearnLinearRegressionWrapper,
    SklearnLogisticRegressionWrapper,
    SklearnMLPWrapper,
)
from app.api.datasets import load_csv_safely

router = APIRouter(prefix="/models", tags=["Models & Training"])


def validate_algorithm_compatibility(
    algorithm: str,
    task_type: str,
    n_unique_target: int,
    target_col: str = "target",
    is_numeric: bool = True,
) -> None:
    """Ensure selected algorithm is mathematically compatible with the target."""
    algo = algorithm.lower()
    task = task_type.lower()

    if algo == "linear_regression":
        if not is_numeric:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Linear Regression requires a numeric target variable, but '{target_col}' contains text categories. Please choose a numeric column (like sales, price, amount) or select Multilayer Perceptron (MLP) for classification.",
            )
        if task not in ("regression",):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Linear Regression only supports regression tasks, but dataset target '{target_col}' has task '{task}'.",
            )
    elif algo == "logistic_regression":
        if task != "binary_classification" or n_unique_target != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Binary Logistic Regression requires exactly 2 distinct classes. Target '{target_col}' has {n_unique_target} classes. For 3+ classes, please use Multilayer Perceptron (MLP).",
            )
    elif algo == "mlp":
        # MLP supports all three tasks (regression, binary, multiclass)
        pass
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported algorithm '{algorithm}'. Choose linear_regression, logistic_regression, or mlp.",
        )


@router.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED)
def train_model(req: TrainRequest, db: Session = Depends(get_db)):

    """Train machine learning model (scratch or library) with strict leakage prevention."""
    # 1. Fetch Dataset
    dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Selected dataset not found.")

    csv_path = Path(dataset.filepath)
    if not csv_path.exists():
        csv_path = settings.DATA_DIR / dataset.filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file missing on disk.")

    try:
        df = load_csv_safely(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

    # 2. Validate columns
    if req.target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{req.target_column}' does not exist in dataset.",
        )

    # Filter target out of input features if present
    features_to_use = [f for f in req.input_features if f != req.target_column]
    if not features_to_use:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please select at least one input feature distinct from the target column.",
        )

    for feat in features_to_use:
        if feat not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input feature '{feat}' not found in dataset.",
            )

    # 3. Determine task type
    target_series = df[req.target_column].dropna()
    n_unique_target = int(target_series.nunique()) if not target_series.empty else 0
    is_numeric_target = bool(pd.api.types.is_numeric_dtype(target_series))

    if n_unique_target == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{req.target_column}' is empty or contains only missing values.",
        )

    task_type = req.task_type
    if not task_type:
        if is_numeric_target:
            if req.algorithm == "linear_regression":
                task_type = "regression"
            elif n_unique_target == 2 and req.algorithm == "logistic_regression":
                task_type = "binary_classification"
            elif n_unique_target > 10:
                task_type = "regression"
            elif n_unique_target == 2:
                task_type = "binary_classification"
            else:
                task_type = "regression"  # Default numeric to regression
        else:
            if n_unique_target == 2:
                task_type = "binary_classification"
            else:
                task_type = "multiclass_classification"

    validate_algorithm_compatibility(
        req.algorithm,
        task_type,
        n_unique_target,
        target_col=req.target_column,
        is_numeric=is_numeric_target,
    )

    # Clean target: drop rows where target is NaN
    valid_mask = ~df[req.target_column].isna()
    df_clean = df.loc[valid_mask].copy()

    if len(df_clean) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset has only {len(df_clean)} valid rows after dropping missing targets. At least 5 rows are required.",
        )

    X_df = df_clean[features_to_use]
    y_raw = df_clean[req.target_column]

    # Convert binary/multiclass categorical labels to 0-indexed integers
    classes_list = None
    if task_type in ["binary_classification", "multiclass_classification"]:
        unique_vals = sorted(y_raw.unique())
        if len(unique_vals) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Classification requires at least 2 distinct classes. Target '{req.target_column}' only contains 1 unique value: '{unique_vals[0]}'.",
            )
        class_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        y_arr = np.array([class_to_idx[val] for val in y_raw])
        classes_list = [str(v) for v in unique_vals]
    else:
        y_arr = pd.to_numeric(y_raw, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


    # 4. Train / Test Split (Strict leakage prevention)
    stratify = y_arr if task_type != "regression" else None
    try:
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y_arr,
            test_size=req.test_size,
            random_state=req.random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y_arr,
            test_size=req.test_size,
            random_state=req.random_state,
            stratify=None,
        )

    # 5. Fit Preprocessor strictly on training data!
    num_cols = [c for c in features_to_use if pd.api.types.is_numeric_dtype(X_train_df[c])]
    cat_cols = [c for c in features_to_use if c not in num_cols]

    preprocessor = DataPreprocessor(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        imputation_strategy=req.imputation,
        encoding_strategy=req.encoding,
        scaling_strategy=req.scaling,
    )
    try:
        X_train_trans = preprocessor.fit_transform(X_train_df)
        X_test_trans = preprocessor.transform(X_test_df)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessing failed on selected features: {str(e)}",
        )

    # 6. Instantiate Algorithm (Scratch vs Library)
    is_scratch = req.implementation_type.lower() == "scratch"
    algo_name = req.algorithm.lower()

    start_time = time.perf_counter()

    if is_scratch:
        if algo_name == "linear_regression":
            model = ScratchLinearRegression(
                learning_rate=req.learning_rate,
                epochs=req.epochs,
                random_state=req.random_state,
            )
        elif algo_name == "logistic_regression":
            model = ScratchLogisticRegression(
                learning_rate=req.learning_rate,
                epochs=req.epochs,
                threshold=req.threshold,
                random_state=req.random_state,
            )
        elif algo_name == "mlp":
            model = ScratchPyTorchMLP(
                hidden_layers=req.hidden_layers,
                activation=req.activation,
                optimizer=req.optimizer,
                learning_rate=req.learning_rate,
                epochs=req.epochs,
                batch_size=req.batch_size,
                task=task_type,
                random_state=req.random_state,
            )
    else:  # Library implementation (Scikit-learn)
        if algo_name == "linear_regression":
            model = SklearnLinearRegressionWrapper(fit_intercept=True)
        elif algo_name == "logistic_regression":
            model = SklearnLogisticRegressionWrapper(
                learning_rate=req.learning_rate,
                epochs=req.epochs,
                threshold=req.threshold,
                random_state=req.random_state,
            )
        elif algo_name == "mlp":
            model = SklearnMLPWrapper(
                hidden_layers=req.hidden_layers,
                activation=req.activation,
                optimizer=req.optimizer,
                learning_rate=req.learning_rate,
                epochs=req.epochs,
                batch_size=req.batch_size,
                task=task_type,
                random_state=req.random_state,
            )

    # 7. Model Training & Loss Curve extraction
    try:
        model.fit(X_train_trans, y_train, X_val=X_test_trans, y_val=y_test)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model training calculation failed: {str(e)}",
        )

    training_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 8. Evaluation on Test Data
    try:
        y_pred = model.predict(X_test_trans)

        if task_type == "regression":
            eval_metrics = evaluate_regression(y_test, y_pred)
        else:
            y_prob = model.predict_proba(X_test_trans) if hasattr(model, "predict_proba") else None
            eval_metrics = evaluate_classification(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                classes=list(range(len(classes_list))) if classes_list else None,
            )
            if classes_list:
                eval_metrics["class_labels"] = classes_list
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model evaluation failed: {str(e)}",
        )

    # 8b. Extract feature importance for supported models and persist in metrics
    try:
        feat_importance = extract_feature_importance(
            model=model,
            feature_names=preprocessor.transformed_feature_names_,
            algorithm=req.algorithm,
        )
        eval_metrics["feature_importance"] = feat_importance
    except Exception as e:
        eval_metrics["feature_importance"] = {
            "supported": False,
            "method": "Not applicable",
            "explanation": f"Feature importance calculation failed: {str(e)}",
            "rankings": [],
            "scores": {},
            "raw_coefficients": {},
        }

    loss_history = getattr(model, "loss_history", [])
    val_loss_history = getattr(model, "val_loss_history", [])

    # 9. Serialize Model & Preprocessor Bundle
    experiment_id = str(uuid.uuid4())
    artifact_filename = f"bundle_{experiment_id}.joblib"

    try:
        bundle = PipelineBundle(
            model=model,
            preprocessor=preprocessor,
            algorithm=req.algorithm,
            implementation_type=req.implementation_type,
            task_type=task_type,
            feature_names=features_to_use,
            target_column=req.target_column,
            classes=classes_list,
            hyperparameters=req.model_dump(),
        )
        artifact_path = save_pipeline_bundle(bundle, artifact_filename)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save model bundle: {str(e)}",
        )

    # 10. Persist Experiment Record in SQLite
    exp_name = req.experiment_name or f"{req.algorithm.upper()} ({req.implementation_type.capitalize()})"
    experiment_record = ExperimentModel(
        id=experiment_id,
        name=exp_name,
        dataset_id=dataset.id,
        algorithm=req.algorithm,
        implementation_type=req.implementation_type,
        task_type=task_type,
        hyperparameters=req.model_dump(),
        metrics=eval_metrics,
        loss_history=loss_history,
        val_loss_history=val_loss_history,
        feature_names=features_to_use,
        target_column=req.target_column,
        artifact_path=artifact_path,
        training_time_ms=training_duration_ms,
        status="completed",
    )
    db.add(experiment_record)
    db.commit()
    db.refresh(experiment_record)


    return TrainResponse(
        experiment_id=experiment_record.id,
        name=experiment_record.name,
        status=experiment_record.status,
        algorithm=experiment_record.algorithm,
        implementation_type=experiment_record.implementation_type,
        task_type=experiment_record.task_type,
        metrics=eval_metrics,
        loss_history=loss_history,
        val_loss_history=val_loss_history,
        training_time_ms=training_duration_ms,
        feature_names=features_to_use,
        target_column=req.target_column,
        created_at=experiment_record.created_at,
    )



@router.get("", response_model=List[Dict[str, Any]])
def list_trained_models(db: Session = Depends(get_db)):
    """List all trained model experiments."""
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
            "created_at": exp.created_at,
        }
        for exp in experiments
    ]


@router.get("/{experiment_id}/feature-info", response_model=ModelFeatureInfo)
def get_model_feature_info(experiment_id: str, db: Session = Depends(get_db)):
    """Return feature names and stats needed by Prediction Studio to dynamically generate UI inputs."""
    experiment = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Model experiment not found.")

    try:
        bundle = load_pipeline_bundle(experiment.artifact_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model bundle: {str(e)}")

    return ModelFeatureInfo(
        experiment_id=experiment.id,
        name=experiment.name,
        algorithm=experiment.algorithm,
        implementation_type=experiment.implementation_type,
        task_type=experiment.task_type,
        feature_names=experiment.feature_names,
        target_column=experiment.target_column,
        feature_stats=bundle.preprocessor.feature_stats_,
    )


@router.post("/{experiment_id}/predict", response_model=PredictResponse)
async def predict_with_model(
    experiment_id: str,
    req: PredictRequest,
    db: Session = Depends(get_db),
):

    """Generate real-time predictions using the exact saved preprocessing pipeline."""
    experiment = db.query(ExperimentModel).filter(ExperimentModel.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Model experiment not found.")

    try:
        bundle = load_pipeline_bundle(experiment.artifact_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model bundle: {str(e)}")

    # Execute prediction
    try:
        result = bundle.predict_single(req.inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # Store in prediction history
    history_record = PredictionHistoryModel(
        experiment_id=experiment.id,
        input_values=req.inputs,
        predicted_value=str(result["prediction"]),
        probabilities=result.get("probabilities"),
    )
    db.add(history_record)
    db.commit()
    db.refresh(history_record)

    return PredictResponse(
        experiment_id=experiment.id,
        prediction=str(result["prediction"]),
        raw_value=result["raw_value"],
        probabilities=result.get("probabilities"),
        task_type=experiment.task_type,
        timestamp=history_record.created_at,
    )


@router.get("/{experiment_id}/predictions", response_model=List[PredictionHistoryOut])
def get_prediction_history(experiment_id: str, db: Session = Depends(get_db)):
    """Fetch past predictions for a specific model."""
    return (
        db.query(PredictionHistoryModel)
        .filter(PredictionHistoryModel.experiment_id == experiment_id)
        .order_by(PredictionHistoryModel.created_at.desc())
        .limit(100)
        .all()
    )


def get_or_extract_experiment_feature_importance(experiment: ExperimentModel) -> Dict[str, Any]:
    """Retrieve feature importance from metrics, or fallback to bundle extraction for older experiments."""
    metrics = experiment.metrics or {}
    if "feature_importance" in metrics and isinstance(metrics["feature_importance"], dict):
        fi = metrics["feature_importance"]
        if "supported" in fi and "rankings" in fi:
            return fi

    # Fallback for older experiments: load the pipeline bundle and extract importance
    try:
        if experiment.artifact_path:
            bundle = load_pipeline_bundle(experiment.artifact_path)
            feat_names = (
                getattr(bundle.preprocessor, "transformed_feature_names_", None)
                or experiment.feature_names
                or []
            )
            return extract_feature_importance(
                model=bundle.model,
                feature_names=feat_names,
                algorithm=experiment.algorithm,
            )
    except Exception:
        pass

    # If algorithm is MLP
    if (experiment.algorithm or "").lower() == "mlp":
        return {
            "supported": False,
            "method": "Not applicable",
            "explanation": (
                "Multilayer Perceptron (MLP) consists of non-linear hidden layers "
                "without direct linear coefficients. Direct feature importance is "
                "supported for Linear Regression and Logistic Regression models."
            ),
            "rankings": [],
            "scores": {},
            "raw_coefficients": {},
        }

    return {
        "supported": False,
        "method": "Not applicable",
        "explanation": "Feature importance data is unavailable for this model artifact.",
        "rankings": [],
        "scores": {},
        "raw_coefficients": {},
    }


@router.get("/compare/{id_a}/{id_b}", response_model=ModelComparisonOut)
def compare_models(id_a: str, id_b: str, db: Session = Depends(get_db)):
    """Compare two trained models (e.g. Scratch vs. Library) side-by-side."""
    exp_a = db.query(ExperimentModel).filter(ExperimentModel.id == id_a).first()
    exp_b = db.query(ExperimentModel).filter(ExperimentModel.id == id_b).first()

    if not exp_a or not exp_b:
        raise HTTPException(status_code=404, detail="One or both experiments could not be found.")

    if exp_a.task_type != exp_b.task_type:
        return ModelComparisonOut(
            task_type="mismatch",
            model_a={"name": exp_a.name, "task_type": exp_a.task_type},
            model_b={"name": exp_b.name, "task_type": exp_b.task_type},
            metric_differences={},
            comparison_valid=False,
            notes=f"Cannot compare models with differing tasks: '{exp_a.task_type}' vs '{exp_b.task_type}'.",
        )

    # Compute metric differences
    diffs = {}
    metrics_a = exp_a.metrics or {}
    metrics_b = exp_b.metrics or {}

    for key in ["r2_score", "mae", "mse", "rmse", "accuracy", "precision", "recall", "f1_score"]:
        if key in metrics_a and key in metrics_b and metrics_a[key] is not None and metrics_b[key] is not None:
            diffs[key] = round(float(metrics_a[key]) - float(metrics_b[key]), 4)

    # Training time delta
    diffs["training_time_ms_delta"] = round(exp_a.training_time_ms - exp_b.training_time_ms, 2)

    # Feature Importance Extraction & Comparison
    fi_a = get_or_extract_experiment_feature_importance(exp_a)
    fi_b = get_or_extract_experiment_feature_importance(exp_b)

    fi_differences = {}
    fi_comparison_valid = False
    fi_explanation = None

    if fi_a.get("supported") and fi_b.get("supported"):
        fi_comparison_valid = True
        scores_a = fi_a.get("scores", {})
        scores_b = fi_b.get("scores", {})
        # Combine all features in deterministic order (rankings from A then B)
        all_features = list(dict.fromkeys(
            [r["feature"] for r in fi_a.get("rankings", [])] +
            [r["feature"] for r in fi_b.get("rankings", [])]
        ))
        for feat in all_features:
            val_a = scores_a.get(feat, 0.0)
            val_b = scores_b.get(feat, 0.0)
            fi_differences[feat] = round(val_a - val_b, 2)
    else:
        fi_comparison_valid = False
        if not fi_a.get("supported") and not fi_b.get("supported"):
            fi_explanation = "Neither model supports direct coefficient-based feature importance (MLP models lack linear coefficients)."
        elif not fi_a.get("supported"):
            fi_explanation = f"Model A '{exp_a.name}' does not support direct coefficient-based feature importance."
        else:
            fi_explanation = f"Model B '{exp_b.name}' does not support direct coefficient-based feature importance."

    fi_comparison = {
        "model_a": fi_a,
        "model_b": fi_b,
        "differences": fi_differences,
        "comparison_method": "Normalized Absolute Coefficients (|w|)" if fi_comparison_valid else "Not applicable",
        "comparison_valid": fi_comparison_valid,
        "explanation": fi_explanation,
    }

    return ModelComparisonOut(
        task_type=exp_a.task_type,
        model_a={
            "id": exp_a.id,
            "name": exp_a.name,
            "algorithm": exp_a.algorithm,
            "implementation_type": exp_a.implementation_type,
            "metrics": exp_a.metrics,
            "loss_history": exp_a.loss_history,
            "val_loss_history": exp_a.val_loss_history,
            "training_time_ms": exp_a.training_time_ms,
            "created_at": exp_a.created_at,
            "feature_importance": fi_a,
        },
        model_b={
            "id": exp_b.id,
            "name": exp_b.name,
            "algorithm": exp_b.algorithm,
            "implementation_type": exp_b.implementation_type,
            "metrics": exp_b.metrics,
            "loss_history": exp_b.loss_history,
            "val_loss_history": exp_b.val_loss_history,
            "training_time_ms": exp_b.training_time_ms,
            "created_at": exp_b.created_at,
            "feature_importance": fi_b,
        },
        metric_differences=diffs,
        comparison_valid=True,
        feature_importance_comparison=fi_comparison,
    )
