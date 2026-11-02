"""Automated Pipeline API Endpoints: Data Intake, Auto-Cleaning, Method Recommendation, and One-Click Training."""

import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db.models import DatasetModel, ExperimentModel
from app.schemas.auto_pipeline import (
    CleanDatasetRequest,
    CleanDatasetResponse,
    AutoTrainRequest,
    AutoTrainResponse,
)
from app.schemas.training import TrainRequest, TrainResponse
from app.ml.cleaner import DataCleaner
from app.api.models import train_model
from app.api.datasets import load_csv_safely, safe_json_val

router = APIRouter(prefix="/pipeline", tags=["AutoML & Cleaning Pipeline"])


@router.post("/clean", response_model=CleanDatasetResponse)
def clean_dataset_endpoint(req: CleanDatasetRequest, db: Session = Depends(get_db)):
    """Intake and clean a dataset: drops duplicates, sanitizes dirty strings, imputes missing values, and caps outliers."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    csv_path = Path(dataset.filepath)
    if not csv_path.exists():
        csv_path = settings.DATA_DIR / dataset.filename
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file missing on disk.")

    try:
        df = load_csv_safely(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

    cleaner = DataCleaner(
        handle_duplicates=req.handle_duplicates,
        impute_missing=req.impute_missing,
        cap_outliers=req.cap_outliers,
        clean_strings=req.clean_strings,
    )
    df_clean, report = cleaner.clean(df, target_column=req.target_column)

    # Classify column types
    numerical_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
    categorical_cols = [c for c in df_clean.columns if c not in numerical_cols]

    # Infer target & task
    target_col = req.target_column or (dataset.target_column if dataset.target_column in df_clean.columns else df_clean.columns[-1])
    target_series = df_clean[target_col]
    n_unique = target_series.nunique()
    if pd.api.types.is_numeric_dtype(target_series) and n_unique > 10:
        recommended_task = "regression"
    elif n_unique == 2:
        recommended_task = "binary_classification"
    else:
        recommended_task = "multiclass_classification"

    cleaned_dataset_id = None
    cleaned_name = None

    if req.save_as_new_dataset:
        settings.ensure_directories()
        unique_fn = f"cleaned_{uuid.uuid4().hex[:8]}_{Path(dataset.filename).name}"
        save_path = settings.DATA_DIR / unique_fn
        df_clean.to_csv(save_path, index=False)

        cleaned_name = req.new_dataset_name or f"{dataset.name} (Cleaned)"
        new_record = DatasetModel(
            name=cleaned_name,
            filename=unique_fn,
            filepath=str(save_path),
            task_type=recommended_task,
            total_rows=len(df_clean),
            total_columns=len(df_clean.columns),
            feature_names=[str(c) for c in df_clean.columns],
            target_column=target_col,
            is_builtin=False,
            description=f"Auto-cleaned from '{dataset.name}': {', '.join(report['actions_log'][:3])}",
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        cleaned_dataset_id = new_record.id

    head_rows = [{str(k): safe_json_val(v) for k, v in r.items()} for r in df_clean.head(20).to_dict(orient="records")]

    return CleanDatasetResponse(
        original_dataset_id=dataset.id,
        cleaned_dataset_id=cleaned_dataset_id,
        cleaned_dataset_name=cleaned_name,
        audit_report=report,
        preview_rows=head_rows,
        columns=[str(c) for c in df_clean.columns],
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        recommended_target=target_col,
        recommended_task=recommended_task,
    )


@router.post("/auto-train", response_model=AutoTrainResponse)
def auto_train_endpoint(req: AutoTrainRequest, db: Session = Depends(get_db)):
    """Streamlined AutoML pipeline: optionally clean data, select best/chosen training method, train model, and return outputs."""
    active_dataset_id = req.dataset_id
    cleaning_report = None

    # Step 1: Optional Auto-Cleaning
    if req.auto_clean_first:
        clean_res = clean_dataset_endpoint(
            CleanDatasetRequest(
                dataset_id=req.dataset_id,
                target_column=req.target_column,
                save_as_new_dataset=True,
            ),
            db=db,
        )
        if clean_res.cleaned_dataset_id:
            active_dataset_id = clean_res.cleaned_dataset_id
        cleaning_report = clean_res.audit_report

    # Step 2: Fetch Active Dataset
    dataset = db.query(DatasetModel).filter(DatasetModel.id == active_dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset for training not found.")

    csv_path = Path(dataset.filepath)
    if not csv_path.exists():
        csv_path = settings.DATA_DIR / dataset.filename
    df = load_csv_safely(csv_path)

    target_col = req.target_column if req.target_column in df.columns else dataset.target_column
    if not target_col or target_col not in df.columns:
        target_col = df.columns[-1]

    input_features = [c for c in df.columns if c != target_col]
    target_series = df[target_col]
    n_unique = target_series.nunique()

    # Determine Task
    if pd.api.types.is_numeric_dtype(target_series) and n_unique > 10:
        task_type = "regression"
    elif n_unique == 2:
        task_type = "binary_classification"
    else:
        task_type = "multiclass_classification"

    # Step 3: Determine Algorithm & Implementation based on method
    chosen_method = req.method.lower()
    algo = "linear_regression"
    impl_primary = req.implementation_type.lower()
    train_comparison = False

    if chosen_method == "compare_both":
        train_comparison = True
        if task_type == "regression":
            algo = "linear_regression"
        elif task_type == "binary_classification":
            algo = "logistic_regression"
        else:
            algo = "mlp"
    elif chosen_method in ["scratch_linear", "library_linear"]:
        algo = "linear_regression"
        impl_primary = "scratch" if "scratch" in chosen_method else "library"
    elif chosen_method in ["scratch_logistic", "library_logistic"]:
        algo = "logistic_regression"
        impl_primary = "scratch" if "scratch" in chosen_method else "library"
    elif chosen_method in ["scratch_mlp", "library_mlp"]:
        algo = "mlp"
        impl_primary = "scratch" if "scratch" in chosen_method else "library"
    else:  # auto_recommend
        if task_type == "regression":
            algo = "linear_regression"
        elif task_type == "binary_classification":
            algo = "logistic_regression"
        else:
            algo = "mlp"

    epochs = req.epochs or (80 if algo == "mlp" else 300)
    lr = req.learning_rate or (0.01 if algo == "mlp" else (0.1 if algo == "logistic_regression" else 0.05))

    # Step 4: Train Primary Model
    train_req_primary = TrainRequest(
        dataset_id=active_dataset_id,
        experiment_name=f"AutoTrain: {algo.replace('_', ' ').title()} ({impl_primary})",
        algorithm=algo,
        implementation_type=impl_primary,
        input_features=input_features,
        target_column=target_col,
        task_type=task_type,
        test_size=req.test_size,
        epochs=epochs,
        learning_rate=lr,
        hidden_layers=[64, 32],
        activation="relu",
        optimizer="adam",
    )
    primary_res = train_model(train_req_primary, db=db)

    # Step 5: Optional Dual Comparison Model
    comparison_res = None
    diffs = None

    if train_comparison:
        impl_alt = "library" if impl_primary == "scratch" else "scratch"
        train_req_alt = TrainRequest(
            dataset_id=active_dataset_id,
            experiment_name=f"AutoTrain: {algo.replace('_', ' ').title()} ({impl_alt})",
            algorithm=algo,
            implementation_type=impl_alt,
            input_features=input_features,
            target_column=target_col,
            task_type=task_type,
            test_size=req.test_size,
            epochs=epochs,
            learning_rate=lr,
            hidden_layers=[64, 32],
            activation="relu",
            optimizer="adam",
        )
        comparison_res = train_model(train_req_alt, db=db)

        # Compute metric differences
        diffs = {}
        mA = primary_res.metrics or {}
        mB = comparison_res.metrics or {}
        for k in ["r2_score", "mae", "mse", "accuracy", "f1_score", "roc_auc"]:
            if k in mA and k in mB and mA[k] is not None and mB[k] is not None:
                diffs[k] = round(float(mA[k]) - float(mB[k]), 4)
        diffs["training_time_delta_ms"] = round(primary_res.training_time_ms - comparison_res.training_time_ms, 2)

    # Step 6: Generate Sample Prediction Output
    from app.ml.pipeline import load_pipeline_bundle
    sample_pred_data = None
    try:
        exp_record = db.query(ExperimentModel).filter(ExperimentModel.id == primary_res.experiment_id).first()
        if exp_record:
            bundle = load_pipeline_bundle(exp_record.artifact_path)
            sample_input = {}
            for f in input_features:
                stat = bundle.preprocessor.feature_stats_.get(f, {})
                if stat.get("type") == "numerical":
                    sample_input[f] = stat.get("mean", 0.0)
                else:
                    sample_input[f] = stat.get("default", "")
            pred_out = bundle.predict_single(sample_input)
            sample_pred_data = {
                "inputs": sample_input,
                "prediction": pred_out["prediction"],
                "probabilities": pred_out.get("probabilities"),
                "task_type": task_type,
            }
    except Exception:
        sample_pred_data = None

    summary = (
        f"Successfully cleaned dataset and trained {algo.replace('_', ' ').title()} using {impl_primary} implementation. "
        f"Evaluation achieved on test split: "
        + (f"R² = {primary_res.metrics.get('r2_score')}" if task_type == 'regression' else f"Accuracy = {round(primary_res.metrics.get('accuracy', 0) * 100, 2)}%")
    )

    return AutoTrainResponse(
        dataset_id=active_dataset_id,
        cleaning_report=cleaning_report,
        primary_model=primary_res,
        comparison_model=comparison_res,
        metric_differences=diffs,
        sample_prediction=sample_pred_data,
        summary_message=summary,
    )
