"""Tests for Automated DataCleaner and AutoML Pipeline Endpoints."""

import numpy as np
import pandas as pd
import pytest
from app.ml.cleaner import DataCleaner
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_data_cleaner_dedup_and_impute():
    # Dirty dataframe with whitespace, duplicates, dirty currency string, missing values
    raw_df = pd.DataFrame({
        "  age  ": [25, 30, np.nan, 30],
        "salary": ["$50,000", "$60,000", "$55,000", "$60,000"],
        "city": [" New York ", "London", None, "London"],
    })

    cleaner = DataCleaner(
        handle_duplicates=True,
        impute_missing=True,
        clean_strings=True,
    )
    df_clean, report = cleaner.clean(raw_df)

    # 1. Column names trimmed
    assert "age" in df_clean.columns
    # 2. Duplicate row removed (row 4 is identical to row 2)
    assert report["dropped_duplicates"] == 1
    assert len(df_clean) == 3
    # 3. Dirty currency strings converted to numeric
    assert pd.api.types.is_numeric_dtype(df_clean["salary"])
    assert df_clean["salary"].iloc[0] == 50000.0
    # 4. Missing values imputed
    assert df_clean.isna().sum().sum() == 0
    assert report["missing_values_after"] == 0


def test_pipeline_clean_and_auto_train_endpoints():
    datasets_res = client.get("/api/datasets")
    diabetes = next(d for d in datasets_res.json() if "Diabetes" in d["name"])

    # Test /api/pipeline/clean
    clean_res = client.post("/api/pipeline/clean", json={
        "dataset_id": diabetes["id"],
        "save_as_new_dataset": True,
        "new_dataset_name": "Diabetes Cleaned Test",
    })
    assert clean_res.status_code == 200
    clean_data = clean_res.json()
    assert clean_data["cleaned_dataset_id"] is not None
    assert "audit_report" in clean_data
    assert clean_data["recommended_task"] == "regression"

    # Test /api/pipeline/auto-train
    auto_res = client.post("/api/pipeline/auto-train", json={
        "dataset_id": diabetes["id"],
        "target_column": diabetes["target_column"],
        "auto_clean_first": True,
        "method": "scratch_linear",
        "epochs": 100,
    })
    assert auto_res.status_code == 200
    res_data = auto_res.json()
    assert res_data["primary_model"]["status"] == "completed"
    assert "r2_score" in res_data["primary_model"]["metrics"]
    assert res_data["sample_prediction"] is not None
    assert "prediction" in res_data["sample_prediction"]
