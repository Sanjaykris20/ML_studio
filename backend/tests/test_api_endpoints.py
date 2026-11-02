"""Tests for FastAPI Endpoints using TestClient:
- Health and diagnostics
- Dataset listing and rich preview
- Model training (Scratch and Library)
- Real-time inference
- Model comparison
- Experiment lifecycle
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
    assert "pytorch_version" in data


def test_datasets_list_and_preview():
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.json()
    assert len(datasets) >= 3  # Built-ins should be present

    # Test preview on the first dataset
    first_id = datasets[0]["id"]
    preview_res = client.get(f"/api/datasets/{first_id}/preview")
    assert preview_res.status_code == 200
    preview_data = preview_res.json()
    assert len(preview_data["head_rows"]) > 0
    assert "summary_statistics" in preview_data


def test_train_scratch_linear_regression_and_predict():
    # 1. Find Diabetes dataset
    datasets_res = client.get("/api/datasets")
    diabetes_ds = next(d for d in datasets_res.json() if "Diabetes" in d["name"])

    # 2. Train scratch Linear Regression
    train_payload = {
        "dataset_id": diabetes_ds["id"],
        "experiment_name": "Test Linear Reg Scratch",
        "algorithm": "linear_regression",
        "implementation_type": "scratch",
        "input_features": diabetes_ds["feature_names"][:5],  # Use first 5 features
        "target_column": diabetes_ds["target_column"],
        "task_type": "regression",
        "test_size": 0.2,
        "random_state": 42,
        "scaling": "standard",
        "imputation": "mean",
        "learning_rate": 0.05,
        "epochs": 150,
    }
    train_res = client.post("/api/models/train", json=train_payload)
    assert train_res.status_code == 201
    exp_data = train_res.json()
    assert exp_data["status"] == "completed"
    assert "r2_score" in exp_data["metrics"]
    assert len(exp_data["loss_history"]) > 0
    exp_id = exp_data["experiment_id"]

    # 3. Predict using the trained model
    feature_info_res = client.get(f"/api/models/{exp_id}/feature-info")
    assert feature_info_res.status_code == 200

    # Build input dictionary using means
    stats = feature_info_res.json()["feature_stats"]
    input_vals = {feat: stats[feat].get("mean", 0.0) for feat in train_payload["input_features"]}

    predict_res = client.post(
        f"/api/models/{exp_id}/predict",
        json={"experiment_id": exp_id, "inputs": input_vals},
    )
    assert predict_res.status_code == 200
    pred_data = predict_res.json()
    assert "prediction" in pred_data
    assert pred_data["task_type"] == "regression"

    # 4. Check prediction history
    hist_res = client.get(f"/api/models/{exp_id}/predictions")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) >= 1


def test_train_library_linear_regression_and_compare():
    # 1. Find Diabetes dataset
    datasets_res = client.get("/api/datasets")
    diabetes_ds = next(d for d in datasets_res.json() if "Diabetes" in d["name"])
    features = diabetes_ds["feature_names"][:5]

    # Train Library model
    lib_payload = {
        "dataset_id": diabetes_ds["id"],
        "experiment_name": "Test Linear Reg Library",
        "algorithm": "linear_regression",
        "implementation_type": "library",
        "input_features": features,
        "target_column": diabetes_ds["target_column"],
        "task_type": "regression",
        "test_size": 0.2,
        "random_state": 42,
        "scaling": "standard",
        "imputation": "mean",
        "epochs": 150,
    }
    lib_res = client.post("/api/models/train", json=lib_payload)
    assert lib_res.status_code == 201
    lib_id = lib_res.json()["experiment_id"]

    # Also train Scratch model with same setup
    scratch_payload = dict(lib_payload)
    scratch_payload["implementation_type"] = "scratch"
    scratch_payload["experiment_name"] = "Test Linear Reg Scratch for Compare"
    scratch_res = client.post("/api/models/train", json=scratch_payload)
    assert scratch_res.status_code == 201
    scratch_id = scratch_res.json()["experiment_id"]

    # Compare them
    comp_res = client.get(f"/api/models/compare/{scratch_id}/{lib_id}")
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert comp_data["comparison_valid"] is True
    assert "metric_differences" in comp_data
