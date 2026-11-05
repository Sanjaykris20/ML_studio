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

    # Verify Feature Importance Comparison in Module 2
    assert "feature_importance" in comp_data["model_a"]
    assert "feature_importance" in comp_data["model_b"]
    assert comp_data["model_a"]["feature_importance"]["supported"] is True
    assert comp_data["model_b"]["feature_importance"]["supported"] is True

    fi_comp = comp_data.get("feature_importance_comparison")
    assert fi_comp is not None
    assert fi_comp["comparison_valid"] is True
    assert fi_comp["comparison_method"] == "Normalized Absolute Coefficients (|w|)"
    assert len(fi_comp["model_a"]["rankings"]) == len(features)
    assert len(fi_comp["model_b"]["rankings"]) == len(features)
    assert len(fi_comp["differences"]) == len(features)


def test_compare_unsupported_mlp_models():
    """Verify that comparing MLP models properly flags feature importance as unsupported with explanation."""
    datasets_res = client.get("/api/datasets")
    diabetes_ds = next(d for d in datasets_res.json() if "Diabetes" in d["name"])
    features = diabetes_ds["feature_names"][:4]

    # Train MLP model
    mlp_payload = {
        "dataset_id": diabetes_ds["id"],
        "experiment_name": "Test MLP Regression",
        "algorithm": "mlp",
        "implementation_type": "scratch",
        "input_features": features,
        "target_column": diabetes_ds["target_column"],
        "task_type": "regression",
        "test_size": 0.2,
        "epochs": 20,
        "hidden_layers": [16, 8],
    }
    mlp_res = client.post("/api/models/train", json=mlp_payload)
    assert mlp_res.status_code == 201
    mlp_id = mlp_res.json()["experiment_id"]

    # Train Linear model
    lin_payload = {
        "dataset_id": diabetes_ds["id"],
        "experiment_name": "Test Linear for MLP Compare",
        "algorithm": "linear_regression",
        "implementation_type": "library",
        "input_features": features,
        "target_column": diabetes_ds["target_column"],
        "task_type": "regression",
        "test_size": 0.2,
        "epochs": 100,
    }
    lin_res = client.post("/api/models/train", json=lin_payload)
    assert lin_res.status_code == 201
    lin_id = lin_res.json()["experiment_id"]

    # Compare Linear (Model A) vs MLP (Model B)
    comp_res = client.get(f"/api/models/compare/{lin_id}/{mlp_id}")
    assert comp_res.status_code == 200
    comp_data = comp_res.json()

    fi_comp = comp_data["feature_importance_comparison"]
    assert fi_comp is not None
    # Overall metric comparison is valid, but feature importance comparison is NOT valid
    assert fi_comp["comparison_valid"] is False
    assert fi_comp["differences"] == {}
    assert "does not support direct coefficient-based feature importance" in fi_comp["explanation"]

    # Model A (Linear) is supported
    assert fi_comp["model_a"]["supported"] is True
    assert len(fi_comp["model_a"]["rankings"]) == len(features)

    # Model B (MLP) is unsupported
    assert fi_comp["model_b"]["supported"] is False
    assert fi_comp["model_b"]["method"] == "Not applicable"
    assert "Multilayer Perceptron (MLP) consists of non-linear hidden layers" in fi_comp["model_b"]["explanation"]
    assert fi_comp["model_b"]["rankings"] == []


def test_compare_older_experiments_bundle_fallback():
    """Verify that older experiments lacking feature_importance in metrics fallback to artifact bundle extraction."""
    from app.db.session import SessionLocal
    from app.db.models import ExperimentModel

    datasets_res = client.get("/api/datasets")
    diabetes_ds = next(d for d in datasets_res.json() if "Diabetes" in d["name"])
    features = diabetes_ds["feature_names"][:3]

    payload = {
        "dataset_id": diabetes_ds["id"],
        "experiment_name": "Test Fallback Model",
        "algorithm": "linear_regression",
        "implementation_type": "scratch",
        "input_features": features,
        "target_column": diabetes_ds["target_column"],
        "task_type": "regression",
        "test_size": 0.2,
        "epochs": 100,
    }
    res_a = client.post("/api/models/train", json=payload)
    id_a = res_a.json()["experiment_id"]

    payload["implementation_type"] = "library"
    res_b = client.post("/api/models/train", json=payload)
    id_b = res_b.json()["experiment_id"]

    # Simulate legacy database record: strip 'feature_importance' from experiment record A
    db = SessionLocal()
    try:
        exp_record = db.query(ExperimentModel).filter(ExperimentModel.id == id_a).first()
        assert exp_record is not None
        legacy_metrics = dict(exp_record.metrics)
        legacy_metrics.pop("feature_importance", None)
        exp_record.metrics = legacy_metrics
        db.commit()
    finally:
        db.close()

    # Call comparison API
    comp_res = client.get(f"/api/models/compare/{id_a}/{id_b}")
    assert comp_res.status_code == 200
    comp_data = comp_res.json()

    # Verify bundle extraction fallback succeeded for Model A
    fi_comp = comp_data["feature_importance_comparison"]
    assert fi_comp is not None
    assert fi_comp["comparison_valid"] is True
    assert fi_comp["model_a"]["supported"] is True
    assert len(fi_comp["model_a"]["rankings"]) == len(features)
    assert fi_comp["model_b"]["supported"] is True
    assert len(fi_comp["differences"]) == len(features)

