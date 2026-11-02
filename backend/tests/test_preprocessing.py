"""Tests for Data Preprocessing and Leakage Prevention."""

import numpy as np
import pandas as pd
import pytest
from app.ml.preprocessing import DataPreprocessor


def test_preprocessing_leakage_prevention():
    """Verify that scaler mean and std are learned ONLY from train data and applied to test data."""
    train_df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0],
        "cat": ["A", "B", "A"]
    })
    test_df = pd.DataFrame({
        "num": [100.0],
        "cat": ["B"]
    })

    preprocessor = DataPreprocessor(
        numerical_features=["num"],
        categorical_features=["cat"],
        scaling_strategy="standard",
        encoding_strategy="onehot",
    )

    X_train = preprocessor.fit_transform(train_df)
    # Train mean is 20.0
    assert preprocessor.feature_stats_["num"]["mean"] == 20.0

    X_test = preprocessor.transform(test_df)
    assert X_test.shape[1] == X_train.shape[1]

    # Verify single-row inference transform works
    single_res = preprocessor.transform_single({"num": 25.0, "cat": "A"})
    assert single_res.shape == (1, X_train.shape[1])


def test_preprocessing_missing_value_imputation():
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0],
        "cat": ["red", None, "blue", "red"]
    })

    preprocessor = DataPreprocessor(
        numerical_features=["num"],
        categorical_features=["cat"],
        imputation_strategy="mean",
        encoding_strategy="onehot",
    )

    X = preprocessor.fit_transform(df)
    assert not np.any(np.isnan(X))
