"""Data Preprocessing Pipeline with Strict Data Leakage Prevention.

Key guarantees:
1. All scalers, imputers, and encoders are fit ONLY on the training split (X_train).
2. The exact same fitted transformations are applied to evaluation data (X_test) and
   future user predictions in the Prediction Studio.
3. Unseen categorical levels during inference are gracefully handled (e.g. handle_unknown='ignore').
4. Full metadata is saved for dynamic UI form generation (feature names, types, means/ranges).
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """End-to-end preprocessing pipeline for tabular datasets."""

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        imputation_strategy: str = "mean",  # "mean", "median", "most_frequent", "drop"
        encoding_strategy: str = "onehot",  # "onehot", "ordinal", "none"
        scaling_strategy: str = "standard",  # "standard", "minmax", "none"
    ):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.imputation_strategy = imputation_strategy
        self.encoding_strategy = encoding_strategy
        self.scaling_strategy = scaling_strategy

        # Fitted transformers
        self.num_imputer: Optional[SimpleImputer] = None
        self.cat_imputer: Optional[SimpleImputer] = None
        self.encoder: Optional[Union[OneHotEncoder, OrdinalEncoder]] = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None

        # Metadata recorded during fit
        self.feature_names_in_: List[str] = []
        self.transformed_feature_names_: List[str] = []
        self.feature_stats_: Dict[str, Dict[str, Any]] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit all transformers strictly on the training partition."""
        df = df.copy()
        self.feature_names_in_ = list(df.columns)

        # Detect columns if not specified
        if not self.numerical_features and not self.categorical_features:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.numerical_features.append(col)
                else:
                    self.categorical_features.append(col)

        # Compute summary statistics for UI forms and defaults
        for col in self.numerical_features:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                self.feature_stats_[col] = {
                    "type": "numerical",
                    "min": float(series.min()) if not series.empty else 0.0,
                    "max": float(series.max()) if not series.empty else 1.0,
                    "mean": float(series.mean()) if not series.empty else 0.0,
                    "median": float(series.median()) if not series.empty else 0.0,
                    "std": float(series.std()) if not series.empty else 1.0,
                }

        for col in self.categorical_features:
            if col in df.columns:
                unique_vals = [str(v) for v in df[col].dropna().unique().tolist()]
                top_val = str(df[col].mode().iloc[0]) if not df[col].mode().empty else ""
                self.feature_stats_[col] = {
                    "type": "categorical",
                    "unique_values": unique_vals,
                    "default": top_val,
                }

        # 1. Numerical Imputation
        if self.numerical_features:
            strategy = self.imputation_strategy if self.imputation_strategy in ["mean", "median"] else "mean"
            self.num_imputer = SimpleImputer(strategy=strategy)
            num_data = self.num_imputer.fit_transform(df[self.numerical_features])
        else:
            num_data = np.empty((len(df), 0))

        # 2. Categorical Imputation & Encoding
        if self.categorical_features:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            cat_imputed = self.cat_imputer.fit_transform(df[self.categorical_features].astype(str))

            if self.encoding_strategy == "onehot":
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", max_categories=30)
                cat_encoded = self.encoder.fit_transform(cat_imputed)

                cat_feature_names = list(self.encoder.get_feature_names_out(self.categorical_features))
            elif self.encoding_strategy == "ordinal":
                self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                cat_encoded = self.encoder.fit_transform(cat_imputed)
                cat_feature_names = list(self.categorical_features)
            else:
                cat_encoded = np.empty((len(df), 0))
                cat_feature_names = []
        else:
            cat_encoded = np.empty((len(df), 0))
            cat_feature_names = []

        # 3. Combine before scaling
        if self.numerical_features and self.categorical_features and self.encoding_strategy != "none":
            X_combined = np.hstack([num_data, cat_encoded])
            self.transformed_feature_names_ = list(self.numerical_features) + cat_feature_names
        elif self.numerical_features:
            X_combined = num_data
            self.transformed_feature_names_ = list(self.numerical_features)
        elif self.categorical_features and self.encoding_strategy != "none":
            X_combined = cat_encoded
            self.transformed_feature_names_ = cat_feature_names
        else:
            X_combined = num_data
            self.transformed_feature_names_ = list(self.numerical_features)

        # 4. Feature Scaling (fit strictly on X_train)
        if self.scaling_strategy == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(X_combined)
        elif self.scaling_strategy == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler.fit(X_combined)
        else:
            self.scaler = None

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformations to a dataframe."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        df = df.copy()

        # 1. Numerical transform
        if self.numerical_features:
            # Ensure columns exist, fill missing with mean
            for col in self.numerical_features:
                if col not in df.columns:
                    df[col] = self.feature_stats_.get(col, {}).get("mean", 0.0)
            num_data = self.num_imputer.transform(df[self.numerical_features])
        else:
            num_data = np.empty((len(df), 0))

        # 2. Categorical transform
        if self.categorical_features:
            for col in self.categorical_features:
                if col not in df.columns:
                    df[col] = self.feature_stats_.get(col, {}).get("default", "")
            cat_imputed = self.cat_imputer.transform(df[self.categorical_features].astype(str))
            if self.encoder is not None:
                cat_encoded = self.encoder.transform(cat_imputed)
            else:
                cat_encoded = np.empty((len(df), 0))
        else:
            cat_encoded = np.empty((len(df), 0))

        # Combine
        if self.numerical_features and self.categorical_features and self.encoding_strategy != "none":
            X_combined = np.hstack([num_data, cat_encoded])
        elif self.numerical_features:
            X_combined = num_data
        elif self.categorical_features and self.encoding_strategy != "none":
            X_combined = cat_encoded
        else:
            X_combined = num_data

        # 3. Apply fitted scaler
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_combined)
            return X_scaled
        return X_combined

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return transformed numpy array."""
        return self.fit(df).transform(df)

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Transform a single row user input dictionary from Prediction Studio."""
        df_single = pd.DataFrame([input_dict])
        return self.transform(df_single)
