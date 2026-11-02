"""Automated Data Cleaning and Transformation Engine for ML Studio.

Capabilities:
1. Column name sanitization (strip whitespace, normalize names).
2. Duplicate row detection and removal.
3. String cell sanitization & dirty number conversion (currency signs, percentages, commas).
4. Missing value detection and smart imputation (median for skewed numbers, mean for normal, mode for categorical).
5. High-missingness (>70%) and constant (zero variance) column pruning.
6. Optional IQR outlier capping to stabilize gradient descent.
7. Generates a comprehensive before/after Data Cleaning Audit Report.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


class DataCleaner:
    """Automated tabular dataset cleaner."""

    def __init__(
        self,
        handle_duplicates: bool = True,
        impute_missing: bool = True,
        cap_outliers: bool = True,
        clean_strings: bool = True,
        drop_constant_columns: bool = True,
        max_missing_ratio_to_drop_col: float = 0.70,
    ):
        self.handle_duplicates = handle_duplicates
        self.impute_missing = impute_missing
        self.cap_outliers = cap_outliers
        self.clean_strings = clean_strings
        self.drop_constant_columns = drop_constant_columns
        self.max_missing_ratio_to_drop_col = max_missing_ratio_to_drop_col

    def clean(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean dataset and produce a detailed audit report."""
        df_clean = df.copy()
        report: Dict[str, Any] = {
            "initial_rows": int(len(df_clean)),
            "initial_columns": int(len(df_clean.columns)),
            "dropped_duplicates": 0,
            "dropped_columns": [],
            "missing_values_before": int(df_clean.isna().sum().sum()),
            "missing_values_after": 0,
            "imputed_cells": 0,
            "outliers_capped": 0,
            "converted_columns": [],
            "actions_log": [],
        }

        # 1. Clean Column Names
        cleaned_col_names = {}
        for col in df_clean.columns:
            new_col = str(col).strip()
            if new_col != col:
                cleaned_col_names[col] = new_col
        if cleaned_col_names:
            df_clean.rename(columns=cleaned_col_names, inplace=True)
            report["actions_log"].append(
                f"Cleaned whitespace in {len(cleaned_col_names)} column names."
            )
            if target_column and target_column in cleaned_col_names:
                target_column = cleaned_col_names[target_column]

        # 2. Remove Duplicate Rows
        if self.handle_duplicates:
            dup_count = int(df_clean.duplicated().sum())
            if dup_count > 0:
                df_clean.drop_duplicates(inplace=True)
                report["dropped_duplicates"] = dup_count
                report["actions_log"].append(f"Removed {dup_count} duplicate rows.")

        # 3. Clean string values & attempt numeric coercion for currency, %, commas
        if self.clean_strings:
            for col in df_clean.columns:
                if df_clean[col].dtype == object or pd.api.types.is_string_dtype(df_clean[col]):
                    # Strip whitespace from string cells
                    df_clean[col] = df_clean[col].astype(str).str.strip()
                    # Replace 'nan', 'null', 'none', '' with np.nan
                    df_clean[col] = df_clean[col].replace(["nan", "None", "null", "NULL", ""], np.nan)

                    # Check if column could be numeric (e.g. "$120.50", "45%", "1,200")
                    sample = df_clean[col].dropna()
                    if not sample.empty:
                        cleaned_sample = (
                            sample.str.replace("$", "", regex=False)
                            .str.replace("€", "", regex=False)
                            .str.replace("£", "", regex=False)
                            .str.replace("%", "", regex=False)
                            .str.replace(",", "", regex=False)
                        )
                        numeric_attempt = pd.to_numeric(cleaned_sample, errors="coerce")
                        # If >= 90% of non-null values convert cleanly, cast entire column to float
                        if numeric_attempt.notna().sum() / len(sample) >= 0.90:
                            df_clean[col] = pd.to_numeric(
                                df_clean[col]
                                .astype(str)
                                .str.replace("$", "", regex=False)
                                .str.replace("€", "", regex=False)
                                .str.replace("£", "", regex=False)
                                .str.replace("%", "", regex=False)
                                .str.replace(",", "", regex=False),
                                errors="coerce",
                            )
                            report["converted_columns"].append(str(col))
                            report["actions_log"].append(
                                f"Converted dirty string column '{col}' to numeric float."
                            )

        # 4. Drop columns with excessive missingness (> 70%) unless target
        cols_to_drop = []
        for col in df_clean.columns:
            if col == target_column:
                continue
            missing_ratio = df_clean[col].isna().sum() / max(1, len(df_clean))
            if missing_ratio > self.max_missing_ratio_to_drop_col:
                cols_to_drop.append(col)
            elif self.drop_constant_columns and df_clean[col].nunique(dropna=True) <= 1:
                cols_to_drop.append(col)

        if cols_to_drop:
            df_clean.drop(columns=cols_to_drop, inplace=True)
            report["dropped_columns"] = [str(c) for c in cols_to_drop]
            report["actions_log"].append(
                f"Pruned {len(cols_to_drop)} uninformative columns: {', '.join(cols_to_drop)}"
            )

        # 5. Missing Value Imputation
        if self.impute_missing:
            imputed_cells = 0
            for col in df_clean.columns:
                n_missing = int(df_clean[col].isna().sum())
                if n_missing > 0:
                    imputed_cells += n_missing
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        # Use median if skewed, mean if roughly symmetric
                        skewness = abs(float(df_clean[col].skew())) if len(df_clean[col].dropna()) > 2 else 0.0
                        if skewness > 1.5:
                            fill_val = df_clean[col].median()
                            method = "median"
                        else:
                            fill_val = df_clean[col].mean()
                            method = "mean"
                        fill_val = 0.0 if pd.isna(fill_val) else fill_val
                        df_clean[col] = df_clean[col].fillna(fill_val)
                        report["actions_log"].append(
                            f"Imputed {n_missing} missing values in numeric feature '{col}' with {method} ({round(float(fill_val), 3)})."
                        )
                    else:
                        mode_series = df_clean[col].mode()
                        fill_val = str(mode_series.iloc[0]) if not mode_series.empty else "Unknown"
                        df_clean[col] = df_clean[col].fillna(fill_val)
                        report["actions_log"].append(
                            f"Imputed {n_missing} missing values in categorical feature '{col}' with mode ('{fill_val}')."
                        )
            report["imputed_cells"] = imputed_cells

        # 6. Outlier Capping via IQR (Interquartile Range) for numerical features
        if self.cap_outliers:
            total_capped = 0
            for col in df_clean.columns:
                if col == target_column:
                    continue  # Do not alter ground truth target values
                if pd.api.types.is_numeric_dtype(df_clean[col]) and df_clean[col].nunique() > 10:
                    q25 = float(df_clean[col].quantile(0.25))
                    q75 = float(df_clean[col].quantile(0.75))
                    iqr = q75 - q25
                    if iqr > 1e-8:
                        lower_bound = q25 - 2.5 * iqr
                        upper_bound = q75 + 2.5 * iqr
                        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                        outliers_count = int(outliers_mask.sum())
                        if outliers_count > 0:
                            total_capped += outliers_count
                            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                            report["actions_log"].append(
                                f"Capped {outliers_count} extreme outliers in '{col}' to range [{round(lower_bound, 2)}, {round(upper_bound, 2)}]."
                            )
            report["outliers_capped"] = total_capped

        report["final_rows"] = int(len(df_clean))
        report["final_columns"] = int(len(df_clean.columns))
        report["missing_values_after"] = int(df_clean.isna().sum().sum())
        report["columns"] = [str(c) for c in df_clean.columns]

        return df_clean, report
