"""Dataset Management API Endpoints.

Handles:
- CSV dataset uploads with size checks and path sanitization
- Listing built-in and user-uploaded datasets
- Rich dataset preview (head rows, numerical/categorical detection, missingness, duplicates, stats, correlation matrix)
- Dataset deletion
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db.models import DatasetModel
from app.schemas.dataset import DatasetOut, DatasetPreviewOut

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def sanitize_filename(filename: str) -> str:
    """Strip unsafe path traversal tokens and return safe filename."""
    cleaned = Path(filename).name
    # Keep alphanumeric, dashes, dots, underscores
    safe_chars = [c for c in cleaned if c.isalnum() or c in ("-", "_", ".")]
    res = "".join(safe_chars)
    return res if res and res != ".csv" else f"upload_{uuid.uuid4().hex[:6]}.csv"


def load_csv_safely(filepath: Path) -> pd.DataFrame:
    """Load tabular data with automatic encoding and delimiter detection."""
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    delimiters = [",", ";", "\t", "|", None]

    df: Optional[pd.DataFrame] = None
    last_err: Optional[Exception] = None

    for enc in encodings:
        for sep in delimiters:
            try:
                if sep is None:
                    test_df = pd.read_csv(
                        filepath,
                        encoding=enc,
                        engine="python",
                        on_bad_lines="skip",
                    )
                else:
                    test_df = pd.read_csv(
                        filepath,
                        sep=sep,
                        encoding=enc,
                        on_bad_lines="skip",
                    )

                if len(test_df.columns) > 1 and len(test_df) > 0:
                    df = test_df
                    break
                elif df is None and len(test_df.columns) >= 1 and len(test_df) > 0:
                    df = test_df
            except Exception as e:
                last_err = e
                continue
        if df is not None and len(df.columns) > 1:
            break

    if df is None:
        raise ValueError(
            f"Could not parse CSV file. Please verify it is a valid delimited tabular text file. (Details: {last_err})"
        )

    # Clean whitespace and deduplicate column headers
    clean_cols = []
    seen = {}
    for c in df.columns:
        c_str = str(c).strip()
        if not c_str or c_str.lower().startswith("unnamed"):
            c_str = "feature"
        if c_str in seen:
            seen[c_str] += 1
            clean_cols.append(f"{c_str}_{seen[c_str]}")
        else:
            seen[c_str] = 0
            clean_cols.append(c_str)
    df.columns = clean_cols

    # Drop index column if present as first column
    if len(df.columns) > 2:
        first_col = df.columns[0]
        if first_col.lower() in ("unnamed: 0", "index", "id_index"):
            df = df.iloc[:, 1:]

    return df


def safe_json_val(v: Any) -> Any:
    """Convert numpy/pandas types to standard JSON-serializable primitives, replacing NaN/Inf with None."""
    if v is None:
        return None
    try:
        if pd.isna(v) or np.isneginf(v) or np.isposinf(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v) if not isinstance(v, (str, list, dict)) else v


@router.post("/upload", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    task_type: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a CSV dataset with file validation, encoding sniffing, and task type detection."""
    filename_lower = (file.filename or "").lower()
    # 1. Extension check (allow .csv, .tsv, .txt)
    allowed_exts = (".csv", ".tsv", ".txt")
    if not any(filename_lower.endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Supported formats: .csv, .tsv, .txt",
        )

    # 2. File size check (read in chunks up to MAX_UPLOAD_SIZE_MB)
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 3. Safe save to disk
    settings.ensure_directories()
    safe_name = sanitize_filename(file.filename or "dataset.csv")
    unique_filename = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    save_path = settings.DATA_DIR / unique_filename

    try:
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file to disk: {str(e)}",
        )

    # 4. Parse CSV using robust loader
    try:
        df = load_csv_safely(save_path)
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    if df.empty or len(df.columns) < 2:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must contain at least 2 columns and at least 1 row. Please verify the CSV delimiter.",
        )

    # 5. Detect or validate task type
    inferred_task = task_type
    last_col = df.columns[-1]
    if not inferred_task:
        target_series = df.iloc[:, -1].dropna()
        n_unique = int(target_series.nunique()) if not target_series.empty else 0
        if n_unique <= 0:
            inferred_task = "regression"
        elif pd.api.types.is_numeric_dtype(target_series) and n_unique > 10:
            inferred_task = "regression"
        elif n_unique == 2:
            inferred_task = "binary_classification"
        else:
            inferred_task = "multiclass_classification"

    # Derive display name
    raw_stem = Path(file.filename or "dataset").stem
    clean_display_name = (
        name.strip()
        if (name and name.strip())
        else (raw_stem if raw_stem else safe_name.replace(".csv", ""))
    )

    dataset_record = DatasetModel(
        name=clean_display_name,
        filename=unique_filename,
        filepath=str(save_path),
        task_type=inferred_task,
        total_rows=len(df),
        total_columns=len(df.columns),
        feature_names=[str(c) for c in df.columns],
        target_column=str(last_col),
        is_builtin=False,
        description=description,
    )

    try:
        db.add(dataset_record)
        db.commit()
        db.refresh(dataset_record)
    except Exception as e:
        db.rollback()
        if save_path.exists():
            try:
                save_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register dataset in database: {str(e)}",
        )
    return dataset_record


@router.get("", response_model=List[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    """List all available datasets (built-in and uploaded)."""
    return db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Retrieve metadata for a specific dataset."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewOut)
def preview_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Preview dataset: head rows, column classification, statistics, and correlation matrix."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    path = Path(dataset.filepath)
    if not path.exists():
        # Fallback to DATA_DIR / filename
        path = settings.DATA_DIR / dataset.filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset file missing on disk.")

    try:
        df = load_csv_safely(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

    # 1. Feature type classification
    numerical_cols = []
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(str(col))
        else:
            categorical_cols.append(str(col))

    # 2. Missing values & duplicate rows
    missing_counts = {str(col): int(df[col].isna().sum()) for col in df.columns}
    duplicate_count = int(df.duplicated().sum())

    # 3. Statistical summaries (NaN/Inf safe)
    summary_stats: Dict[str, Dict[str, Any]] = {}
    for col in numerical_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            summary_stats[col] = {
                "type": "numerical",
                "count": int(series.count()),
                "mean": safe_json_val(series.mean()) or 0.0,
                "std": safe_json_val(series.std()) if len(series) > 1 else 0.0,
                "min": safe_json_val(series.min()) or 0.0,
                "25%": safe_json_val(series.quantile(0.25)) or 0.0,
                "50%": safe_json_val(series.median()) or 0.0,
                "75%": safe_json_val(series.quantile(0.75)) or 0.0,
                "max": safe_json_val(series.max()) or 0.0,
            }
        else:
            summary_stats[col] = {
                "type": "numerical",
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "25%": 0.0,
                "50%": 0.0,
                "75%": 0.0,
                "max": 0.0,
            }

    for col in categorical_cols:
        series = df[col].dropna().astype(str)
        vc = series.value_counts()
        mode_val = str(vc.index[0]) if not vc.empty else ""
        freq_val = int(vc.iloc[0]) if not vc.empty else 0
        summary_stats[col] = {
            "type": "categorical",
            "count": int(len(series)),
            "unique": int(series.nunique()),
            "top": mode_val,
            "freq": freq_val,
        }

    # 4. Correlation Matrix for numerical features (strictly finite floats)
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    if len(numerical_cols) >= 2:
        try:
            num_df = df[numerical_cols].apply(pd.to_numeric, errors="coerce")
            corr_df = num_df.corr()
            correlation_matrix = {}
            for row in corr_df.index:
                row_str = str(row)
                correlation_matrix[row_str] = {}
                for col in corr_df.columns:
                    col_str = str(col)
                    val = corr_df.loc[row, col]
                    if hasattr(val, "iloc"):
                        val = val.iloc[0]
                    if pd.isna(val) or np.isinf(val):
                        correlation_matrix[row_str][col_str] = 0.0
                    else:
                        correlation_matrix[row_str][col_str] = round(float(val), 3)
        except Exception:
            correlation_matrix = None

    # 5. Sanitize head rows for JSON serialization
    head_df = df.head(25)
    head_rows: List[Dict[str, Any]] = []
    for r in head_df.to_dict(orient="records"):
        head_rows.append({str(k): safe_json_val(v) for k, v in r.items()})

    return DatasetPreviewOut(
        id=dataset.id,
        name=dataset.name,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=[str(c) for c in df.columns],
        head_rows=head_rows,
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        missing_values=missing_counts,
        duplicate_rows=duplicate_count,
        summary_statistics=summary_stats,
        correlation_matrix=correlation_matrix,
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_200_OK)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset and remove its physical file (built-in datasets are protected)."""
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if dataset.is_builtin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Built-in benchmark datasets cannot be deleted.",
        )

    # Delete physical file
    path = Path(dataset.filepath)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass

    db.delete(dataset)
    db.commit()
    return {"message": f"Dataset '{dataset.name}' successfully deleted."}
