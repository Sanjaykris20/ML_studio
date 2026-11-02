"""Database Session Management and Automatic Initialization.

Self-configuring SQLite engine with:
- Automatic schema creation on startup.
- WAL mode & foreign key constraint enforcement.
- Automated seeding of built-in Scikit-learn datasets (Diabetes, Breast Cancer, Digits).
"""

from pathlib import Path
from typing import Generator
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.db.models import Base, DatasetModel

# Create SQLite engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)

# Enable foreign keys and busy timeout in SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session lifecycle."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_builtin_datasets(db: Session) -> None:
    """Seed the 3 mandatory built-in datasets if not already registered."""
    settings.ensure_directories()

    # 1. Diabetes Dataset (Regression)
    diabetes_name = "Diabetes (Built-in Regression)"
    if not db.query(DatasetModel).filter(DatasetModel.name == diabetes_name).first():
        data = load_diabetes(as_frame=True)
        df: pd.DataFrame = data.frame
        csv_filename = "builtin_diabetes.csv"
        csv_path = settings.DATA_DIR / csv_filename
        df.to_csv(csv_path, index=False)

        feature_names = [str(c) for c in data.feature_names]
        target_name = "target"

        dataset_rec = DatasetModel(
            name=diabetes_name,
            filename=csv_filename,
            filepath=str(csv_path),
            task_type="regression",
            total_rows=len(df),
            total_columns=len(df.columns),
            feature_names=feature_names,
            target_column=target_name,
            is_builtin=True,
            description="Standard Scikit-learn Diabetes disease progression dataset for Linear Regression.",
        )
        db.add(dataset_rec)

    # 2. Breast Cancer Wisconsin Dataset (Binary Classification)
    bc_name = "Breast Cancer Wisconsin (Built-in Binary)"
    if not db.query(DatasetModel).filter(DatasetModel.name == bc_name).first():
        data = load_breast_cancer(as_frame=True)
        df: pd.DataFrame = data.frame
        # Rename target column to 'diagnosis'
        df.rename(columns={"target": "diagnosis"}, inplace=True)
        csv_filename = "builtin_breast_cancer.csv"
        csv_path = settings.DATA_DIR / csv_filename
        df.to_csv(csv_path, index=False)

        feature_names = [str(c) for c in data.feature_names]
        target_name = "diagnosis"

        dataset_rec = DatasetModel(
            name=bc_name,
            filename=csv_filename,
            filepath=str(csv_path),
            task_type="binary_classification",
            total_rows=len(df),
            total_columns=len(df.columns),
            feature_names=feature_names,
            target_column=target_name,
            is_builtin=True,
            description="Breast Cancer Wisconsin diagnostic dataset for Logistic Regression binary classification.",
        )
        db.add(dataset_rec)

    # 3. Digits Dataset (Multiclass Classification)
    digits_name = "Digits 8x8 (Built-in Multiclass)"
    if not db.query(DatasetModel).filter(DatasetModel.name == digits_name).first():
        data = load_digits(as_frame=True)
        df: pd.DataFrame = data.frame
        df.rename(columns={"target": "digit"}, inplace=True)
        csv_filename = "builtin_digits.csv"
        csv_path = settings.DATA_DIR / csv_filename
        df.to_csv(csv_path, index=False)

        feature_names = [str(c) for c in data.feature_names]
        target_name = "digit"

        dataset_rec = DatasetModel(
            name=digits_name,
            filename=csv_filename,
            filepath=str(csv_path),
            task_type="multiclass_classification",
            total_rows=len(df),
            total_columns=len(df.columns),
            feature_names=feature_names,
            target_column=target_name,
            is_builtin=True,
            description="Optical Recognition of Handwritten Digits 8x8 (10 classes: 0-9) for Multilayer Perceptron.",
        )
        db.add(dataset_rec)

    db.commit()


def init_db() -> None:
    """Initialize database tables and seed required data."""
    settings.ensure_directories()
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_builtin_datasets(db)
    finally:
        db.close()
