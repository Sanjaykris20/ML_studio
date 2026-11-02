"""Application Configuration and Directory Management.

Self-configuring environment with sensible zero-setup defaults for Windows local development.
Does not require external services or third-party settings packages.
"""

import os
from pathlib import Path
from typing import List


class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "ML Studio")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # Base Paths
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent
    BASE_DIR: Path = BACKEND_DIR.parent

    # Storage paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BACKEND_DIR / "artifacts" / "datasets")))
    ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", str(BACKEND_DIR / "artifacts" / "models")))
    DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", str(BACKEND_DIR / "ml_studio.db")))

    # Database URL: default SQLite with foreign keys enabled
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")

    # File uploads
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    ALLOWED_EXTENSIONS: List[str] = [".csv"]

    # Security & CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ]

    def ensure_directories(self) -> None:
        """Automatically create storage and database directories if they do not exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
