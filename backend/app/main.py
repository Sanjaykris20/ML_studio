"""FastAPI Main Application Entrypoint for ML Studio.

Self-configuring server with:
- Automated database table initialization & dataset seeding via lifespan handler.
- Configurable CORS for seamless local frontend development.
- Modular REST routers for datasets, models, experiments, and diagnostics.
- Clear error handling.
"""

import faulthandler
faulthandler.enable()

try:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
except Exception:
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import init_db
from app.api import datasets, models, experiments, health, pipeline



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler: initializes SQLite tables & seeds datasets on startup."""
    print("🚀 Initializing ML Studio database and built-in datasets...")
    init_db()
    print("✅ ML Studio Database and artifacts ready.")
    yield
    print("🛑 ML Studio backend shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Interactive Machine Learning Platform REST API for dataset management, scratch vs. library training, real-time prediction, and analytics.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from fastapi.exceptions import HTTPException as FastHTTPException
    if isinstance(exc, FastHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected server error occurred: {str(exc)}"},
    )

# Include Routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(datasets.router, prefix=settings.API_V1_STR)
app.include_router(models.router, prefix=settings.API_V1_STR)
app.include_router(experiments.router, prefix=settings.API_V1_STR)
app.include_router(pipeline.router, prefix=settings.API_V1_STR)


# Mount frontend production build if available
frontend_dist = settings.BASE_DIR / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from fastapi import HTTPException

    assets_path = frontend_dist / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API or Swagger documentation routes
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("redoc") or full_path == "openapi.json":
            raise HTTPException(status_code=404, detail="Not Found")
        target_file = frontend_dist / full_path
        if target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(frontend_dist / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "message": "Welcome to ML Studio API",
            "docs": "/docs",
            "health": f"{settings.API_V1_STR}/health",
            "status": "operational",
        }
