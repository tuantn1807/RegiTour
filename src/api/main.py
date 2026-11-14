"""
FastAPI application entry point for RegiTour Enterprise REST API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router

app = FastAPI(
    title="RegiTour Enterprise Decision Support API",
    description=(
        "Production-grade RESTful API for destination management, macro stress testing, "
        "capacity/staffing planning, and consistent econometric tourism forecasting (RA-HE-DLinear)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for cross-origin web apps and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 routes
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["System"])
def root_redirect():
    """Root endpoint welcoming developers with link to interactive Swagger docs."""
    return JSONResponse(
        {
            "name": "RegiTour Decision Support REST API",
            "version": "1.0.0",
            "documentation": "/docs",
            "redoc": "/redoc",
            "health": "/api/v1/health",
            "status": "online",
        }
    )

