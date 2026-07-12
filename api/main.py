"""FastAPI entry point for IntelliFlow."""

from __future__ import annotations

from fastapi import FastAPI

from .routers import analytics, automl


app = FastAPI(
    title="IntelliFlow API",
    version="1.0.0",
    description="Unified API gateway for IntelliFlow engines.",
)

app.include_router(automl.router)
app.include_router(analytics.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic service health check."""

    return {"status": "ok"}

