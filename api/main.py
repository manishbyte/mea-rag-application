
"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from core.config import get_settings
from core.logging_config import configure_logging
from database.session import dispose_engine


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure logging and release database resources on shutdown."""

    configure_logging(settings.log_level)
    try:
        yield
    finally:
        await dispose_engine()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Authenticated question-answering API for the "
        "MEA Annual Report 2025."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(health_router)


@app.get("/", tags=["General"])
async def root() -> dict[str, str]:
    return {
        "application": settings.app_name,
        "status": "running",
    }
