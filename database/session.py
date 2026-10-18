"""Async database engine and session management."""

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import get_settings


def get_async_database_url() -> URL:
    """Normalize a Neon/PostgreSQL URL for SQLAlchemy's asyncpg driver."""

    settings = get_settings()
    configured_url = settings.database_url

    if configured_url is None or not configured_url.get_secret_value().strip():
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it in your local .env file."
        )

    url = make_url(configured_url.get_secret_value())

    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+asyncpg")
    elif url.drivername != "postgresql+asyncpg":
        raise ValueError(
            "DATABASE_URL must use PostgreSQL or postgresql+asyncpg."
        )

    query = dict(url.query)
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)

    if sslmode == "require":
        query["ssl"] = "require"

    return url.set(query=query)


@lru_cache(maxsize=1)
def get_engine():
    """Create one async engine for the application process."""

    return create_async_engine(
        get_async_database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=300,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create the async session factory."""

    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield a database session and close it after use."""

    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """Close pooled connections during application shutdown."""

    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        get_engine.cache_clear()
        get_session_factory.cache_clear()