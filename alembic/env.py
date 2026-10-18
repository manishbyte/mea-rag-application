"""Alembic migration environment."""

import asyncio


from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from core.config import get_settings
from database.base import Base
from database.session import get_async_database_url
from database import models  # noqa: F401 - register model metadata

config = context.config

if config.config_file_name is not None:
    from pathlib import Path

    

target_metadata = Base.metadata


def database_url_for_alembic() -> str:
    """Get the normalized async URL without logging credentials."""

    return get_async_database_url().render_as_string(
        hide_password=False
    )


def run_migrations_offline() -> None:
    """Run migrations without opening a database connection."""

    context.configure(
        url=database_url_for_alembic(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using SQLAlchemy's async engine."""

    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = database_url_for_alembic()

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()