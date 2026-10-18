
"""Liveness and readiness endpoints."""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from api.dependencies import DatabaseSession

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready")
async def readiness(session: DatabaseSession) -> dict[str, str]:
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "dependencies": {"database": "unavailable"},
            },
        ) from None

    return {
        "status": "ready",
        "dependencies": {"database": "available"},
    }
