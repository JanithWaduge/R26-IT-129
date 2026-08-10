from typing import Any

from fastapi import HTTPException, Request, status


def get_database(request: Request) -> Any:
    """Return the MongoDB database attached to the FastAPI application."""

    database = getattr(request.app.state, "database", None)

    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is unavailable.",
        )

    return database