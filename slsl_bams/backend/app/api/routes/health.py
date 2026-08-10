from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Check API and database health",
)
async def check_health(
    request: Request,
) -> HealthResponse:
    settings = request.app.state.settings
    database = request.app.state.database

    await database.command("ping")

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        database="connected",
        timestamp=datetime.now(timezone.utc),
    )