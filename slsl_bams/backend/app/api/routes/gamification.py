from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.dependencies.auth import CurrentUserDependency
from app.db.dependencies import get_database
from app.repositories.gamification_repository import GamificationRepository
from app.schemas.gamification import (
    GamificationHistoryResponse,
    GamificationProfileResponse,
)
from app.services.gamification_service import GamificationService

router = APIRouter()


def get_gamification_service(
    request: Request,
    database=Depends(get_database),
) -> GamificationService:
    return GamificationService(
        repository=GamificationRepository(database),
        settings=request.app.state.settings,
    )


GamificationServiceDependency = Annotated[
    GamificationService,
    Depends(get_gamification_service),
]


@router.get(
    "/profile",
    response_model=GamificationProfileResponse,
    summary="Get my gamification profile",
)
async def get_my_gamification_profile(
    current_user: CurrentUserDependency,
    service: GamificationServiceDependency,
) -> GamificationProfileResponse:
    return await service.get_profile(
        student_id=current_user.student_id,
    )


@router.get(
    "/history",
    response_model=GamificationHistoryResponse,
    summary="Get my quiz reward history",
)
async def get_my_reward_history(
    current_user: CurrentUserDependency,
    service: GamificationServiceDependency,
    limit: int = Query(default=20, ge=1, le=100),
) -> GamificationHistoryResponse:
    return await service.get_history(
        student_id=current_user.student_id,
        limit=limit,
    )
