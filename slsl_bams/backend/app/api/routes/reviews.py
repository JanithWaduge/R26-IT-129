from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from app.api.dependencies.auth import CurrentUserDependency
from app.db.dependencies import get_database
from app.repositories.mastery_repository import MasteryRepository
from app.repositories.review_repository import ReviewRepository
from app.schemas.review import DueReviewListResponse, ReviewOverviewResponse
from app.services.review_service import ReviewService

router = APIRouter()


def get_review_service(
    request: Request,
    database=Depends(get_database),
) -> ReviewService:
    return ReviewService(
        review_repository=ReviewRepository(database),
        mastery_repository=MasteryRepository(database),
        environment=request.app.state.settings.environment,
    )


ReviewServiceDependency = Annotated[
    ReviewService,
    Depends(get_review_service),
]


@router.get(
    "/overview",
    response_model=ReviewOverviewResponse,
    summary="Get my review overview",
)
async def get_review_overview(
    current_user: CurrentUserDependency,
    service: ReviewServiceDependency,
) -> ReviewOverviewResponse:
    return await service.get_overview(
        student_id=current_user.student_id
    )


@router.get(
    "/due",
    response_model=DueReviewListResponse,
    summary="List my due reviews",
)
async def list_due_reviews(
    current_user: CurrentUserDependency,
    service: ReviewServiceDependency,
    mode: str = Query(
        default="mixed",
        pattern="^(receptive|productive|mixed)$",
    ),
    limit: int = Query(default=20, ge=1, le=100),
) -> DueReviewListResponse:
    return await service.get_due_reviews(
        student_id=current_user.student_id,
        mode=mode,
        limit=limit,
    )
