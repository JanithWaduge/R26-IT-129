from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from app.api.dependencies.auth import (
    CurrentUserDependency,
)
from app.db.dependencies import get_database
from app.repositories.mastery_repository import (
    MasteryRecordNotFoundError,
    MasteryRepository,
)
from app.schemas.mastery import (
    MasteryOverviewResponse,
    PaginatedMasteryResponse,
    SignMasteryResponse,
)
from app.services.mastery_service import (
    MasteryService,
)

router = APIRouter()


def get_mastery_service(
    request: Request,
    database=Depends(get_database),
) -> MasteryService:
    return MasteryService(
        repository=MasteryRepository(
            database
        ),
        settings=request.app.state.settings,
    )


MasteryServiceDependency = Annotated[
    MasteryService,
    Depends(get_mastery_service),
]


@router.get(
    "/overview",
    response_model=MasteryOverviewResponse,
    summary="Get my mastery overview",
)
async def get_mastery_overview(
    current_user: CurrentUserDependency,
    service: MasteryServiceDependency,
) -> MasteryOverviewResponse:
    return await service.get_overview(
        student_id=current_user.student_id
    )


@router.get(
    "/signs",
    response_model=PaginatedMasteryResponse,
    summary="List my sign mastery states",
)
async def list_sign_mastery(
    current_user: CurrentUserDependency,
    service: MasteryServiceDependency,
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    category_id: str | None = Query(
        default=None
    ),
    mastery_status: str | None = Query(
        default=None,
        pattern=(
            "^(new|very_weak|weak|learning|"
            "proficient|mastered)$"
        ),
    ),
) -> PaginatedMasteryResponse:
    try:
        return await service.list_sign_mastery(
            student_id=(
                current_user.student_id
            ),
            page=page,
            page_size=page_size,
            category_id=category_id,
            status_filter=mastery_status,
        )
    except MasteryRecordNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(error),
        ) from error


@router.get(
    "/signs/{sign_id}",
    response_model=SignMasteryResponse,
    summary="Get my mastery for one sign",
)
async def get_one_sign_mastery(
    sign_id: str,
    current_user: CurrentUserDependency,
    service: MasteryServiceDependency,
) -> SignMasteryResponse:
    try:
        return await service.get_sign_mastery(
            student_id=(
                current_user.student_id
            ),
            sign_id=sign_id,
        )
    except MasteryRecordNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(error),
        ) from error