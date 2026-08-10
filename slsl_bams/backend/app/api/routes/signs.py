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
from app.repositories.sign_repository import (
    SignNotFoundError,
    SignRepository,
)
from app.schemas.sign import (
    PaginatedSignsResponse,
    SignDetailResponse,
)
from app.services.sign_service import (
    SignService,
)

router = APIRouter()


def get_sign_service(
    request: Request,
    database=Depends(get_database),
) -> SignService:
    return SignService(
        repository=SignRepository(database),
        settings=request.app.state.settings,
    )


SignServiceDependency = Annotated[
    SignService,
    Depends(get_sign_service),
]


@router.get(
    "",
    response_model=PaginatedSignsResponse,
    summary="Search and filter signs",
)
async def list_signs(
    current_user: CurrentUserDependency,
    service: SignServiceDependency,
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
    competency_id: str | None = Query(
        default=None
    ),
    difficulty: int | None = Query(
        default=None,
        ge=1,
        le=5,
    ),
    search: str | None = Query(
        default=None,
        max_length=100,
    ),
) -> PaginatedSignsResponse:
    del current_user

    try:
        return await service.list_signs(
            page=page,
            page_size=page_size,
            category_id=category_id,
            competency_id=competency_id,
            difficulty=difficulty,
            search=search,
        )
    except SignNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(error),
        ) from error


@router.get(
    "/{sign_id}",
    response_model=SignDetailResponse,
    summary="Get complete sign details",
)
async def get_sign(
    sign_id: str,
    current_user: CurrentUserDependency,
    service: SignServiceDependency,
) -> SignDetailResponse:
    del current_user

    try:
        return await service.get_sign(
            sign_id
        )
    except SignNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(error),
        ) from error