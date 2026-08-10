from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from app.api.dependencies.auth import (
    CurrentUserDependency,
)
from app.db.dependencies import get_database
from app.repositories.curriculum_repository import (
    CurriculumCategoryNotFoundError,
    CurriculumRepository,
)
from app.schemas.curriculum import (
    CurriculumCategoryResponse,
    CurriculumCompetencyResponse,
)
from app.services.curriculum_service import (
    CurriculumService,
)

router = APIRouter()


def get_curriculum_service(
    database=Depends(get_database),
) -> CurriculumService:
    return CurriculumService(
        CurriculumRepository(database)
    )


CurriculumServiceDependency = Annotated[
    CurriculumService,
    Depends(get_curriculum_service),
]


@router.get(
    "/categories",
    response_model=list[
        CurriculumCategoryResponse
    ],
    summary="List curriculum categories",
)
async def list_categories(
    current_user: CurrentUserDependency,
    service: CurriculumServiceDependency,
) -> list[
    CurriculumCategoryResponse
]:
    del current_user

    return await service.list_categories()


@router.get(
    "/categories/{category_id}",
    response_model=(
        CurriculumCategoryResponse
    ),
    summary="Get one curriculum category",
)
async def get_category(
    category_id: str,
    current_user: CurrentUserDependency,
    service: CurriculumServiceDependency,
) -> CurriculumCategoryResponse:
    del current_user

    try:
        return await service.get_category(
            category_id
        )
    except (
        CurriculumCategoryNotFoundError
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(error),
        ) from error


@router.get(
    "/competencies",
    response_model=list[
        CurriculumCompetencyResponse
    ],
    summary="List curriculum competencies",
)
async def list_competencies(
    current_user: CurrentUserDependency,
    service: CurriculumServiceDependency,
    category_id: str | None = Query(
        default=None
    ),
) -> list[
    CurriculumCompetencyResponse
]:
    del current_user

    try:
        return await service.list_competencies(
            category_id
        )
    except (
        CurriculumCategoryNotFoundError
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(error),
        ) from error