from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.api.dependencies.auth import (
    CurrentUserDependency,
)
from app.db.dependencies import get_database
from app.repositories.student_repository import (
    StudentNotFoundError,
    StudentRepository,
)
from app.schemas.student import (
    StudentProfileUpdate,
    StudentResponse,
    StudentUpdate,
)

router = APIRouter()


@router.get(
    "/me",
    response_model=StudentResponse,
    summary="Get my student profile",
)
async def get_my_student_profile(
    current_user: CurrentUserDependency,
    database=Depends(get_database),
) -> StudentResponse:
    try:
        return await StudentRepository(
            database
        ).get_by_id(
            current_user.student_id
        )
    except StudentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.patch(
    "/me",
    response_model=StudentResponse,
    summary="Update my student profile",
)
async def update_my_student_profile(
    profile_data: StudentProfileUpdate,
    current_user: CurrentUserDependency,
    database=Depends(get_database),
) -> StudentResponse:
    try:
        return await StudentRepository(
            database
        ).update(
            student_id=current_user.student_id,
            student_data=StudentUpdate(
                **profile_data.model_dump(
                    exclude_none=True
                )
            ),
        )
    except StudentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error