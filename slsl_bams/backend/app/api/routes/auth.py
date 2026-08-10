from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.security import OAuth2PasswordRequestForm

from app.api.dependencies.auth import (
    CurrentUserDependency,
)
from app.db.dependencies import get_database
from app.repositories.auth_repository import (
    AuthRepository,
)
from app.repositories.student_repository import (
    StudentRepository,
)
from app.schemas.auth import (
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserProfileResponse,
)
from app.services.auth_service import (
    AuthService,
    EmailAlreadyRegisteredError,
    InactiveAccountError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
)
from app.services.audit_service import AuditService

router = APIRouter()


def get_auth_service(
    request: Request,
    database=Depends(get_database),
) -> AuthService:
    return AuthService(
        auth_repository=AuthRepository(database),
        student_repository=StudentRepository(
            database
        ),
        settings=request.app.state.settings,
    )


AuthServiceDependency = Annotated[
    AuthService,
    Depends(get_auth_service),
]


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a student account",
)
async def register(
    registration: RegisterRequest,
    service: AuthServiceDependency,
) -> TokenResponse:
    try:
        return await service.register(
            registration
        )
    except EmailAlreadyRegisteredError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in with email and password",
)
async def login(
    form_data: Annotated[
        OAuth2PasswordRequestForm,
        Depends(),
    ],
    service: AuthServiceDependency,
    request: Request,
    database=Depends(get_database),
) -> TokenResponse:
    try:
        return await service.login(
            email=form_data.username,
            password=form_data.password,
        )
    except InvalidCredentialsError as error:
        await AuditService(
            database=database,
            retention_days=request.app.state.settings.audit_retention_days,
        ).record(
            event_type="authentication_failure", outcome="failure",
            request_id=getattr(request.state, "request_id", None),
            resource_type="account", details={"reason": "invalid_credentials"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from error
    except InactiveAccountError as error:
        await AuditService(
            database=database,
            retention_days=request.app.state.settings.audit_retention_days,
        ).record(
            event_type="authentication_failure", outcome="denied",
            request_id=getattr(request.state, "request_id", None),
            resource_type="account", details={"reason": "inactive_account"},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(error),
        ) from error


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate a refresh token",
)
async def refresh(
    request_data: RefreshRequest,
    service: AuthServiceDependency,
) -> TokenResponse:
    try:
        return await service.refresh(
            request_data.refresh_token
        )
    except InvalidRefreshTokenError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from error


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out the current refresh session",
)
async def logout(
    request_data: LogoutRequest,
    service: AuthServiceDependency,
) -> Response:
    await service.logout(
        request_data.refresh_token
    )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


@router.post(
    "/logout-all",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out from all devices",
)
async def logout_all(
    current_user: CurrentUserDependency,
    service: AuthServiceDependency,
) -> Response:
    await service.logout_all(
        current_user.id
    )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get authenticated student profile",
)
async def get_my_profile(
    current_user: CurrentUserDependency,
    database=Depends(get_database),
) -> UserProfileResponse:
    student = await StudentRepository(
        database
    ).get_by_id(
        current_user.student_id
    )

    return UserProfileResponse(
        user_id=current_user.id,
        student_id=current_user.student_id,
        email=current_user.email,
        role=current_user.role,
        full_name=student.full_name,
        preferred_language=(
            student.preferred_language
        ),
        grade_level=student.grade_level,
        created_at=student.created_at,
    )
