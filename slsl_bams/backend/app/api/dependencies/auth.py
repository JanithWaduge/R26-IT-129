from typing import Annotated

from fastapi import (
    Depends,
    HTTPException,
    Request,
    status,
)
from fastapi.security import OAuth2PasswordBearer

from app.core.security import (
    SecurityTokenError,
    decode_token,
)
from app.db.dependencies import get_database
from app.repositories.auth_repository import (
    AuthRepository,
)
from app.schemas.auth import AuthenticatedUser


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)


async def get_current_user(
    request: Request,
    token: Annotated[
        str,
        Depends(oauth2_scheme),
    ],
    database=Depends(get_database),
) -> AuthenticatedUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials are invalid.",
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )

    try:
        payload = decode_token(
            token=token,
            expected_type="access",
            settings=request.app.state.settings,
        )
    except SecurityTokenError as error:
        raise credentials_exception from error

    repository = AuthRepository(database)

    user = await repository.get_user_by_id(
        payload.subject
    )

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is inactive.",
        )

    if (
        payload.token_version
        != user.token_version
    ):
        raise credentials_exception

    return AuthenticatedUser(
        id=user.id,
        student_id=user.student_id,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        token_version=user.token_version,
    )


CurrentUserDependency = Annotated[
    AuthenticatedUser,
    Depends(get_current_user),
]