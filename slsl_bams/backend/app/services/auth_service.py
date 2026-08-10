from app.core.config import Settings
from app.core.security import (
    SecurityTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token_identifier,
    verify_password,
)
from app.repositories.auth_repository import (
    AuthRepository,
    DuplicateUserEmailError,
)
from app.repositories.student_repository import (
    DuplicateStudentEmailError,
    StudentRepository,
)
from app.schemas.auth import (
    AuthUserRecord,
    RegisterRequest,
    TokenResponse,
)
from app.schemas.student import StudentCreate


class EmailAlreadyRegisteredError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class InvalidRefreshTokenError(Exception):
    pass


class InactiveAccountError(Exception):
    pass


class AuthService:
    def __init__(
        self,
        *,
        auth_repository: AuthRepository,
        student_repository: StudentRepository,
        settings: Settings,
    ) -> None:
        self.auth_repository = auth_repository
        self.student_repository = student_repository
        self.settings = settings

    async def register(
        self,
        registration: RegisterRequest,
    ) -> TokenResponse:
        existing_user = (
            await self.auth_repository.get_user_by_email(
                str(registration.email)
            )
        )

        if existing_user is not None:
            raise EmailAlreadyRegisteredError(
                "An account with this email already exists."
            )

        try:
            student = await self.student_repository.create(
                StudentCreate(
                    full_name=registration.full_name,
                    email=registration.email,
                    preferred_language=(
                        registration.preferred_language
                    ),
                    grade_level=registration.grade_level,
                )
            )
        except DuplicateStudentEmailError as error:
            raise EmailAlreadyRegisteredError(
                "An account with this email already exists."
            ) from error

        try:
            user = await self.auth_repository.create_user(
                student_id=student.id,
                email=str(registration.email),
                password_hash=hash_password(
                    registration.password
                ),
            )
        except DuplicateUserEmailError as error:
            await (
                self.student_repository
                .hard_delete_for_rollback(
                    student.id
                )
            )

            raise EmailAlreadyRegisteredError(
                "An account with this email already exists."
            ) from error
        except Exception:
            await (
                self.student_repository
                .hard_delete_for_rollback(
                    student.id
                )
            )
            raise

        return await self._issue_token_pair(user)

    async def login(
        self,
        *,
        email: str,
        password: str,
    ) -> TokenResponse:
        user = (
            await self.auth_repository.get_user_by_email(
                email
            )
        )

        if user is None:
            raise InvalidCredentialsError(
                "Invalid email or password."
            )

        if not verify_password(
            password,
            user.password_hash,
        ):
            raise InvalidCredentialsError(
                "Invalid email or password."
            )

        if not user.is_active:
            raise InactiveAccountError(
                "This account is inactive."
            )

        return await self._issue_token_pair(user)

    async def refresh(
        self,
        refresh_token: str,
    ) -> TokenResponse:
        try:
            payload = decode_token(
                token=refresh_token,
                expected_type="refresh",
                settings=self.settings,
            )
        except SecurityTokenError as error:
            raise InvalidRefreshTokenError(
                "Refresh token is invalid or expired."
            ) from error

        if payload.family_id is None:
            raise InvalidRefreshTokenError(
                "Refresh token family is missing."
            )

        user = (
            await self.auth_repository.get_user_by_id(
                payload.subject
            )
        )

        if user is None or not user.is_active:
            raise InvalidRefreshTokenError(
                "Refresh token is invalid."
            )

        if (
            payload.token_version
            != user.token_version
        ):
            raise InvalidRefreshTokenError(
                "Refresh token has been invalidated."
            )

        next_refresh = create_refresh_token(
            settings=self.settings,
            user_id=user.id,
            token_version=user.token_version,
            family_id=payload.family_id,
        )

        old_jti_hash = hash_token_identifier(
            payload.token_id
        )

        consumed = (
            await self.auth_repository
            .consume_refresh_session(
                user_id=user.id,
                jti_hash=old_jti_hash,
                replacement_jti_hash=(
                    next_refresh.token_id_hash
                ),
            )
        )

        if not consumed:
            await (
                self.auth_repository
                .revoke_token_family(
                    user_id=user.id,
                    family_id=payload.family_id,
                )
            )

            await (
                self.auth_repository
                .increment_token_version(
                    user_id=user.id
                )
            )

            raise InvalidRefreshTokenError(
                "Refresh-token reuse was detected. "
                "All sessions in this token family "
                "have been revoked."
            )

        await self.auth_repository.create_refresh_session(
            user_id=user.id,
            jti_hash=next_refresh.token_id_hash,
            family_id=next_refresh.family_id,
            expires_at=next_refresh.expires_at,
        )

        access_token = create_access_token(
            settings=self.settings,
            user_id=user.id,
            role=str(user.role),
            token_version=user.token_version,
        )

        return TokenResponse(
            access_token=access_token.token,
            refresh_token=next_refresh.token,
            expires_in=(
                self.settings
                .access_token_expire_minutes
                * 60
            ),
        )

    async def logout(
        self,
        refresh_token: str,
    ) -> None:
        try:
            payload = decode_token(
                token=refresh_token,
                expected_type="refresh",
                settings=self.settings,
            )
        except SecurityTokenError:
            return

        await (
            self.auth_repository
            .revoke_refresh_session(
                jti_hash=hash_token_identifier(
                    payload.token_id
                )
            )
        )

    async def logout_all(
        self,
        user_id: str,
    ) -> None:
        await self.auth_repository.revoke_all_sessions(
            user_id=user_id
        )

        await (
            self.auth_repository
            .increment_token_version(
                user_id=user_id
            )
        )

    async def _issue_token_pair(
        self,
        user: AuthUserRecord,
    ) -> TokenResponse:
        access_token = create_access_token(
            settings=self.settings,
            user_id=user.id,
            role=str(user.role),
            token_version=user.token_version,
        )

        refresh_token = create_refresh_token(
            settings=self.settings,
            user_id=user.id,
            token_version=user.token_version,
        )

        if refresh_token.family_id is None:
            raise RuntimeError(
                "Refresh-token family was not created."
            )

        await self.auth_repository.create_refresh_session(
            user_id=user.id,
            jti_hash=refresh_token.token_id_hash,
            family_id=refresh_token.family_id,
            expires_at=refresh_token.expires_at,
        )

        return TokenResponse(
            access_token=access_token.token,
            refresh_token=refresh_token.token,
            expires_in=(
                self.settings
                .access_token_expire_minutes
                * 60
            ),
        )