from datetime import datetime
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
)

from app.schemas.student import PreferredLanguage


class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"


class RegisterRequest(BaseModel):
    full_name: str = Field(
        min_length=2,
        max_length=120,
    )

    email: EmailStr

    password: str = Field(
        min_length=10,
        max_length=128,
    )

    preferred_language: PreferredLanguage

    grade_level: str = Field(
        min_length=1,
        max_length=50,
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    @field_validator("password")
    @classmethod
    def validate_password(
        cls,
        value: str,
    ) -> str:
        has_letter = any(
            character.isalpha()
            for character in value
        )

        has_number = any(
            character.isdigit()
            for character in value
        )

        if not has_letter or not has_number:
            raise ValueError(
                "Password must contain at least "
                "one letter and one number."
            )

        return value


class RefreshRequest(BaseModel):
    refresh_token: str = Field(
        min_length=20,
    )


class LogoutRequest(BaseModel):
    refresh_token: str = Field(
        min_length=20,
    )


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class AuthenticatedUser(BaseModel):
    id: str
    student_id: str
    email: EmailStr
    role: UserRole
    is_active: bool
    token_version: int

    model_config = ConfigDict(
        use_enum_values=True,
    )


class UserProfileResponse(BaseModel):
    user_id: str
    student_id: str
    email: EmailStr
    role: UserRole

    full_name: str
    preferred_language: PreferredLanguage
    grade_level: str

    created_at: datetime

    model_config = ConfigDict(
        use_enum_values=True,
    )


class TokenPayload(BaseModel):
    subject: str
    token_type: str
    token_id: str
    token_version: int
    role: str | None = None
    family_id: str | None = None
    expires_at: datetime


class AuthUserRecord(BaseModel):
    id: str
    student_id: str
    email: EmailStr
    password_hash: str
    role: UserRole
    is_active: bool
    token_version: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        use_enum_values=True,
    )