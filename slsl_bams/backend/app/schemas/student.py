from datetime import datetime
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


class PreferredLanguage(str, Enum):
    SINHALA = "sinhala"
    TAMIL = "tamil"
    ENGLISH = "english"


class StudentCreate(BaseModel):
    full_name: str = Field(
        min_length=2,
        max_length=120,
    )

    email: EmailStr

    preferred_language: PreferredLanguage

    grade_level: str = Field(
        min_length=1,
        max_length=50,
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    @field_validator("full_name", "grade_level")
    @classmethod
    def reject_empty_whitespace(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Value cannot be empty.")

        return cleaned_value


class StudentUpdate(BaseModel):
    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=120,
    )

    email: EmailStr | None = None

    preferred_language: PreferredLanguage | None = None

    grade_level: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )

    is_active: bool | None = None

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    @model_validator(mode="after")
    def require_at_least_one_field(self) -> "StudentUpdate":
        values = self.model_dump(exclude_none=True)

        if not values:
            raise ValueError(
                "At least one student field must be supplied."
            )

        return self


class StudentResponse(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    preferred_language: PreferredLanguage
    grade_level: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        use_enum_values=True,
    )


class StudentProfileUpdate(BaseModel):
    full_name: str | None = Field(
        default=None,
        min_length=2,
        max_length=120,
    )

    preferred_language: PreferredLanguage | None = None

    grade_level: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=True,
    )

    @model_validator(mode="after")
    def require_profile_field(
        self,
    ) -> "StudentProfileUpdate":
        if not self.model_dump(exclude_none=True):
            raise ValueError(
                "At least one profile field must be supplied."
            )

        return self