from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ValidationStatus(str, Enum):
    PROVISIONAL = "provisional"
    TEACHER_APPROVED = "teacher_approved"


class LocalizedText(BaseModel):
    english: str
    sinhala: str
    tamil: str


class CurriculumCategoryResponse(BaseModel):
    id: str
    code: str
    name: LocalizedText
    description: LocalizedText
    display_order: int
    icon_key: str
    adaptive_priority_weight: float = Field(
        ge=0.50,
        le=2.00,
    )
    validation_status: ValidationStatus
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        use_enum_values=True,
    )


class CurriculumCompetencyResponse(BaseModel):
    id: str
    category_id: str
    code: str
    title: LocalizedText
    description: LocalizedText
    grade_levels: list[str]
    display_order: int

    receptive_mastery_threshold: float = Field(
        ge=0,
        le=1,
    )

    productive_mastery_threshold: float = Field(
        ge=0,
        le=1,
    )

    adaptive_priority_weight: float = Field(
        ge=0.50,
        le=2.00,
    )

    validation_status: ValidationStatus
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        use_enum_values=True,
    )
