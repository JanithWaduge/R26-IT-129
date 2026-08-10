from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.curriculum import (
    CurriculumCategoryResponse,
    CurriculumCompetencyResponse,
    LocalizedText,
    ValidationStatus,
)


class SignContentStatus(str, Enum):
    DEVELOPMENT = "development"
    APPROVED = "approved"
    ARCHIVED = "archived"


class SignMediaSourceType(str, Enum):
    PENDING = "pending"
    VIDEO = "video"
    AVATAR_ANIMATION = "avatar_animation"


class SignMediaObjectiveSource(str, Enum):
    PENDING = "pending"
    OBJECTIVE_1 = "objective_1"
    TEACHER_UPLOAD = "teacher_upload"


class SignMediaResponse(BaseModel):
    source_type: SignMediaSourceType
    uri: str | None
    thumbnail_uri: str | None
    duration_ms: int | None = Field(
        default=None,
        ge=0,
    )
    objective_source: SignMediaObjectiveSource

    model_config = ConfigDict(
        use_enum_values=True,
    )


class SignSummaryResponse(BaseModel):
    id: str
    code: str
    gloss: str
    meanings: LocalizedText
    category_id: str
    difficulty: int = Field(
        ge=1,
        le=5,
    )
    media: SignMediaResponse
    content_status: SignContentStatus
    validation_status: ValidationStatus

    model_config = ConfigDict(
        use_enum_values=True,
    )


class SignDetailResponse(BaseModel):
    id: str
    code: str
    gloss: str
    meanings: LocalizedText

    category: CurriculumCategoryResponse
    competencies: list[
        CurriculumCompetencyResponse
    ]

    difficulty: int = Field(
        ge=1,
        le=5,
    )

    tags: list[str]
    prerequisites: list[
        SignSummaryResponse
    ]

    media: SignMediaResponse
    content_status: SignContentStatus
    validation_status: ValidationStatus
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        use_enum_values=True,
    )


class PaginatedSignsResponse(BaseModel):
    items: list[SignSummaryResponse]
    page: int
    page_size: int
    total_items: int
    total_pages: int