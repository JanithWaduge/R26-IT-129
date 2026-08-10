from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.curriculum import LocalizedText


class ReviewDirection(str, Enum):
    RECEPTIVE = "receptive"
    PRODUCTIVE = "productive"


class ReviewMode(str, Enum):
    RECEPTIVE = "receptive"
    PRODUCTIVE = "productive"
    MIXED = "mixed"


class DueReviewItemResponse(BaseModel):
    sign_id: str
    code: str
    gloss: str
    meanings: LocalizedText

    direction: ReviewDirection
    difficulty: int

    mastery_score: float = Field(
        ge=0,
        le=1,
    )

    mastery_status: str

    ease_factor: float
    interval_days: int
    repetition_count: int
    lapse_count: int

    next_review_at: datetime | None

    is_due: bool
    days_overdue: float
    priority_score: float
    selection_reason: str

    curriculum_weight: float

    model_config = ConfigDict(
        use_enum_values=True,
    )


class ReviewOverviewResponse(BaseModel):
    total_signs: int

    receptive_due: int
    productive_due: int
    total_due_directions: int

    new_receptive_directions: int
    new_productive_directions: int

    overdue_directions: int

    next_scheduled_review_at: (
        datetime | None
    )

    highest_priority_score: float


class DueReviewListResponse(BaseModel):
    items: list[DueReviewItemResponse]
    total_items: int
    mode: ReviewMode

    model_config = ConfigDict(
        use_enum_values=True,
    )