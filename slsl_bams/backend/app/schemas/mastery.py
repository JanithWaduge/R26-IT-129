from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.curriculum import LocalizedText


class MasteryStatus(str, Enum):
    NEW = "new"
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    LEARNING = "learning"
    PROFICIENT = "proficient"
    MASTERED = "mastered"


class MasteryBalanceStatus(str, Enum):
    UNASSESSED = "unassessed"
    RECEPTIVE_UNASSESSED = (
        "receptive_unassessed"
    )
    PRODUCTIVE_UNASSESSED = (
        "productive_unassessed"
    )
    RECEPTIVE_WEAKER = (
        "receptive_weaker"
    )
    PRODUCTIVE_WEAKER = (
        "productive_weaker"
    )
    BALANCED = "balanced"


class DirectionMasteryResponse(BaseModel):
    score: float = Field(
        ge=0,
        le=1,
    )

    status: MasteryStatus

    total_reviews: int
    successful_reviews: int
    independent_successes: int
    assisted_successes: int
    failure_count: int

    average_quality_score: float

    last_quality_score: float | None
    last_quality_band: int | None

    average_response_time_ms: float | None

    average_recognition_confidence: (
        float | None
    )

    last_recognition_confidence: (
        float | None
    )

    last_reviewed_at: datetime | None

    ease_factor: float
    previous_interval_days: int
    interval_days: int
    base_sm2_interval_days: int
    last_ml_probability: float | None
    last_ml_adjustment_factor: float | None
    last_ml_model_version: str | None
    last_ml_status: str
    last_ml_event_id: str | None
    last_ml_applied_at: datetime | None
    repetition_count: int
    lapse_count: int
    next_review_at: datetime | None
    last_scheduled_at: datetime | None
    last_schedule_reason: str
    is_due: bool
    days_overdue: float

    model_config = ConfigDict(
        use_enum_values=True,
    )


class SignMasteryResponse(BaseModel):
    sign_id: str
    code: str
    gloss: str
    meanings: LocalizedText

    category_id: str
    difficulty: int

    receptive: DirectionMasteryResponse
    productive: DirectionMasteryResponse

    combined_score: float
    overall_status: MasteryStatus
    balance_status: MasteryBalanceStatus

    attempted: bool
    updated_at: datetime | None

    model_config = ConfigDict(
        use_enum_values=True,
    )


class PaginatedMasteryResponse(BaseModel):
    items: list[SignMasteryResponse]

    page: int
    page_size: int
    total_items: int
    total_pages: int


class MasteryStatusCounts(BaseModel):
    new: int
    very_weak: int
    weak: int
    learning: int
    proficient: int
    mastered: int


class MasteryOverviewResponse(BaseModel):
    total_signs: int
    attempted_signs: int
    unattempted_signs: int

    receptive_attempted: int
    productive_attempted: int
    bidirectionally_attempted: int

    average_receptive_score: float
    average_productive_score: float
    average_combined_score: float

    statuses: MasteryStatusCounts

    receptive_weaker_count: int
    productive_weaker_count: int
    balanced_count: int
