from datetime import datetime

from pydantic import BaseModel, Field


class GamificationBadgeResponse(
    BaseModel
):
    code: str
    title: str
    description: str
    icon_key: str

    unlocked: bool
    unlocked_at: datetime | None


class GamificationRewardBreakdownResponse(
    BaseModel
):
    question_xp: int
    due_review_bonus: int
    completion_bonus: int
    perfect_bonus: int


class GamificationRewardResponse(
    BaseModel
):
    xp_awarded: int

    total_xp: int
    level: int
    xp_into_level: int
    xp_to_next_level: int

    current_streak_days: int

    badges_awarded: list[
        GamificationBadgeResponse
    ]

    breakdown: (
        GamificationRewardBreakdownResponse
    )


class GamificationProfileResponse(
    BaseModel
):
    total_xp: int
    level: int
    xp_into_level: int
    xp_to_next_level: int

    current_streak_days: int
    longest_streak_days: int
    last_activity_date: str | None

    total_quizzes: int
    total_questions: int
    total_correct: int
    total_incorrect: int
    total_skipped: int
    total_hints: int
    total_retries: int
    perfect_quizzes: int

    accuracy_percentage: float = Field(
        ge=0,
        le=100,
    )

    badges: list[
        GamificationBadgeResponse
    ]


class GamificationHistoryItemResponse(
    BaseModel
):
    session_id: str
    xp_awarded: int

    badges_awarded: list[str]

    total_xp_after: int
    level_after: int
    streak_after: int

    created_at: datetime


class GamificationHistoryResponse(
    BaseModel
):
    items: list[
        GamificationHistoryItemResponse
    ]