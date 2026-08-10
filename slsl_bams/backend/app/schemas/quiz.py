from datetime import datetime
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    UUID4,
    model_validator,
)

from app.schemas.curriculum import LocalizedText
from app.schemas.sign import SignMediaResponse
from app.schemas.student import PreferredLanguage

from app.schemas.gamification import (
    GamificationRewardResponse,
)


class QuizMode(str, Enum):
    RECEPTIVE = "receptive"
    PRODUCTIVE = "productive"
    MIXED = "mixed"


class QuizDirection(str, Enum):
    RECEPTIVE = "receptive"
    PRODUCTIVE = "productive"


class QuizSessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    EXPIRED = "expired"

class QuizSelectionStrategy(str, Enum):
    ADAPTIVE = "adaptive"
    DUE_ONLY = "due_only"
    RANDOM_CONTROL = "random_control"

    # Existing sessions created before adaptive
    # selection was introduced.
    LEGACY_RANDOM = "legacy_random"


class QuizSessionCreate(BaseModel):
    client_request_id: UUID4

    mode: QuizMode = QuizMode.RECEPTIVE

    prompt_language: PreferredLanguage | None = None

    question_count: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    category_id: str | None = None
    competency_id: str | None = None

    difficulty: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    model_config = ConfigDict(
        use_enum_values=True,
    )

    selection_strategy: QuizSelectionStrategy = (
        QuizSelectionStrategy.ADAPTIVE
    )


class RecognitionTopPrediction(BaseModel):
    sign_id: str | None = None
    sign_code: str | None = None

    confidence: float = Field(
        ge=0,
        le=1,
    )


class RecognitionResult(BaseModel):
    request_id: UUID4

    model_name: str = Field(
        min_length=1,
        max_length=100,
    )

    model_version: str = Field(
        min_length=1,
        max_length=50,
    )

    predicted_sign_id: str | None = None
    predicted_code: str | None = None

    confidence: float = Field(
        ge=0,
        le=1,
    )

    inference_time_ms: int = Field(
        ge=0,
        le=60000,
    )

    top_predictions: list[
        RecognitionTopPrediction
    ] = Field(
        default_factory=list,
        max_length=5,
    )


class QuizAnswerRequest(BaseModel):
    submission_id: UUID4
    question_id: UUID4
    direction: QuizDirection

    selected_sign_id: str | None = None
    recognition_result: RecognitionResult | None = None

    client_response_time_ms: int = Field(
        ge=0,
        le=600000,
    )

    model_config = ConfigDict(
        use_enum_values=True,
    )

    @model_validator(mode="after")
    def validate_answer_type(
        self,
    ) -> "QuizAnswerRequest":
        if self.direction == QuizDirection.RECEPTIVE:
            if self.selected_sign_id is None:
                raise ValueError(
                    "selected_sign_id is required "
                    "for a receptive answer."
                )

            if self.recognition_result is not None:
                raise ValueError(
                    "recognition_result cannot be supplied "
                    "for a receptive answer."
                )

        if self.direction == QuizDirection.PRODUCTIVE:
            if self.recognition_result is None:
                raise ValueError(
                    "recognition_result is required "
                    "for a productive answer."
                )

            if self.selected_sign_id is not None:
                raise ValueError(
                    "selected_sign_id cannot be supplied "
                    "for a productive answer."
                )

        return self


class QuizActionRequest(BaseModel):
    action_id: UUID4
    question_id: UUID4


class QuizOptionResponse(BaseModel):
    sign_id: str
    text: str


class QuizCurrentQuestionResponse(BaseModel):
    question_id: str
    direction: QuizDirection
    order_index: int

    prompt_text: str | None
    media: SignMediaResponse | None

    category_name: str
    difficulty: int

    options: list[QuizOptionResponse]

    attempt_number: int
    remaining_attempts: int
    hint_used: bool

    started_at: datetime

    model_config = ConfigDict(
        use_enum_values=True,
    )

    selection_priority: float
    selection_reason: str
    was_due: bool
    days_overdue: float


class QuizSummaryResponse(BaseModel):
    correct_count: int
    incorrect_count: int
    skipped_count: int
    hints_used: int
    total_retries: int
    receptive_correct: int
    productive_correct: int


class QuizSessionResponse(BaseModel):
    id: str

    mode: QuizMode
    prompt_language: PreferredLanguage
    status: QuizSessionStatus

    question_count: int
    current_question_index: int
    answered_count: int
    progress_percentage: float

    summary: QuizSummaryResponse
    current_question: QuizCurrentQuestionResponse | None

    started_at: datetime
    expires_at: datetime
    completed_at: datetime | None

    version: int

    model_config = ConfigDict(
        use_enum_values=True,
    )

    selection_strategy: QuizSelectionStrategy
    experiment_name: str | None
    experiment_arm: str


class QuizFeedbackResponse(BaseModel):
    action: str
    is_correct: bool | None
    question_finalized: bool
    retry_allowed: bool
    message: str
    recognition_confidence: float | None = None


class QuizActionResponse(BaseModel):
    session: QuizSessionResponse
    feedback: QuizFeedbackResponse

    reward: (
        GamificationRewardResponse
        | None
    ) = None


class QuizHintResponse(BaseModel):
    hint_type: str
    message: str
    hint_count: int
    session: QuizSessionResponse
