from pydantic import BaseModel, Field

from app.schemas.quiz import (
    QuizFeedbackResponse,
    QuizSessionResponse,
    RecognitionResult,
)


class RecognitionCaptureResponse(BaseModel):
    request_id: str

    content_type: str

    byte_size: int = Field(
        ge=1,
    )

    sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    duration_ms: int = Field(
        ge=0,
    )

    retained: bool


class ProductiveRecognitionAnswerResponse(
    BaseModel
):
    session: QuizSessionResponse
    feedback: QuizFeedbackResponse

    recognition: RecognitionResult
    capture: RecognitionCaptureResponse