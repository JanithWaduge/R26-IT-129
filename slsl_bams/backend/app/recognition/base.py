from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class RecognizerUnavailableError(Exception):
    pass


class InvalidRecognizerResponseError(Exception):
    pass


@dataclass(frozen=True)
class RecognitionPredictionOutput:
    sign_id: str | None
    sign_code: str | None
    confidence: float


@dataclass(frozen=True)
class RecognitionOutput:
    request_id: str

    model_name: str
    model_version: str

    predicted_sign_id: str | None
    predicted_code: str | None

    confidence: float
    inference_time_ms: int

    top_predictions: list[
        RecognitionPredictionOutput
    ]


class SignRecognizer(Protocol):
    async def recognize(
        self,
        *,
        video_path: Path,
        content_type: str,
        request_id: str,
        expected_sign_id: str,
        expected_code: str,
    ) -> RecognitionOutput:
        ...