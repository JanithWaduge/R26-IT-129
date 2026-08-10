from pathlib import Path

import httpx
from pydantic import ValidationError

from app.recognition.base import (
    InvalidRecognizerResponseError,
    RecognitionOutput,
    RecognitionPredictionOutput,
    RecognizerUnavailableError,
)
from app.schemas.quiz import RecognitionResult


class HttpSignRecognizer:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
    ) -> None:
        self.endpoint = (
            f"{base_url.rstrip('/')}/recognize"
        )

        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def recognize(
        self,
        *,
        video_path: Path,
        content_type: str,
        request_id: str,
        expected_sign_id: str,
        expected_code: str,
    ) -> RecognitionOutput:
        # The expected answer is deliberately not
        # transmitted to Objective 2.
        del expected_sign_id
        del expected_code

        headers: dict[str, str] = {}

        if self.api_key:
            headers["X-API-Key"] = (
                self.api_key
            )

        try:
            with video_path.open("rb") as video_file:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds
                ) as client:
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        data={
                            "request_id": request_id,
                        },
                        files={
                            "video": (
                                video_path.name,
                                video_file,
                                content_type,
                            )
                        },
                    )

            response.raise_for_status()

        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ) as error:
            raise RecognizerUnavailableError(
                "The Objective 2 recognition "
                "service is unavailable."
            ) from error

        try:
            validated = (
                RecognitionResult.model_validate(
                    response.json()
                )
            )
        except (
            ValueError,
            ValidationError,
        ) as error:
            raise InvalidRecognizerResponseError(
                "Objective 2 returned an invalid "
                "recognition response."
            ) from error

        if str(validated.request_id) != request_id:
            raise InvalidRecognizerResponseError(
                "Objective 2 returned a mismatched "
                "request identifier."
            )

        return RecognitionOutput(
            request_id=request_id,
            model_name=validated.model_name,
            model_version=(
                validated.model_version
            ),
            predicted_sign_id=(
                validated.predicted_sign_id
            ),
            predicted_code=(
                validated.predicted_code
            ),
            confidence=float(
                validated.confidence
            ),
            inference_time_ms=(
                validated.inference_time_ms
            ),
            top_predictions=[
                RecognitionPredictionOutput(
                    sign_id=prediction.sign_id,
                    sign_code=(
                        prediction.sign_code
                    ),
                    confidence=float(
                        prediction.confidence
                    ),
                )
                for prediction
                in validated.top_predictions
            ],
        )