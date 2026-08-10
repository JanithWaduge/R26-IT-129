from pathlib import Path

from app.recognition.base import (
    RecognitionOutput,
    RecognitionPredictionOutput,
)


class MockSignRecognizer:
    """
    Development-only recognizer.

    It returns the expected sign so the complete
    Objective 3 integration can be tested before the
    Objective 2 service is available.

    Never use this adapter for model evaluation.
    """

    async def recognize(
        self,
        *,
        video_path: Path,
        content_type: str,
        request_id: str,
        expected_sign_id: str,
        expected_code: str,
    ) -> RecognitionOutput:
        del content_type

        byte_size = video_path.stat().st_size

        confidence = (
            0.92
            if byte_size > 0
            else 0.0
        )

        return RecognitionOutput(
            request_id=request_id,
            model_name=(
                "mock-recognizer-do-not-evaluate"
            ),
            model_version="0.1.0",
            predicted_sign_id=(
                expected_sign_id
                if byte_size > 0
                else None
            ),
            predicted_code=(
                expected_code
                if byte_size > 0
                else None
            ),
            confidence=confidence,
            inference_time_ms=5,
            top_predictions=[
                RecognitionPredictionOutput(
                    sign_id=expected_sign_id,
                    sign_code=expected_code,
                    confidence=confidence,
                )
            ],
        )