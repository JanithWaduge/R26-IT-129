from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from bson import ObjectId
from fastapi import UploadFile

from app.core.config import Settings
from app.recognition.base import (
    InvalidRecognizerResponseError,
    SignRecognizer,
)
from app.repositories.quiz_repository import (
    QuizRepository,
)
from app.repositories.recognition_repository import (
    RecognitionRepository,
)
from app.schemas.auth import AuthenticatedUser
from app.schemas.quiz import (
    QuizAnswerRequest,
    QuizFeedbackResponse,
    RecognitionResult,
    RecognitionTopPrediction,
)
from app.schemas.recognition import (
    ProductiveRecognitionAnswerResponse,
    RecognitionCaptureResponse,
)
from app.services.quiz_service import (
    QuizService,
)


class UnsupportedRecognitionMediaError(
    Exception
):
    pass


class RecognitionFileTooLargeError(
    Exception
):
    pass


class RecognitionDurationError(Exception):
    pass


class ProductiveQuestionRequiredError(
    Exception
):
    pass


class RecognitionService:
    ALLOWED_CONTENT_TYPES = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
    }

    def __init__(
        self,
        *,
        repository: RecognitionRepository,
        quiz_repository: QuizRepository,
        quiz_service: QuizService,
        recognizer: SignRecognizer,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.quiz_repository = (
            quiz_repository
        )
        self.quiz_service = quiz_service
        self.recognizer = recognizer
        self.settings = settings

    async def recognize_and_answer(
        self,
        *,
        current_user: AuthenticatedUser,
        session_id: str,
        request_id: UUID,
        question_id: UUID,
        client_response_time_ms: int,
        duration_ms: int,
        video: UploadFile,
    ) -> ProductiveRecognitionAnswerResponse:
        request_id_text = str(request_id)
        question_id_text = str(question_id)

        existing = (
            await self.repository
            .get_by_request(
                student_id=(
                    current_user.student_id
                ),
                request_id=(
                    request_id_text
                ),
            )
        )

        if existing is not None:
            return await (
                self._existing_response(
                    existing=existing,
                    current_user=(
                        current_user
                    ),
                    session_id=session_id,
                )
            )

        session_document = (
            await self.quiz_repository
            .get_owned_session(
                session_id=session_id,
                student_id=(
                    current_user.student_id
                ),
            )
        )

        if (
            session_document["status"]
            != "in_progress"
        ):
            raise ProductiveQuestionRequiredError(
                "The quiz session is no longer "
                "in progress."
            )

        current_index = session_document[
            "current_question_index"
        ]

        if current_index >= len(
            session_document["questions"]
        ):
            raise ProductiveQuestionRequiredError(
                "The quiz has no current question."
            )

        question = session_document[
            "questions"
        ][current_index]

        if (
            question["question_id"]
            != question_id_text
        ):
            raise ProductiveQuestionRequiredError(
                "The submitted question is not "
                "the current quiz question."
            )

        if (
            question["direction"]
            != "productive"
        ):
            raise ProductiveQuestionRequiredError(
                "Video recognition can only be "
                "used for productive questions."
            )

        self._validate_duration(
            duration_ms
        )

        temporary_path: Path | None = None
        keep_file = False

        try:
            (
                temporary_path,
                byte_size,
                file_hash,
                content_type,
            ) = await self._save_upload(
                video=video,
                request_id=(
                    request_id_text
                ),
            )

            output = await (
                self.recognizer.recognize(
                    video_path=temporary_path,
                    content_type=content_type,
                    request_id=(
                        request_id_text
                    ),
                    expected_sign_id=str(
                        question["sign_id"]
                    ),
                    expected_code=question[
                        "expected_code"
                    ],
                )
            )

            if (
                output.request_id
                != request_id_text
            ):
                raise (
                    InvalidRecognizerResponseError(
                        "Recognizer request ID "
                        "does not match."
                    )
                )

            recognition_result = (
                RecognitionResult(
                    request_id=(
                        request_id_text
                    ),
                    model_name=(
                        output.model_name
                    ),
                    model_version=(
                        output.model_version
                    ),
                    predicted_sign_id=(
                        output
                        .predicted_sign_id
                    ),
                    predicted_code=(
                        output.predicted_code
                    ),
                    confidence=(
                        output.confidence
                    ),
                    inference_time_ms=(
                        output
                        .inference_time_ms
                    ),
                    top_predictions=[
                        RecognitionTopPrediction(
                            sign_id=(
                                prediction
                                .sign_id
                            ),
                            sign_code=(
                                prediction
                                .sign_code
                            ),
                            confidence=(
                                prediction
                                .confidence
                            ),
                        )
                        for prediction
                        in output.top_predictions
                    ],
                )
            )

            answer_request = (
                QuizAnswerRequest(
                    submission_id=request_id,
                    question_id=question_id,
                    direction="productive",
                    selected_sign_id=None,
                    recognition_result=(
                        recognition_result
                    ),
                    client_response_time_ms=(
                        client_response_time_ms
                    ),
                )
            )

            action_result = await (
                self.quiz_service
                .submit_answer(
                    current_user=(
                        current_user
                    ),
                    session_id=session_id,
                    request_data=(
                        answer_request
                    ),
                    trusted_recognition=True,
                )
            )

            keep_file = (
                self.settings
                .recognition_retain_uploads
            )

            stored_result = (
                self._result_to_document(
                    recognition_result
                )
            )

            stored_path = (
                str(temporary_path)
                if keep_file
                else None
            )

            await self.repository.create_attempt(
                {
                    "request_id": (
                        request_id_text
                    ),
                    "student_id": ObjectId(
                        current_user.student_id
                    ),
                    "session_id": ObjectId(
                        session_id
                    ),
                    "question_id": (
                        question_id_text
                    ),
                    "expected_sign_id": (
                        question["sign_id"]
                    ),
                    "expected_code": (
                        question[
                            "expected_code"
                        ]
                    ),
                    "capture": {
                        "content_type": (
                            content_type
                        ),
                        "byte_size": byte_size,
                        "sha256": file_hash,
                        "duration_ms": (
                            duration_ms
                        ),
                        "retained": keep_file,
                        "storage_path": (
                            stored_path
                        ),
                    },
                    "result": stored_result,
                    "accepted": bool(
                        action_result
                        .feedback.is_correct
                    ),
                    "question_finalized": (
                        action_result.feedback
                        .question_finalized
                    ),
                    "retry_allowed": (
                        action_result.feedback
                        .retry_allowed
                    ),
                    "acceptance_threshold": (
                        self.settings
                        .productive_acceptance_confidence
                    ),
                    "recognizer_mode": (
                        self.settings
                        .recognizer_mode
                    ),
                    "created_at": (
                        datetime.now(
                            timezone.utc
                        )
                    ),
                }
            )

            return (
                ProductiveRecognitionAnswerResponse(
                    session=(
                        action_result.session
                    ),
                    feedback=(
                        action_result.feedback
                    ),
                    recognition=(
                        recognition_result
                    ),
                    capture=(
                        RecognitionCaptureResponse(
                            request_id=(
                                request_id_text
                            ),
                            content_type=(
                                content_type
                            ),
                            byte_size=(
                                byte_size
                            ),
                            sha256=file_hash,
                            duration_ms=(
                                duration_ms
                            ),
                            retained=(
                                keep_file
                            ),
                        )
                    ),
                )
            )

        finally:
            await video.close()

            if (
                temporary_path is not None
                and temporary_path.exists()
                and not keep_file
            ):
                temporary_path.unlink(
                    missing_ok=True
                )

    def _validate_duration(
        self,
        duration_ms: int,
    ) -> None:
        if (
            duration_ms
            < self.settings
            .recognition_min_duration_ms
        ):
            raise RecognitionDurationError(
                "The recording is too short."
            )

        if (
            duration_ms
            > self.settings
            .recognition_max_duration_ms
        ):
            raise RecognitionDurationError(
                "The recording is too long."
            )

    async def _save_upload(
        self,
        *,
        video: UploadFile,
        request_id: str,
    ) -> tuple[
        Path,
        int,
        str,
        str,
    ]:
        content_type = (
            video.content_type or ""
        ).lower()

        extension = (
            self.ALLOWED_CONTENT_TYPES.get(
                content_type
            )
        )

        if extension is None:
            raise (
                UnsupportedRecognitionMediaError(
                    "Supported video formats are "
                    "MP4, MOV and WebM."
                )
            )

        directory = Path(
            self.settings
            .recognition_temp_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = directory / (
            f"{request_id}{extension}"
        )

        digest = sha256()
        byte_size = 0

        try:
            with destination.open(
                "wb"
            ) as output_file:
                while True:
                    chunk = await video.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    byte_size += len(chunk)

                    if (
                        byte_size
                        > self.settings
                        .recognition_max_upload_bytes
                    ):
                        raise (
                            RecognitionFileTooLargeError(
                                "The uploaded video "
                                "is too large."
                            )
                        )

                    digest.update(chunk)
                    output_file.write(chunk)

        except Exception:
            destination.unlink(
                missing_ok=True
            )
            raise

        if byte_size == 0:
            destination.unlink(
                missing_ok=True
            )

            raise (
                UnsupportedRecognitionMediaError(
                    "The uploaded video is empty."
                )
            )

        return (
            destination,
            byte_size,
            digest.hexdigest(),
            content_type,
        )

    def _result_to_document(
        self,
        result: RecognitionResult,
    ) -> dict:
        return {
            "request_id": str(
                result.request_id
            ),
            "model_name": (
                result.model_name
            ),
            "model_version": (
                result.model_version
            ),
            "predicted_sign_id": (
                ObjectId(
                    result
                    .predicted_sign_id
                )
                if (
                    result
                    .predicted_sign_id
                )
                else None
            ),
            "predicted_code": (
                result.predicted_code
            ),
            "confidence": float(
                result.confidence
            ),
            "inference_time_ms": (
                result.inference_time_ms
            ),
            "top_predictions": [
                {
                    "sign_id": (
                        ObjectId(
                            prediction
                            .sign_id
                        )
                        if (
                            prediction
                            .sign_id
                        )
                        else None
                    ),
                    "sign_code": (
                        prediction
                        .sign_code
                    ),
                    "confidence": float(
                        prediction
                        .confidence
                    ),
                }
                for prediction
                in result.top_predictions
            ],
        }

    def _document_to_result(
        self,
        document: dict,
    ) -> RecognitionResult:
        return RecognitionResult(
            request_id=document[
                "request_id"
            ],
            model_name=document[
                "model_name"
            ],
            model_version=document[
                "model_version"
            ],
            predicted_sign_id=(
                str(
                    document[
                        "predicted_sign_id"
                    ]
                )
                if document[
                    "predicted_sign_id"
                ]
                else None
            ),
            predicted_code=document[
                "predicted_code"
            ],
            confidence=float(
                document["confidence"]
            ),
            inference_time_ms=document[
                "inference_time_ms"
            ],
            top_predictions=[
                RecognitionTopPrediction(
                    sign_id=(
                        str(
                            prediction[
                                "sign_id"
                            ]
                        )
                        if prediction[
                            "sign_id"
                        ]
                        else None
                    ),
                    sign_code=prediction[
                        "sign_code"
                    ],
                    confidence=float(
                        prediction[
                            "confidence"
                        ]
                    ),
                )
                for prediction
                in document[
                    "top_predictions"
                ]
            ],
        )

    async def _existing_response(
        self,
        *,
        existing: dict,
        current_user: AuthenticatedUser,
        session_id: str,
    ) -> ProductiveRecognitionAnswerResponse:
        session = await (
            self.quiz_service.get_session(
                current_user=current_user,
                session_id=session_id,
            )
        )

        recognition = (
            self._document_to_result(
                existing["result"]
            )
        )

        capture = existing["capture"]

        return ProductiveRecognitionAnswerResponse(
            session=session,
            feedback=QuizFeedbackResponse(
                action="answer",
                is_correct=existing[
                    "accepted"
                ],
                question_finalized=existing[
                    "question_finalized"
                ],
                retry_allowed=existing[
                    "retry_allowed"
                ],
                message=(
                    "This video submission was "
                    "already processed."
                ),
                recognition_confidence=(
                    recognition.confidence
                ),
            ),
            recognition=recognition,
            capture=(
                RecognitionCaptureResponse(
                    request_id=existing[
                        "request_id"
                    ],
                    content_type=capture[
                        "content_type"
                    ],
                    byte_size=capture[
                        "byte_size"
                    ],
                    sha256=capture[
                        "sha256"
                    ],
                    duration_ms=capture[
                        "duration_ms"
                    ],
                    retained=capture[
                        "retained"
                    ],
                )
            ),
        )