from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.api.dependencies.auth import (
    CurrentUserDependency,
)
from app.core.config import Settings
from app.db.dependencies import get_database
from app.recognition.base import (
    InvalidRecognizerResponseError,
    RecognizerUnavailableError,
)
from app.recognition.factory import (
    build_recognizer,
)
from app.repositories.mastery_repository import (
    MasteryRepository,
)
from app.repositories.gamification_repository import GamificationRepository
from app.repositories.quiz_repository import (
    QuizRepository,
    QuizSessionNotFoundError,
)
from app.repositories.recognition_repository import (
    RecognitionRepository,
)
from app.repositories.review_repository import (
    ReviewRepository,
)
from app.repositories.student_repository import (
    StudentRepository,
)
from app.schemas.recognition import (
    ProductiveRecognitionAnswerResponse,
)
from app.services.mastery_service import (
    MasteryService,
)
from app.services.gamification_service import GamificationService
from app.ml.model_runtime import RecallModelRuntime
from app.repositories.hybrid_scheduler_repository import HybridSchedulerRepository
from app.services.hybrid_scheduler_service import HybridSchedulerService
from app.repositories.experiment_repository import ExperimentRepository
from app.services.experiment_service import ExperimentService
from app.services.quiz_service import (
    QuizConcurrentUpdateError,
    QuizQuestionMismatchError,
    QuizService,
    QuizSessionStateError,
)
from app.services.recognition_service import (
    ProductiveQuestionRequiredError,
    RecognitionDurationError,
    RecognitionFileTooLargeError,
    RecognitionService,
    UnsupportedRecognitionMediaError,
)
from app.services.review_service import (
    ReviewService,
)

router = APIRouter()


def get_recognition_service(
    request: Request,
    database=Depends(get_database),
) -> RecognitionService:
    settings: Settings = (
        request.app.state.settings
    )

    mastery_repository = (
        MasteryRepository(database)
    )

    mastery_service = MasteryService(
        repository=mastery_repository,
        settings=settings,
    )

    review_service = ReviewService(
        review_repository=(
            ReviewRepository(database)
        ),
        mastery_repository=(
            mastery_repository
        ),
        environment=(
            settings.environment
        ),
    )

    quiz_repository = QuizRepository(
        database
    )

    quiz_service = QuizService(
        repository=quiz_repository,
        student_repository=(
            StudentRepository(database)
        ),
        mastery_service=(
            mastery_service
        ),
        review_service=review_service,
        gamification_service=GamificationService(
            repository=GamificationRepository(database),
            settings=settings,
        ),
        hybrid_scheduler_service=HybridSchedulerService(
            repository=HybridSchedulerRepository(database),
            runtime=RecallModelRuntime(settings),
            settings=settings,
        ),
        experiment_service=ExperimentService(
            repository=ExperimentRepository(database), settings=settings,
        ),
        settings=settings,
    )

    return RecognitionService(
        repository=RecognitionRepository(
            database
        ),
        quiz_repository=quiz_repository,
        quiz_service=quiz_service,
        recognizer=build_recognizer(
            settings
        ),
        settings=settings,
    )


@router.post(
    "/sessions/{session_id}/recognize-answer",
    response_model=(
        ProductiveRecognitionAnswerResponse
    ),
    summary=(
        "Recognize and submit a productive answer"
    ),
)
async def recognize_productive_answer(
    session_id: str,
    current_user: CurrentUserDependency,
    service: Annotated[
        RecognitionService,
        Depends(get_recognition_service),
    ],
    request_id: Annotated[
        UUID,
        Form(),
    ],
    question_id: Annotated[
        UUID,
        Form(),
    ],
    client_response_time_ms: Annotated[
        int,
        Form(ge=0, le=600000),
    ],
    duration_ms: Annotated[
        int,
        Form(ge=0, le=60000),
    ],
    video: Annotated[
        UploadFile,
        File(
            description=(
                "Short sign-performance video"
            )
        ),
    ],
) -> ProductiveRecognitionAnswerResponse:
    try:
        return await (
            service.recognize_and_answer(
                current_user=current_user,
                session_id=session_id,
                request_id=request_id,
                question_id=question_id,
                client_response_time_ms=(
                    client_response_time_ms
                ),
                duration_ms=duration_ms,
                video=video,
            )
        )

    except RecognitionFileTooLargeError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_413_CONTENT_TOO_LARGE
            ),
            detail=str(error),
        ) from error

    except UnsupportedRecognitionMediaError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail=str(error),
        ) from error

    except RecognitionDurationError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(error),
        ) from error

    except (
        RecognizerUnavailableError,
        InvalidRecognizerResponseError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_502_BAD_GATEWAY
            ),
            detail=str(error),
        ) from error

    except QuizSessionNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(error),
        ) from error

    except (
        ProductiveQuestionRequiredError,
        QuizSessionStateError,
        QuizQuestionMismatchError,
        QuizConcurrentUpdateError,
    ) as error:
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=str(error),
        ) from error
