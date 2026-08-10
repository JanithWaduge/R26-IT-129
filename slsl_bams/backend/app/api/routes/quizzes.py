import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)

from app.api.dependencies.auth import (
    CurrentUserDependency,
)
from app.db.dependencies import get_database
from app.repositories.quiz_repository import (
    QuizConcurrentUpdateError,
    QuizRepository,
    QuizSessionNotFoundError,
)
from app.repositories.student_repository import (
    StudentRepository,
)
from app.schemas.quiz import (
    QuizActionRequest,
    QuizActionResponse,
    QuizAnswerRequest,
    QuizHintResponse,
    QuizSessionCreate,
    QuizSessionResponse,
)
from app.services.quiz_service import (
    QuizContentUnavailableError,
    QuizQuestionMismatchError,
    QuizService,
    QuizSessionStateError,
)

from app.repositories.mastery_repository import (
    MasteryRepository,
)
from app.repositories.review_repository import (
    ReviewRepository,
)
from app.repositories.gamification_repository import GamificationRepository
from app.services.gamification_service import GamificationService
from app.ml.model_runtime import RecallModelRuntime
from app.repositories.hybrid_scheduler_repository import HybridSchedulerRepository
from app.services.hybrid_scheduler_service import HybridSchedulerService
from app.repositories.experiment_repository import ExperimentRepository
from app.services.experiment_service import ExperimentService
from app.services.mastery_service import (
    MasteryService,
)
from app.services.review_service import (
    ReviewService,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_quiz_service(
    request: Request,
    database=Depends(get_database),
) -> QuizService:
    settings = request.app.state.settings

    mastery_repository = MasteryRepository(
        database
    )

    mastery_service = MasteryService(
        repository=mastery_repository,
        settings=settings,
    )

    review_service = ReviewService(
        review_repository=ReviewRepository(
            database
        ),
        mastery_repository=(
            mastery_repository
        ),
        environment=settings.environment,
    )

    gamification_service = GamificationService(
        repository=GamificationRepository(database),
        settings=settings,
    )
    hybrid_scheduler_service = HybridSchedulerService(
        repository=HybridSchedulerRepository(database),
        runtime=RecallModelRuntime(settings),
        settings=settings,
    )
    experiment_service = ExperimentService(
        repository=ExperimentRepository(database), settings=settings
    )

    return QuizService(
        repository=QuizRepository(
            database
        ),
        student_repository=(
            StudentRepository(database)
        ),
        mastery_service=(
            mastery_service
        ),
        review_service=review_service,
        gamification_service=gamification_service,
        hybrid_scheduler_service=hybrid_scheduler_service,
        experiment_service=experiment_service,
        settings=settings,
    )


QuizServiceDependency = Annotated[
    QuizService,
    Depends(get_quiz_service),
]


def _handle_quiz_error(
    error: Exception,
) -> HTTPException:
    if isinstance(
        error,
        QuizSessionNotFoundError,
    ):
        return HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(error),
        )

    if isinstance(
        error,
        QuizContentUnavailableError,
    ):
        return HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(error),
        )

    if isinstance(
        error,
        (
            QuizConcurrentUpdateError,
            QuizQuestionMismatchError,
            QuizSessionStateError,
        ),
    ):
        return HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=str(error),
        )

    logger.exception(
        "Unexpected quiz error.",
        exc_info=error,
    )

    return HTTPException(
        status_code=(
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        detail="Unexpected quiz error.",
    )


@router.post(
    "/sessions",
    response_model=QuizSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an adaptive quiz session",
)
async def create_quiz_session(
    request_data: QuizSessionCreate,
    current_user: CurrentUserDependency,
    service: QuizServiceDependency,
) -> QuizSessionResponse:
    try:
        return await service.create_session(
            current_user=current_user,
            request_data=request_data,
        )
    except Exception as error:
        raise _handle_quiz_error(
            error
        ) from error


@router.get(
    "/sessions/{session_id}",
    response_model=QuizSessionResponse,
    summary="Get a quiz session",
)
async def get_quiz_session(
    session_id: str,
    current_user: CurrentUserDependency,
    service: QuizServiceDependency,
) -> QuizSessionResponse:
    try:
        return await service.get_session(
            current_user=current_user,
            session_id=session_id,
        )
    except Exception as error:
        raise _handle_quiz_error(
            error
        ) from error


@router.post(
    "/sessions/{session_id}/hint",
    response_model=QuizHintResponse,
    summary="Request a hint",
)
async def request_quiz_hint(
    session_id: str,
    request_data: QuizActionRequest,
    current_user: CurrentUserDependency,
    service: QuizServiceDependency,
) -> QuizHintResponse:
    try:
        return await service.request_hint(
            current_user=current_user,
            session_id=session_id,
            request_data=request_data,
        )
    except Exception as error:
        raise _handle_quiz_error(
            error
        ) from error


@router.post(
    "/sessions/{session_id}/answer",
    response_model=QuizActionResponse,
    summary="Submit a quiz answer",
)
async def submit_quiz_answer(
    session_id: str,
    request_data: QuizAnswerRequest,
    current_user: CurrentUserDependency,
    service: QuizServiceDependency,
) -> QuizActionResponse:
    try:
        return await service.submit_answer(
            current_user=current_user,
            session_id=session_id,
            request_data=request_data,
        )
    except Exception as error:
        raise _handle_quiz_error(
            error
        ) from error


@router.post(
    "/sessions/{session_id}/skip",
    response_model=QuizActionResponse,
    summary="Skip the current question",
)
async def skip_quiz_question(
    session_id: str,
    request_data: QuizActionRequest,
    current_user: CurrentUserDependency,
    service: QuizServiceDependency,
) -> QuizActionResponse:
    try:
        return await service.skip_question(
            current_user=current_user,
            session_id=session_id,
            request_data=request_data,
        )
    except Exception as error:
        raise _handle_quiz_error(
            error
        ) from error


@router.post(
    "/sessions/{session_id}/abandon",
    response_model=QuizSessionResponse,
    summary="Abandon a quiz session",
)
async def abandon_quiz_session(
    session_id: str,
    current_user: CurrentUserDependency,
    service: QuizServiceDependency,
) -> QuizSessionResponse:
    try:
        return await service.abandon_session(
            current_user=current_user,
            session_id=session_id,
        )
    except Exception as error:
        raise _handle_quiz_error(
            error
        ) from error
