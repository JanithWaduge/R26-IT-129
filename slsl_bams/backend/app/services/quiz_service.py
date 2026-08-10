import random
import secrets
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4
from app.services.mastery_service import MasteryService

from bson import ObjectId

from app.core.config import Settings
from app.repositories.quiz_repository import (
    QuizConcurrentUpdateError,
    QuizRepository,
)
from app.repositories.student_repository import (
    StudentRepository,
)
from app.schemas.auth import AuthenticatedUser
from app.schemas.quiz import (
    QuizActionRequest,
    QuizActionResponse,
    QuizAnswerRequest,
    QuizCurrentQuestionResponse,
    QuizFeedbackResponse,
    QuizHintResponse,
    QuizOptionResponse,
    QuizSessionCreate,
    QuizSessionResponse,
    QuizSummaryResponse,
)

from app.services.review_service import (
    ReviewService,
)
from app.services.gamification_service import GamificationService
from app.services.hybrid_scheduler_service import HybridSchedulerService
from app.services.experiment_service import ExperimentService

class QuizContentUnavailableError(Exception):
    pass


class QuizSessionStateError(Exception):
    pass


class QuizQuestionMismatchError(Exception):
    pass


class QuizService:
    def __init__(
        self,
        *,
        repository: QuizRepository,
        student_repository: StudentRepository,
        mastery_service: MasteryService,
        review_service: ReviewService,
        gamification_service: GamificationService,
        hybrid_scheduler_service: HybridSchedulerService,
        experiment_service: ExperimentService,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.student_repository = (
            student_repository
        )
        self.mastery_service = (
            mastery_service
        )
        self.review_service = review_service
        self.gamification_service = gamification_service
        self.hybrid_scheduler_service = hybrid_scheduler_service
        self.experiment_service = experiment_service
        self.settings = settings

    async def _apply_reward_if_completed(
        self,
        document: dict,
    ):
        if document["status"] != "completed":
            return None

        return await self.gamification_service.apply_completed_session(
            document
        )

    def _visible_statuses(
        self,
    ) -> list[str]:
        if self.settings.environment in {
            "development",
            "test",
        }:
            return [
                "approved",
                "development",
            ]

        return ["approved"]

    async def create_session(
        self,
        *,
        current_user: AuthenticatedUser,
        request_data: QuizSessionCreate,
    ) -> QuizSessionResponse:
        assignment = await self.experiment_service.resolve_assignment(
            student_id=current_user.student_id
        )
        requested_strategy = (
            request_data.selection_strategy.value
            if hasattr(request_data.selection_strategy, "value")
            else str(request_data.selection_strategy)
        )
        if assignment.arm == "random_control":
            effective_strategy = "random_control"
        elif assignment.arm in {"modified_sm2", "hybrid_ml"}:
            effective_strategy = "adaptive"
        else:
            effective_strategy = requested_strategy

        client_request_id = str(
            request_data.client_request_id
        )

        existing = (
            await self.repository
            .find_existing_creation(
                student_id=(
                    current_user.student_id
                ),
                client_request_id=(
                    client_request_id
                ),
            )
        )

        if existing is not None:
            return self._to_response(existing)

        student = await (
            self.student_repository.get_by_id(
                current_user.student_id
            )
        )

        prompt_language = (
            request_data.prompt_language
            or student.preferred_language
        )

        candidates = (
            await self.repository
            .find_sign_candidates(
                visible_statuses=(
                    self._visible_statuses()
                ),
                category_id=(
                    request_data.category_id
                ),
                competency_id=(
                    request_data.competency_id
                ),
                difficulty=(
                    request_data.difficulty
                ),
            )
        )

        if len(candidates) < (
            request_data.question_count
        ):
            raise QuizContentUnavailableError(
                "Not enough eligible signs exist "
                "for the requested quiz."
            )

        all_visible_signs = (
            await self.repository
            .find_all_visible_signs(
                visible_statuses=(
                    self._visible_statuses()
                )
            )
        )

        if len(all_visible_signs) < 2:
            raise QuizContentUnavailableError(
                "At least two visible signs are "
                "required to generate a quiz."
            )

        category_ids = list(
            {
                sign["category_id"]
                for sign in all_visible_signs
            }
        )

        categories = (
            await self.repository
            .get_category_documents(
                category_ids
            )
        )

        # selection_seed = secrets.randbits(
        #     63
        # )

        # rng = random.Random(
        #     selection_seed
        # )

        # selected_signs = list(candidates)
        # rng.shuffle(selected_signs)

        # selected_signs = selected_signs[
        #     : request_data.question_count
        # ]

        selection_seed = secrets.randbits(
            63
        )

        rng = random.Random(
            selection_seed
        )

        strategy = effective_strategy

        if strategy == "random_control":
            shuffled = list(candidates)
            rng.shuffle(shuffled)

            selected_candidates = [
                {
                    "sign": sign,
                    "direction": (
                        self._resolve_direction(
                            mode=str(
                                request_data.mode
                            ),
                            index=index,
                        )
                    ),
                    "priority_score": 0.0,
                    "selection_reason": (
                        "random_control"
                    ),
                    "was_due": False,
                    "days_overdue": 0.0,
                }
                for index, sign in enumerate(
                    shuffled[
                        : request_data
                        .question_count
                    ]
                )
            ]

        else:
            ranked_candidates = (
                await self.review_service
                .rank_candidates(
                    student_id=(
                        current_user
                        .student_id
                    ),
                    signs=candidates,
                    mode=str(
                        request_data.mode
                    ),
                )
            )

            if strategy == "due_only":
                ranked_candidates = [
                    item
                    for item
                    in ranked_candidates
                    if item["was_due"]
                ]

            selected_candidates = (
                ranked_candidates[
                    : request_data
                    .question_count
                ]
            )

        if not selected_candidates:
            raise QuizContentUnavailableError(
                "No eligible review items exist "
                "for the selected strategy."
            )

        if (
            strategy != "due_only"
            and len(selected_candidates)
            < request_data.question_count
        ):
            raise QuizContentUnavailableError(
                "Not enough eligible signs exist "
                "for the requested quiz."
            )

        now = datetime.now(timezone.utc)

        questions = []

        for index, selected in enumerate(
            selected_candidates
):
            sign = selected["sign"]
            direction = selected[
                "direction"
            ]

            category = categories.get(
                str(sign["category_id"])
            )

            if category is None:
                raise QuizContentUnavailableError(
                    "A sign is linked to a missing "
                    "curriculum category."
                )

            options = []

            if direction == "receptive":
                options = self._build_options(
                    target_sign=sign,
                    all_signs=all_visible_signs,
                    rng=rng,
                )

            questions.append(
                {
                    "question_id": str(
                        uuid4()
                    ),
                    "sign_id": sign["_id"],
                    "expected_code": sign[
                        "code"
                    ],
                    "direction": direction,
                    "order_index": index,
                    "prompt_snapshot": {
                        "meanings": deepcopy(
                            sign["meanings"]
                        ),
                        "category_name": deepcopy(
                            category["name"]
                        ),
                        "difficulty": sign[
                            "difficulty"
                        ],
                        "media": deepcopy(
                            sign["media"]
                        ),
                    },
                    "option_snapshots": options,
                    "status": "pending",
                    "started_at": (
                        now
                        if index == 0
                        else None
                    ),
                    "completed_at": None,
                    "interactions": [],
                    "hint_count": 0,
                    "retry_count": 0,
                    "final_correct": None,
                    "final_response_time_ms": None,
                    # "selection_priority": float(
                    #     selected[
                    #         "priority_score"
                    #     ]
                    # ),
                    # "selection_reason": selected[
                    #     "selection_reason"
                    # ],
                    # "was_due": bool(
                    #     selected["was_due"]
                    # ),
                    # "days_overdue": float(
                    #     selected[
                    #         "days_overdue"
                    #     ]
                    # ),

                    "selection_priority": float(
                        selected["priority_score"]
                    ),
                    "selection_reason": selected[
                        "selection_reason"
                    ],
                    "was_due": bool(
                        selected["was_due"]
                    ),
                    "days_overdue": float(
                        selected["days_overdue"]
                    ),

                    "mastery_event_id": str(
                        uuid4()
                    ),
                    "mastery_applied_at": None,
                    "quality_score": None,
                }
            )

        document = {
            "student_id": ObjectId(
                current_user.student_id
            ),
            "client_request_id": (
                client_request_id
            ),
            "mode": str(request_data.mode),
            "selection_strategy": strategy,
            "experiment_name": assignment.experiment_name,
            "experiment_arm": assignment.arm,
            "prompt_language": str(
                prompt_language
            ),
            "status": "in_progress",
            "filters": {
                "category_id": (
                    ObjectId(
                        request_data.category_id
                    )
                    if request_data.category_id
                    else None
                ),
                "competency_id": (
                    ObjectId(
                        request_data.competency_id
                    )
                    if request_data.competency_id
                    else None
                ),
                "difficulty": (
                    request_data.difficulty
                ),
            },
            "question_count": len(
                questions
            ),
            "current_question_index": 0,
            "questions": questions,
            "summary": {
                "correct_count": 0,
                "incorrect_count": 0,
                "skipped_count": 0,
                "hints_used": 0,
                "total_retries": 0,
                "receptive_correct": 0,
                "productive_correct": 0,
            },
            "selection_seed": selection_seed,
            "version": 1,
            "started_at": now,
            "expires_at": now + timedelta(
                minutes=(
                    self.settings
                    .quiz_session_duration_minutes
                )
            ),
            "completed_at": None,
            "created_at": now,
            "updated_at": now,
        }

        created = (
            await self.repository
            .create_session(document)
        )

        return self._to_response(created)

    

    async def get_session(
        self,
        *,
        current_user: AuthenticatedUser,
        session_id: str,
    ) -> QuizSessionResponse:
        document = (
            await self.repository
            .get_owned_session(
                session_id=session_id,
                student_id=(
                    current_user.student_id
                ),
            )
        )

        document = await self._expire_if_needed(
            document
        )

        document = await (
            self._reconcile_mastery(
                document
            )
        )

        if document["status"] == "completed":
            await self._apply_reward_if_completed(document)

        return self._to_response(document)

    async def request_hint(
        self,
        *,
        current_user: AuthenticatedUser,
        session_id: str,
        request_data: QuizActionRequest,
    ) -> QuizHintResponse:
        document = (
            await self.repository
            .get_owned_session(
                session_id=session_id,
                student_id=(
                    current_user.student_id
                ),
            )
        )

        document = await self._expire_if_needed(
            document
        )

        self._require_in_progress(
            document
        )

        question = self._get_current_question(
            document
        )

        self._require_question_id(
            question,
            str(request_data.question_id),
        )

        action_id = str(
            request_data.action_id
        )

        existing = self._find_interaction(
            question,
            action_id,
        )

        if existing is not None:
            return self._build_hint_response(
                document=document,
                question=question,
            )

        now = datetime.now(timezone.utc)

        question["interactions"].append(
            {
                "event_id": action_id,
                "event_type": "hint",
                "attempt_number": 0,
                "selected_sign_id": None,
                "recognition_result": None,
                "is_correct": None,
                "client_response_time_ms": None,
                "server_response_time_ms": None,
                "created_at": now,
            }
        )

        question["hint_count"] += 1

        document["summary"][
            "hints_used"
        ] += 1

        document["updated_at"] = now

        updated = (
            await self.repository
            .replace_with_version(
                document=document,
                expected_version=document[
                    "version"
                ],
            )
        )

        updated = await (
            self._reconcile_mastery(
                updated
            )
        )

        updated_question = (
            self._get_current_question(
                updated
            )
        )

        return self._build_hint_response(
            document=updated,
            question=updated_question,
        )

    async def _reconcile_mastery(
        self,
        document: dict,
    ) -> dict:
        for _ in range(20):
            pending_question = next(
                (
                    question
                    for question in document[
                        "questions"
                    ]
                    if (
                        question["status"]
                        != "pending"
                        and question[
                            "mastery_applied_at"
                        ]
                        is None
                    )
                ),
                None,
            )

            if pending_question is None:
                await self.hybrid_scheduler_service.reconcile_session(document)
                return document

            result = await (
                self.mastery_service
                .apply_finalized_question(
                    session_document=(
                        document
                    ),
                    question=(
                        pending_question
                    ),
                )
            )

            pending_question[
                "mastery_applied_at"
            ] = result.applied_at

            pending_question[
                "quality_score"
            ] = result.quality_score

            document["updated_at"] = (
                datetime.now(timezone.utc)
            )

            try:
                document = await (
                    self.repository
                    .replace_with_version(
                        document=document,
                        expected_version=document[
                            "version"
                        ],
                    )
                )
            except QuizConcurrentUpdateError:
                document = await (
                    self.repository
                    .get_owned_session(
                        session_id=str(
                            document["_id"]
                        ),
                        student_id=str(
                            document[
                                "student_id"
                            ]
                        ),
                    )
                )

        raise QuizConcurrentUpdateError(
            "Pending mastery events could not "
            "be reconciled."
        )

    async def submit_answer(
    self,
    *,
    current_user: AuthenticatedUser,
    session_id: str,
    request_data: QuizAnswerRequest,
    trusted_recognition: bool = False,
) -> QuizActionResponse:
        document = (
            await self.repository
            .get_owned_session(
                session_id=session_id,
                student_id=(
                    current_user.student_id
                ),
            )
        )

        document = await self._expire_if_needed(
            document
        )

        self._require_in_progress(
            document
        )

        question = self._get_current_question(
            document
        )

        self._require_question_id(
            question,
            str(request_data.question_id),
        )

        if (
            question["direction"]
            != str(request_data.direction)
        ):
            raise QuizQuestionMismatchError(
                "Answer direction does not match "
                "the current question."
            )
        if (
            question["direction"]
            == "productive"
            and not trusted_recognition
            and not self.settings
            .allow_client_recognition_results
        ):
            raise QuizSessionStateError(
                "Productive answers must be "
                "submitted through the secure "
                "recognition upload endpoint."
            )

        submission_id = str(
            request_data.submission_id
        )

        existing = self._find_interaction(
            question,
            submission_id,
        )

        if existing is not None:
            return QuizActionResponse(
                session=self._to_response(
                    document
                ),
                feedback=QuizFeedbackResponse(
                    action="answer",
                    is_correct=existing[
                        "is_correct"
                    ],
                    question_finalized=(
                        question["status"]
                        != "pending"
                    ),
                    retry_allowed=(
                        question["status"]
                        == "pending"
                    ),
                    message=(
                        "This submission was "
                        "already processed."
                    ),
                    recognition_confidence=(
                        (
                            existing[
                                "recognition_result"
                            ] or {}
                        ).get("confidence")
                    ),
                ),
            )

        now = datetime.now(timezone.utc)

        server_response_time_ms = (
            self._calculate_server_time(
                question,
                now,
            )
        )

        answer_interactions = [
            interaction
            for interaction in question[
                "interactions"
            ]
            if interaction[
                "event_type"
            ] == "answer"
        ]

        attempt_number = (
            len(answer_interactions) + 1
        )

        selected_sign_id = None
        recognition_document = None

        if question["direction"] == "receptive":
            selected_sign_id = (
                self.repository.parse_object_id(
                    request_data.selected_sign_id,
                    message=(
                        "Selected answer "
                        "identifier is invalid."
                    ),
                )
            )

            is_correct = (
                selected_sign_id
                == question["sign_id"]
            )

        else:
            recognition_document = (
                self._build_recognition_document(
                    request_data
                )
            )

            predicted_sign_id = (
                recognition_document[
                    "predicted_sign_id"
                ]
            )

            is_correct = (
                predicted_sign_id
                == question["sign_id"]
                and recognition_document[
                    "confidence"
                ]
                >= self.settings
                .productive_acceptance_confidence
            )

        question["interactions"].append(
            {
                "event_id": submission_id,
                "event_type": "answer",
                "attempt_number": (
                    attempt_number
                ),
                "selected_sign_id": (
                    selected_sign_id
                ),
                "recognition_result": (
                    recognition_document
                ),
                "is_correct": is_correct,
                "client_response_time_ms": (
                    request_data
                    .client_response_time_ms
                ),
                "server_response_time_ms": (
                    server_response_time_ms
                ),
                "created_at": now,
            }
        )

        question["retry_count"] = max(
            0,
            attempt_number - 1,
        )

        document["summary"][
            "total_retries"
        ] = sum(
            item["retry_count"]
            for item in document["questions"]
        )

        maximum_attempts = (
            1
            + self.settings.quiz_max_retries
        )

        question_finalized = (
            is_correct
            or attempt_number
            >= maximum_attempts
        )

        if question_finalized:
            self._finalize_question(
                document=document,
                question=question,
                is_correct=is_correct,
                response_time_ms=(
                    request_data
                    .client_response_time_ms
                ),
                completed_at=now,
            )

            self._advance_session(
                document=document,
                now=now,
            )

        document["updated_at"] = now

        updated = (
            await self.repository
            .replace_with_version(
                document=document,
                expected_version=document[
                    "version"
                ],
            )
        )

        updated = await (
            self._reconcile_mastery(
                updated
            )
        )

        reward = await self._apply_reward_if_completed(updated)

        retry_allowed = (
            not question_finalized
        )

        if is_correct:
            message = "Correct answer."
        elif retry_allowed:
            message = (
                "Incorrect answer. "
                "Try the same sign again."
            )
        else:
            message = (
                "Incorrect answer. "
                "The maximum number of attempts "
                "has been reached."
            )

        confidence = None

        if recognition_document is not None:
            confidence = (
                recognition_document[
                    "confidence"
                ]
            )

        return QuizActionResponse(
            session=self._to_response(
                updated
            ),
            feedback=QuizFeedbackResponse(
                action="answer",
                is_correct=is_correct,
                question_finalized=(
                    question_finalized
                ),
                retry_allowed=retry_allowed,
                message=message,
                recognition_confidence=(
                    confidence
                ),
            ),
            reward=reward,
        )

    async def skip_question(
        self,
        *,
        current_user: AuthenticatedUser,
        session_id: str,
        request_data: QuizActionRequest,
    ) -> QuizActionResponse:
        document = (
            await self.repository
            .get_owned_session(
                session_id=session_id,
                student_id=(
                    current_user.student_id
                ),
            )
        )

        document = await self._expire_if_needed(
            document
        )

        self._require_in_progress(
            document
        )

        question = self._get_current_question(
            document
        )

        self._require_question_id(
            question,
            str(request_data.question_id),
        )

        action_id = str(
            request_data.action_id
        )

        existing = self._find_interaction(
            question,
            action_id,
        )

        if existing is not None:
            return QuizActionResponse(
                session=self._to_response(
                    document
                ),
                feedback=QuizFeedbackResponse(
                    action="skip",
                    is_correct=None,
                    question_finalized=True,
                    retry_allowed=False,
                    message=(
                        "This skip request was "
                        "already processed."
                    ),
                ),
            )

        now = datetime.now(timezone.utc)

        question["interactions"].append(
            {
                "event_id": action_id,
                "event_type": "skip",
                "attempt_number": 0,
                "selected_sign_id": None,
                "recognition_result": None,
                "is_correct": None,
                "client_response_time_ms": None,
                "server_response_time_ms": (
                    self._calculate_server_time(
                        question,
                        now,
                    )
                ),
                "created_at": now,
            }
        )

        question["status"] = "skipped"
        question["completed_at"] = now
        question["final_correct"] = None
        question[
            "final_response_time_ms"
        ] = None

        document["summary"][
            "skipped_count"
        ] += 1

        self._advance_session(
            document=document,
            now=now,
        )

        document["updated_at"] = now

        updated = (
            await self.repository
            .replace_with_version(
                document=document,
                expected_version=document[
                    "version"
                ],
            )
        )

        updated = await (
            self._reconcile_mastery(
                updated
            )
        )

        reward = await self._apply_reward_if_completed(updated)

        return QuizActionResponse(
            session=self._to_response(
                updated
            ),
            feedback=QuizFeedbackResponse(
                action="skip",
                is_correct=None,
                question_finalized=True,
                retry_allowed=False,
                message="Question skipped.",
            ),
            reward=reward,
        )

    async def abandon_session(
        self,
        *,
        current_user: AuthenticatedUser,
        session_id: str,
    ) -> QuizSessionResponse:
        document = (
            await self.repository
            .get_owned_session(
                session_id=session_id,
                student_id=(
                    current_user.student_id
                ),
            )
        )

        if document["status"] != "in_progress":
            return self._to_response(
                document
            )

        now = datetime.now(timezone.utc)

        document["status"] = "abandoned"
        document["completed_at"] = now
        document["updated_at"] = now

        updated = (
            await self.repository
            .replace_with_version(
                document=document,
                expected_version=document[
                    "version"
                ],
            )
        )

        return self._to_response(updated)

    def _resolve_direction(
        self,
        *,
        mode: str,
        index: int,
    ) -> str:
        if mode == "receptive":
            return "receptive"

        if mode == "productive":
            return "productive"

        return (
            "receptive"
            if index % 2 == 0
            else "productive"
        )

    def _build_options(
        self,
        *,
        target_sign: dict,
        all_signs: list[dict],
        rng: random.Random,
    ) -> list[dict]:
        distractors = [
            sign
            for sign in all_signs
            if sign["_id"]
            != target_sign["_id"]
        ]

        rng.shuffle(distractors)

        selected_distractors = distractors[
            : (
                self.settings
                .quiz_option_count
                - 1
            )
        ]

        option_signs = [
            target_sign,
            *selected_distractors,
        ]

        rng.shuffle(option_signs)

        return [
            {
                "sign_id": sign["_id"],
                "meanings": deepcopy(
                    sign["meanings"]
                ),
            }
            for sign in option_signs
        ]

    def _build_recognition_document(
        self,
        request_data: QuizAnswerRequest,
    ) -> dict:
        recognition = (
            request_data.recognition_result
        )

        if recognition is None:
            raise QuizQuestionMismatchError(
                "Recognition result is missing."
            )

        predicted_sign_id = None

        if recognition.predicted_sign_id:
            predicted_sign_id = (
                self.repository.parse_object_id(
                    recognition.predicted_sign_id,
                    message=(
                        "Predicted sign identifier "
                        "is invalid."
                    ),
                )
            )

        top_predictions = []

        for prediction in (
            recognition.top_predictions
        ):
            prediction_sign_id = None

            if prediction.sign_id:
                prediction_sign_id = (
                    self.repository
                    .parse_object_id(
                        prediction.sign_id,
                        message=(
                            "Top prediction sign "
                            "identifier is invalid."
                        ),
                    )
                )

            top_predictions.append(
                {
                    "sign_id": (
                        prediction_sign_id
                    ),
                    "sign_code": (
                        prediction.sign_code
                    ),
                    "confidence": float(
                        prediction.confidence
                    ),
                }
            )

        return {
            "request_id": str(
                recognition.request_id
            ),
            "model_name": (
                recognition.model_name
            ),
            "model_version": (
                recognition.model_version
            ),
            "predicted_sign_id": (
                predicted_sign_id
            ),
            "predicted_code": (
                recognition.predicted_code
            ),
            "confidence": float(
                recognition.confidence
            ),
            "inference_time_ms": (
                recognition
                .inference_time_ms
            ),
            "top_predictions": (
                top_predictions
            ),
        }

    def _finalize_question(
        self,
        *,
        document: dict,
        question: dict,
        is_correct: bool,
        response_time_ms: int,
        completed_at: datetime,
    ) -> None:
        question["status"] = (
            "correct"
            if is_correct
            else "incorrect"
        )

        question[
            "completed_at"
        ] = completed_at

        question[
            "final_correct"
        ] = is_correct

        question[
            "final_response_time_ms"
        ] = response_time_ms

        if is_correct:
            document["summary"][
                "correct_count"
            ] += 1

            if (
                question["direction"]
                == "receptive"
            ):
                document["summary"][
                    "receptive_correct"
                ] += 1
            else:
                document["summary"][
                    "productive_correct"
                ] += 1

        else:
            document["summary"][
                "incorrect_count"
            ] += 1

    def _advance_session(
        self,
        *,
        document: dict,
        now: datetime,
    ) -> None:
        next_index = (
            document[
                "current_question_index"
            ]
            + 1
        )

        if next_index >= document[
            "question_count"
        ]:
            document["status"] = "completed"
            document["completed_at"] = now
            document[
                "current_question_index"
            ] = document["question_count"]

            return

        document[
            "current_question_index"
        ] = next_index

        document["questions"][
            next_index
        ]["started_at"] = now

    async def _expire_if_needed(
        self,
        document: dict,
    ) -> dict:
        now = datetime.now(timezone.utc)
        expires_at = document["expires_at"]

        # MongoDB stores datetimes as UTC but PyMongo
        # returns them without timezone information unless
        # the client is configured with tz_aware=True.
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(
                tzinfo=timezone.utc
            )

        if (
            document["status"]
            == "in_progress"
            and now
            > expires_at
        ):
            document["status"] = "expired"
            document["completed_at"] = now
            document["updated_at"] = now

            return await (
                self.repository
                .replace_with_version(
                    document=document,
                    expected_version=document[
                        "version"
                    ],
                )
            )

        return document

    def _require_in_progress(
        self,
        document: dict,
    ) -> None:
        if document["status"] != "in_progress":
            raise QuizSessionStateError(
                "This quiz session is no longer "
                "in progress."
            )

    def _get_current_question(
        self,
        document: dict,
    ) -> dict:
        index = document[
            "current_question_index"
        ]

        if index >= len(
            document["questions"]
        ):
            raise QuizSessionStateError(
                "The quiz has no current question."
            )

        return document["questions"][
            index
        ]

    def _require_question_id(
        self,
        question: dict,
        question_id: str,
    ) -> None:
        if (
            question["question_id"]
            != question_id
        ):
            raise QuizQuestionMismatchError(
                "The submitted question is not "
                "the current quiz question."
            )

    def _find_interaction(
        self,
        question: dict,
        event_id: str,
    ) -> dict | None:
        return next(
            (
                interaction
                for interaction in question[
                    "interactions"
                ]
                if interaction[
                    "event_id"
                ] == event_id
            ),
            None,
        )

    def _calculate_server_time(
        self,
        question: dict,
        now: datetime,
    ) -> int:
        started_at = question[
            "started_at"
        ]

        if started_at is None:
            return 0

        if started_at.tzinfo is None:
            started_at = started_at.replace(
                tzinfo=timezone.utc
            )

        elapsed = now - started_at

        return max(
            0,
            int(
                elapsed.total_seconds()
                * 1000
            ),
        )

    def _localized_value(
        self,
        localized: dict,
        language: str,
    ) -> str:
        return localized.get(
            language,
            localized["english"],
        )

    def _build_hint_response(
        self,
        *,
        document: dict,
        question: dict,
    ) -> QuizHintResponse:
        language = document[
            "prompt_language"
        ]

        category_name = (
            self._localized_value(
                question[
                    "prompt_snapshot"
                ]["category_name"],
                language,
            )
        )

        return QuizHintResponse(
            hint_type="category",
            message=(
                "This sign belongs to the "
                f"'{category_name}' category."
            ),
            hint_count=question[
                "hint_count"
            ],
            session=self._to_response(
                document
            ),
        )

    def _to_response(
        self,
        document: dict,
    ) -> QuizSessionResponse:
        answered_count = sum(
            1
            for question in document[
                "questions"
            ]
            if question["status"]
            != "pending"
        )

        progress_percentage = round(
            (
                answered_count
                / document["question_count"]
            )
            * 100,
            2,
        )

        current_question = None

        if (
            document["status"]
            == "in_progress"
            and document[
                "current_question_index"
            ]
            < document["question_count"]
        ):
            question = document[
                "questions"
            ][
                document[
                    "current_question_index"
                ]
            ]

            language = document[
                "prompt_language"
            ]

            answer_count = sum(
                1
                for interaction in question[
                    "interactions"
                ]
                if interaction[
                    "event_type"
                ] == "answer"
            )

            maximum_attempts = (
                1
                + self.settings
                .quiz_max_retries
            )

            prompt_text = None
            media = None
            options = []

            if (
                question["direction"]
                == "receptive"
            ):
                media = question[
                    "prompt_snapshot"
                ]["media"]

                options = [
                    QuizOptionResponse(
                        sign_id=str(
                            option[
                                "sign_id"
                            ]
                        ),
                        text=(
                            self._localized_value(
                                option[
                                    "meanings"
                                ],
                                language,
                            )
                        ),
                    )
                    for option in question[
                        "option_snapshots"
                    ]
                ]

            else:
                prompt_text = (
                    self._localized_value(
                        question[
                            "prompt_snapshot"
                        ]["meanings"],
                        language,
                    )
                )

            current_question = (
                QuizCurrentQuestionResponse(
                    question_id=question[
                        "question_id"
                    ],
                    direction=question[
                        "direction"
                    ],
                    order_index=question[
                        "order_index"
                    ],
                    prompt_text=prompt_text,
                    media=media,
                    category_name=(
                        self._localized_value(
                            question[
                                "prompt_snapshot"
                            ]["category_name"],
                            language,
                        )
                    ),
                    difficulty=question[
                        "prompt_snapshot"
                    ]["difficulty"],
                    options=options,
                    attempt_number=(
                        answer_count + 1
                    ),
                    remaining_attempts=max(
                        0,
                        maximum_attempts
                        - answer_count,
                    ),
                    hint_used=(
                        question[
                            "hint_count"
                        ]
                        > 0
                    ),
                    started_at=question[
                        "started_at"
                    ],
                    selection_priority=float(
                        question[
                            "selection_priority"
                        ]
                    ),
                    selection_reason=question[
                        "selection_reason"
                    ],
                    was_due=question[
                        "was_due"
                    ],
                    days_overdue=float(
                        question[
                            "days_overdue"
                        ]
                    ),
                )
            )

        return QuizSessionResponse(
            id=str(document["_id"]),
            mode=document["mode"],
            selection_strategy=document[
                "selection_strategy"
            ],
            experiment_name=document.get("experiment_name"),
            experiment_arm=document.get("experiment_arm", "legacy_unassigned"),
            prompt_language=document[
                "prompt_language"
            ],
            status=document["status"],
            question_count=document[
                "question_count"
            ],
            current_question_index=document[
                "current_question_index"
            ],
            answered_count=answered_count,
            progress_percentage=(
                progress_percentage
            ),
            summary=QuizSummaryResponse(
                **document["summary"]
            ),
            current_question=(
                current_question
            ),
            started_at=document[
                "started_at"
            ],
            expires_at=document[
                "expires_at"
            ],
            completed_at=document[
                "completed_at"
            ],
            version=document["version"],
        )

    
