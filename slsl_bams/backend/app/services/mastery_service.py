from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from typing import Any

from bson import ObjectId

from app.algorithms.mastery_update import (
    initial_direction_state,
    update_direction_state,
)
from app.algorithms.quality_score import (
    calculate_quality_score,
)
from app.core.config import Settings
from app.repositories.mastery_repository import (
    MasteryConcurrentUpdateError,
    MasteryRepository,
)
from app.schemas.mastery import (
    DirectionMasteryResponse,
    MasteryOverviewResponse,
    MasteryStatusCounts,
    PaginatedMasteryResponse,
    SignMasteryResponse,
)
from app.algorithms.modified_sm2 import (
    apply_schedule_to_state,
    update_sm2_schedule,
)


@dataclass(frozen=True)
class AppliedMasteryResult:
    event_id: str
    quality_score: float
    quality_band: int
    applied_at: datetime


class MasteryService:
    def __init__(
        self,
        *,
        repository: MasteryRepository,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.settings = settings

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

    async def apply_finalized_question(
        self,
        *,
        session_document: dict,
        question: dict,
    ) -> AppliedMasteryResult:
        if question["status"] == "pending":
            raise ValueError(
                "Pending questions cannot "
                "update mastery."
            )

        event_id = question[
            "mastery_event_id"
        ]

        existing_event = (
            await self.repository
            .create_or_get_event(
                self._build_event_document(
                    session_document=(
                        session_document
                    ),
                    question=question,
                )
            )
        )

        if (
            existing_event["status"]
            == "applied"
        ):
            return AppliedMasteryResult(
                event_id=event_id,
                quality_score=float(
                    existing_event[
                        "quality_score"
                    ]
                ),
                quality_band=(
                    existing_event[
                        "quality_band"
                    ]
                ),
                applied_at=(
                    existing_event[
                        "applied_at"
                    ]
                ),
            )

        recognition_confidence = (
            self._recognition_confidence(
                question
            )
        )

        quality = calculate_quality_score(
            direction=question["direction"],
            question_status=question[
                "status"
            ],
            final_correct=question[
                "final_correct"
            ],
            retry_count=question[
                "retry_count"
            ],
            hint_count=question[
                "hint_count"
            ],
            response_time_ms=question[
                "final_response_time_ms"
            ],
            difficulty=question[
                "prompt_snapshot"
            ]["difficulty"],
            recognition_confidence=(
                recognition_confidence
            ),
            productive_acceptance_threshold=(
                self.settings
                .productive_acceptance_confidence
            ),
        )

        student_id = session_document[
            "student_id"
        ]

        sign_id = question["sign_id"]

        for _ in range(8):
            record = await (
                self.repository
                .ensure_mastery_record(
                    student_id=student_id,
                    sign_id=sign_id,
                )
            )

            processed_event = next(
                (
                    event
                    for event in record[
                        "processed_events"
                    ]
                    if event[
                        "event_id"
                    ] == event_id
                ),
                None,
            )

            if processed_event is not None:
                applied_at = processed_event[
                    "applied_at"
                ]

                await self.repository.mark_event_applied(
                    event_id=event_id,
                    before_state={
                        "score": processed_event[
                            "before_score"
                        ]
                    },
                    after_state={
                        "score": processed_event[
                            "after_score"
                        ]
                    },
                    applied_at=applied_at,
                )

                return AppliedMasteryResult(
                    event_id=event_id,
                    quality_score=(
                        processed_event[
                            "quality_score"
                        ]
                    ),
                    quality_band=(
                        processed_event[
                            "quality_band"
                        ]
                    ),
                    applied_at=applied_at,
                )

            direction = question[
                "direction"
            ]

            before_state = deepcopy(
                record[direction]
            )

            applied_at = datetime.now(
                timezone.utc
            )

            after_state = (
                update_direction_state(
                    current_state=(
                        before_state
                    ),
                    quality_score=(
                        quality.score
                    ),
                    quality_band=(
                        quality.band
                    ),
                    final_correct=question[
                        "final_correct"
                    ],
                    retry_count=question[
                        "retry_count"
                    ],
                    hint_count=question[
                        "hint_count"
                    ],
                    response_time_ms=question[
                        "final_response_time_ms"
                    ],
                    recognition_confidence=(
                        recognition_confidence
                    ),
                    reviewed_at=applied_at,
                    session_id=(
                        session_document["_id"]
                    ),
                    question_id=question[
                        "question_id"
                    ],
                    settings=self.settings,
                )
            )

            schedule = update_sm2_schedule(
                current_state=before_state,
                quality_band=quality.band,
                mastery_score=float(
                    after_state["score"]
                ),
                difficulty=question[
                    "prompt_snapshot"
                ]["difficulty"],
                reviewed_at=applied_at,
                settings=self.settings,
            )

            after_state = apply_schedule_to_state(
                state=after_state,
                schedule=schedule,
            )

            updated_record = await (
                self.repository
                .apply_mastery_update(
                    current_record=record,
                    event_id=event_id,
                    direction=direction,
                    updated_direction_state=(
                        after_state
                    ),
                    quality_score=(
                        quality.score
                    ),
                    quality_band=(
                        quality.band
                    ),
                    applied_at=applied_at,
                )
            )

            if updated_record is None:
                continue

            await self.repository.mark_event_applied(
                event_id=event_id,
                before_state=before_state,
                after_state=after_state,
                applied_at=applied_at,
            )

            return AppliedMasteryResult(
                event_id=event_id,
                quality_score=(
                    quality.score
                ),
                quality_band=quality.band,
                applied_at=applied_at,
            )

        raise MasteryConcurrentUpdateError(
            "Mastery could not be updated "
            "after multiple concurrent attempts."
        )

    async def get_overview(
        self,
        *,
        student_id: str,
    ) -> MasteryOverviewResponse:
        student_object_id = (
            self.repository.parse_object_id(
                student_id,
                message=(
                    "Student identifier "
                    "is invalid."
                ),
            )
        )

        signs = await (
            self.repository.list_visible_signs(
                visible_statuses=(
                    self._visible_statuses()
                )
            )
        )

        records = await (
            self.repository
            .get_records_for_student(
                student_id=(
                    student_object_id
                )
            )
        )

        record_map = {
            str(record["sign_id"]): record
            for record in records
        }

        statuses = {
            "new": 0,
            "very_weak": 0,
            "weak": 0,
            "learning": 0,
            "proficient": 0,
            "mastered": 0,
        }

        receptive_total = 0.0
        productive_total = 0.0
        combined_total = 0.0

        attempted_signs = 0
        receptive_attempted = 0
        productive_attempted = 0
        bidirectional = 0

        receptive_weaker = 0
        productive_weaker = 0
        balanced = 0

        for sign in signs:
            record = record_map.get(
                str(sign["_id"])
            )

            if record is None:
                statuses["new"] += 1
                continue

            receptive = record[
                "receptive"
            ]

            productive = record[
                "productive"
            ]

            receptive_total += float(
                receptive["score"]
            )

            productive_total += float(
                productive["score"]
            )

            combined_total += float(
                record["combined_score"]
            )

            attempted = (
                receptive["total_reviews"] > 0
                or productive[
                    "total_reviews"
                ] > 0
            )

            if attempted:
                attempted_signs += 1

            if receptive[
                "total_reviews"
            ] > 0:
                receptive_attempted += 1

            if productive[
                "total_reviews"
            ] > 0:
                productive_attempted += 1

            if (
                receptive[
                    "total_reviews"
                ] > 0
                and productive[
                    "total_reviews"
                ] > 0
            ):
                bidirectional += 1

            status = record[
                "overall_status"
            ]

            statuses[status] += 1

            balance = record[
                "balance_status"
            ]

            if balance == "receptive_weaker":
                receptive_weaker += 1
            elif balance == "productive_weaker":
                productive_weaker += 1
            elif balance == "balanced":
                balanced += 1

        total_signs = len(signs)

        divisor = max(total_signs, 1)

        return MasteryOverviewResponse(
            total_signs=total_signs,
            attempted_signs=attempted_signs,
            unattempted_signs=(
                total_signs
                - attempted_signs
            ),
            receptive_attempted=(
                receptive_attempted
            ),
            productive_attempted=(
                productive_attempted
            ),
            bidirectionally_attempted=(
                bidirectional
            ),
            average_receptive_score=round(
                receptive_total / divisor,
                4,
            ),
            average_productive_score=round(
                productive_total / divisor,
                4,
            ),
            average_combined_score=round(
                combined_total / divisor,
                4,
            ),
            statuses=MasteryStatusCounts(
                **statuses
            ),
            receptive_weaker_count=(
                receptive_weaker
            ),
            productive_weaker_count=(
                productive_weaker
            ),
            balanced_count=balanced,
        )

    async def list_sign_mastery(
        self,
        *,
        student_id: str,
        page: int,
        page_size: int,
        category_id: str | None,
        status_filter: str | None,
    ) -> PaginatedMasteryResponse:
        student_object_id = (
            self.repository.parse_object_id(
                student_id,
                message=(
                    "Student identifier "
                    "is invalid."
                ),
            )
        )

        signs = await (
            self.repository.list_visible_signs(
                visible_statuses=(
                    self._visible_statuses()
                ),
                category_id=category_id,
            )
        )

        records = await (
            self.repository
            .get_records_for_student(
                student_id=(
                    student_object_id
                )
            )
        )

        record_map = {
            str(record["sign_id"]): record
            for record in records
        }

        items = []

        for sign in signs:
            record = record_map.get(
                str(sign["_id"])
            )

            response = (
                self._to_sign_response(
                    sign=sign,
                    record=record,
                )
            )

            if (
                status_filter is not None
                and response.overall_status
                != status_filter
            ):
                continue

            items.append(response)

        total_items = len(items)

        start = (page - 1) * page_size
        end = start + page_size

        total_pages = (
            ceil(total_items / page_size)
            if total_items > 0
            else 0
        )

        return PaginatedMasteryResponse(
            items=items[start:end],
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
        )

    async def get_sign_mastery(
        self,
        *,
        student_id: str,
        sign_id: str,
    ) -> SignMasteryResponse:
        student_object_id = (
            self.repository.parse_object_id(
                student_id,
                message=(
                    "Student identifier "
                    "is invalid."
                ),
            )
        )

        sign = await (
            self.repository.get_visible_sign(
                sign_id=sign_id,
                visible_statuses=(
                    self._visible_statuses()
                ),
            )
        )

        record = await (
            self.repository.get_record(
                student_id=(
                    student_object_id
                ),
                sign_id=sign["_id"],
            )
        )

        return self._to_sign_response(
            sign=sign,
            record=record,
        )

    def _recognition_confidence(
        self,
        question: dict,
    ) -> float | None:
        if (
            question["direction"]
            != "productive"
        ):
            return None

        answer_interactions = [
            interaction
            for interaction in question[
                "interactions"
            ]
            if (
                interaction["event_type"]
                == "answer"
                and interaction[
                    "recognition_result"
                ]
                is not None
            )
        ]

        if not answer_interactions:
            return None

        latest = answer_interactions[-1]

        return float(
            latest[
                "recognition_result"
            ]["confidence"]
        )

    def _build_event_document(
        self,
        *,
        session_document: dict,
        question: dict,
    ) -> dict:
        confidence = (
            self._recognition_confidence(
                question
            )
        )

        quality = calculate_quality_score(
            direction=question["direction"],
            question_status=question[
                "status"
            ],
            final_correct=question[
                "final_correct"
            ],
            retry_count=question[
                "retry_count"
            ],
            hint_count=question[
                "hint_count"
            ],
            response_time_ms=question[
                "final_response_time_ms"
            ],
            difficulty=question[
                "prompt_snapshot"
            ]["difficulty"],
            recognition_confidence=(
                confidence
            ),
            productive_acceptance_threshold=(
                self.settings
                .productive_acceptance_confidence
            ),
        )

        return {
            "event_id": question[
                "mastery_event_id"
            ],
            "status": "pending",
            "student_id": session_document[
                "student_id"
            ],
            "sign_id": question["sign_id"],
            "session_id": session_document[
                "_id"
            ],
            "question_id": question[
                "question_id"
            ],
            "direction": question[
                "direction"
            ],
            "question_status": question[
                "status"
            ],
            "final_correct": question[
                "final_correct"
            ],
            "retry_count": question[
                "retry_count"
            ],
            "hint_count": question[
                "hint_count"
            ],
            "response_time_ms": question[
                "final_response_time_ms"
            ],
            "recognition_confidence": (
                confidence
            ),
            "difficulty": question[
                "prompt_snapshot"
            ]["difficulty"],
            "quality_score": (
                quality.score
            ),
            "quality_band": (
                quality.band
            ),
            "quality_components": (
                quality.to_components()
            ),
            "algorithm_version": (
                self.settings
                .mastery_algorithm_version
            ),
            "scheduler_algorithm_version": (
                self.settings
                .sm2_algorithm_version
            ),
            "before_state": None,
            "after_state": None,
            "created_at": datetime.now(
                timezone.utc
            ),
            "applied_at": None,
        }

    def _to_direction_response(
        self,
        state: dict,
    ) -> DirectionMasteryResponse:
        now = datetime.now(timezone.utc)
        next_review_at = state[
            "next_review_at"
        ]

        if (
            state["total_reviews"] == 0
            or next_review_at is None
        ):
            is_due = True
            days_overdue = 0.0
        else:
            if next_review_at.tzinfo is None:
                next_review_at = (
                    next_review_at.replace(
                        tzinfo=timezone.utc
                    )
                )
            difference = (
                now - next_review_at
            ).total_seconds() / 86400
            is_due = difference >= 0
            days_overdue = round(
                max(difference, 0.0),
                2,
            )

        return DirectionMasteryResponse(
            score=float(
                state["score"]
            ),
            status=state["status"],
            total_reviews=state[
                "total_reviews"
            ],
            successful_reviews=state[
                "successful_reviews"
            ],
            independent_successes=state[
                "independent_successes"
            ],
            assisted_successes=state[
                "assisted_successes"
            ],
            failure_count=state[
                "failure_count"
            ],
            average_quality_score=float(
                state[
                    "average_quality_score"
                ]
            ),
            last_quality_score=state[
                "last_quality_score"
            ],
            last_quality_band=state[
                "last_quality_band"
            ],
            average_response_time_ms=state[
                "average_response_time_ms"
            ],
            average_recognition_confidence=(
                state[
                    "average_recognition_confidence"
                ]
            ),
            last_recognition_confidence=(
                state[
                    "last_recognition_confidence"
                ]
            ),
            last_reviewed_at=state[
                "last_reviewed_at"
            ],
            ease_factor=float(
                state["ease_factor"]
            ),
            previous_interval_days=state[
                "previous_interval_days"
            ],
            interval_days=state[
                "interval_days"
            ],
            base_sm2_interval_days=state["base_sm2_interval_days"],
            last_ml_probability=state["last_ml_probability"],
            last_ml_adjustment_factor=state["last_ml_adjustment_factor"],
            last_ml_model_version=state["last_ml_model_version"],
            last_ml_status=state["last_ml_status"],
            last_ml_event_id=state["last_ml_event_id"],
            last_ml_applied_at=state["last_ml_applied_at"],
            repetition_count=state[
                "repetition_count"
            ],
            lapse_count=state[
                "lapse_count"
            ],
            next_review_at=next_review_at,
            last_scheduled_at=state[
                "last_scheduled_at"
            ],
            last_schedule_reason=state[
                "last_schedule_reason"
            ],
            is_due=is_due,
            days_overdue=days_overdue,
        )

    def _to_sign_response(
        self,
        *,
        sign: dict,
        record: dict | None,
    ) -> SignMasteryResponse:
        if record is None:
            receptive = (
                initial_direction_state()
            )

            productive = (
                initial_direction_state()
            )

            combined_score = 0.0
            overall_status = "new"
            balance_status = "unassessed"
            updated_at = None
        else:
            receptive = record[
                "receptive"
            ]

            productive = record[
                "productive"
            ]

            combined_score = record[
                "combined_score"
            ]

            overall_status = record[
                "overall_status"
            ]

            balance_status = record[
                "balance_status"
            ]

            updated_at = record[
                "updated_at"
            ]

        attempted = (
            receptive["total_reviews"] > 0
            or productive[
                "total_reviews"
            ] > 0
        )

        return SignMasteryResponse(
            sign_id=str(sign["_id"]),
            code=sign["code"],
            gloss=sign["gloss"],
            meanings=sign["meanings"],
            category_id=str(
                sign["category_id"]
            ),
            difficulty=sign[
                "difficulty"
            ],
            receptive=(
                self._to_direction_response(
                    receptive
                )
            ),
            productive=(
                self._to_direction_response(
                    productive
                )
            ),
            combined_score=float(
                combined_score
            ),
            overall_status=(
                overall_status
            ),
            balance_status=(
                balance_status
            ),
            attempted=attempted,
            updated_at=updated_at,
        )
