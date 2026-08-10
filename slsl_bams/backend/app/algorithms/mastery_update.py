from copy import deepcopy
from datetime import datetime
from typing import Any

from app.core.config import Settings


def initial_direction_state() -> dict[str, Any]:
    return {
        "score": 0.0,
        "status": "new",
        "total_reviews": 0,
        "successful_reviews": 0,
        "independent_successes": 0,
        "assisted_successes": 0,
        "failure_count": 0,
        "average_quality_score": 0.0,
        "last_quality_score": None,
        "last_quality_band": None,
        "response_samples": 0,
        "average_response_time_ms": None,
        "confidence_samples": 0,
        "average_recognition_confidence": None,
        "last_recognition_confidence": None,
        "last_reviewed_at": None,
        "last_session_id": None,
        "last_question_id": None,

        # Modified SM-2 scheduling state
        "ease_factor": 2.50,
        "previous_interval_days": 0,
        "interval_days": 0,
        "repetition_count": 0,
        "lapse_count": 0,
        "next_review_at": None,
        "last_scheduled_at": None,
        "last_schedule_reason": "unassessed",
        "base_sm2_interval_days": 0,
        "last_ml_probability": None,
        "last_ml_adjustment_factor": None,
        "last_ml_model_version": None,
        "last_ml_status": "not_evaluated",
        "last_ml_event_id": None,
        "last_ml_applied_at": None,
    }


def mastery_status(
    *,
    score: float,
    total_reviews: int,
) -> str:
    if total_reviews == 0:
        return "new"

    if score < 0.30:
        return "very_weak"

    if score < 0.50:
        return "weak"

    if score < 0.70:
        return "learning"

    if score < 0.85:
        return "proficient"

    return "mastered"


def _learning_rate(
    *,
    total_reviews: int,
    final_correct: bool | None,
    settings: Settings,
) -> float:
    if final_correct is not True:
        return (
            settings
            .mastery_failure_learning_rate
        )

    if total_reviews < 3:
        return (
            settings
            .mastery_initial_learning_rate
        )

    if total_reviews < 10:
        return (
            settings
            .mastery_intermediate_learning_rate
        )

    return (
        settings
        .mastery_stable_learning_rate
    )


def _running_average(
    *,
    previous_average: float | None,
    previous_count: int,
    new_value: float,
) -> float:
    if previous_count == 0:
        return round(new_value, 4)

    assert previous_average is not None

    updated = (
        (
            previous_average
            * previous_count
        )
        + new_value
    ) / (previous_count + 1)

    return round(updated, 4)


def update_direction_state(
    *,
    current_state: dict[str, Any],
    quality_score: float,
    quality_band: int,
    final_correct: bool | None,
    retry_count: int,
    hint_count: int,
    response_time_ms: int | None,
    recognition_confidence: float | None,
    reviewed_at: datetime,
    session_id: Any,
    question_id: str,
    settings: Settings,
) -> dict[str, Any]:
    state = deepcopy(current_state)

    previous_total = state[
        "total_reviews"
    ]

    evidence = max(
        0.0,
        min(1.0, quality_score / 5.0),
    )

    learning_rate = _learning_rate(
        total_reviews=previous_total,
        final_correct=final_correct,
        settings=settings,
    )

    previous_score = float(
        state["score"]
    )

    updated_score = (
        previous_score
        + learning_rate
        * (evidence - previous_score)
    )

    updated_score = round(
        max(0.0, min(1.0, updated_score)),
        4,
    )

    state["score"] = updated_score
    state["total_reviews"] = (
        previous_total + 1
    )

    state["average_quality_score"] = (
        _running_average(
            previous_average=float(
                state[
                    "average_quality_score"
                ]
            ),
            previous_count=previous_total,
            new_value=quality_score,
        )
    )

    state["last_quality_score"] = (
        quality_score
    )

    state["last_quality_band"] = (
        quality_band
    )

    if final_correct is True:
        state["successful_reviews"] += 1

        if (
            retry_count == 0
            and hint_count == 0
            and quality_score >= 4.0
        ):
            state[
                "independent_successes"
            ] += 1
        else:
            state[
                "assisted_successes"
            ] += 1

    else:
        state["failure_count"] += 1

    if response_time_ms is not None:
        previous_response_count = state[
            "response_samples"
        ]

        state[
            "average_response_time_ms"
        ] = _running_average(
            previous_average=state[
                "average_response_time_ms"
            ],
            previous_count=(
                previous_response_count
            ),
            new_value=float(
                response_time_ms
            ),
        )

        state["response_samples"] = (
            previous_response_count + 1
        )

    if recognition_confidence is not None:
        previous_confidence_count = state[
            "confidence_samples"
        ]

        state[
            "average_recognition_confidence"
        ] = _running_average(
            previous_average=state[
                "average_recognition_confidence"
            ],
            previous_count=(
                previous_confidence_count
            ),
            new_value=float(
                recognition_confidence
            ),
        )

        state["confidence_samples"] = (
            previous_confidence_count + 1
        )

        state[
            "last_recognition_confidence"
        ] = round(
            recognition_confidence,
            4,
        )

    state["last_reviewed_at"] = reviewed_at
    state["last_session_id"] = session_id
    state["last_question_id"] = question_id

    state["status"] = mastery_status(
        score=updated_score,
        total_reviews=state[
            "total_reviews"
        ],
    )

    return state


def calculate_combined_state(
    *,
    receptive: dict[str, Any],
    productive: dict[str, Any],
) -> tuple[float, str, str]:
    receptive_reviews = receptive[
        "total_reviews"
    ]

    productive_reviews = productive[
        "total_reviews"
    ]

    receptive_score = float(
        receptive["score"]
    )

    productive_score = float(
        productive["score"]
    )

    if (
        receptive_reviews == 0
        and productive_reviews == 0
    ):
        combined_score = 0.0
        overall_status = "new"
        balance_status = "unassessed"

        return (
            combined_score,
            overall_status,
            balance_status,
        )

    if receptive_reviews == 0:
        combined_score = 0.0
        overall_status = "new"
        balance_status = (
            "receptive_unassessed"
        )

        return (
            combined_score,
            overall_status,
            balance_status,
        )

    if productive_reviews == 0:
        combined_score = 0.0
        overall_status = "new"
        balance_status = (
            "productive_unassessed"
        )

        return (
            combined_score,
            overall_status,
            balance_status,
        )

    combined_score = round(
        min(
            receptive_score,
            productive_score,
        ),
        4,
    )

    overall_status = mastery_status(
        score=combined_score,
        total_reviews=(
            receptive_reviews
            + productive_reviews
        ),
    )

    difference = (
        receptive_score
        - productive_score
    )

    if abs(difference) < 0.10:
        balance_status = "balanced"
    elif difference < 0:
        balance_status = (
            "receptive_weaker"
        )
    else:
        balance_status = (
            "productive_weaker"
        )

    return (
        combined_score,
        overall_status,
        balance_status,
    )
