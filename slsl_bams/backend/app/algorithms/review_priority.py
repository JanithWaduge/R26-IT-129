from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ReviewPriorityResult:
    score: float
    is_due: bool
    days_overdue: float
    reason: str

    due_component: float
    mastery_component: float
    failure_component: float
    direction_gap_component: float
    difficulty_component: float
    curriculum_component: float


def _normalise_datetime(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


def calculate_review_priority(
    *,
    direction_state: dict[str, Any],
    opposite_state: dict[str, Any],
    difficulty: int,
    curriculum_weight: float,
    now: datetime,
) -> ReviewPriorityResult:
    total_reviews = int(
        direction_state["total_reviews"]
    )

    score = float(
        direction_state["score"]
    )

    next_review_at = direction_state.get(
        "next_review_at"
    )

    if (
        total_reviews == 0
        or next_review_at is None
    ):
        is_due = True
        days_overdue = 0.0
        due_component = 30.0
        reason = "new_direction"

    else:
        normalised_due = (
            _normalise_datetime(
                next_review_at
            )
        )

        normalised_now = (
            _normalise_datetime(now)
        )

        days_overdue = (
            normalised_now
            - normalised_due
        ).total_seconds() / 86400

        is_due = days_overdue >= 0

        if is_due:
            due_component = min(
                45.0,
                25.0
                + max(
                    days_overdue,
                    0.0,
                )
                * 3.0,
            )

            reason = (
                "overdue"
                if days_overdue >= 1
                else "due_today"
            )

        else:
            due_component = 0.0
            reason = "upcoming_review"

    mastery_component = (
        1.0 - score
    ) * 25.0

    failure_rate = (
        direction_state["failure_count"]
        / max(total_reviews, 1)
    )

    failure_component = min(
        15.0,
        failure_rate * 10.0
        + direction_state[
            "lapse_count"
        ]
        * 2.0,
    )

    opposite_score = float(
        opposite_state["score"]
    )

    direction_gap_component = min(
        10.0,
        max(
            0.0,
            opposite_score - score,
        )
        * 20.0,
    )

    difficulty_component = max(
        0.0,
        min(
            7.5,
            (difficulty - 1) * 1.875,
        ),
    )

    curriculum_component = max(
        -5.0,
        min(
            10.0,
            (
                curriculum_weight
                - 1.0
            )
            * 10.0,
        ),
    )

    priority_score = (
        due_component
        + mastery_component
        + failure_component
        + direction_gap_component
        + difficulty_component
        + curriculum_component
    )

    priority_score = round(
        max(
            0.0,
            min(
                100.0,
                priority_score,
            ),
        ),
        2,
    )

    if (
        reason == "upcoming_review"
        and direction_gap_component >= 4
    ):
        reason = "weaker_direction"

    elif (
        reason == "upcoming_review"
        and score < 0.50
    ):
        reason = "low_mastery"

    return ReviewPriorityResult(
        score=priority_score,
        is_due=is_due,
        days_overdue=round(
            max(days_overdue, 0.0),
            2,
        ),
        reason=reason,
        due_component=round(
            due_component,
            2,
        ),
        mastery_component=round(
            mastery_component,
            2,
        ),
        failure_component=round(
            failure_component,
            2,
        ),
        direction_gap_component=round(
            direction_gap_component,
            2,
        ),
        difficulty_component=round(
            difficulty_component,
            2,
        ),
        curriculum_component=round(
            curriculum_component,
            2,
        ),
    )