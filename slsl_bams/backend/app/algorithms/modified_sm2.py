from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.core.config import Settings


@dataclass(frozen=True)
class ScheduleUpdate:
    ease_factor: float
    previous_interval_days: int
    interval_days: int
    repetition_count: int
    lapse_count: int
    next_review_at: datetime
    last_scheduled_at: datetime
    last_schedule_reason: str


def _updated_ease_factor(
    *,
    current_ease: float,
    quality_band: int,
    settings: Settings,
) -> float:
    """
    Standard SM-2 ease-factor adjustment.

    EF' = EF + [0.1 - (5-q)(0.08 + (5-q)0.02)]
    """

    quality_difference = 5 - quality_band

    updated = current_ease + (
        0.10
        - quality_difference
        * (
            0.08
            + quality_difference * 0.02
        )
    )

    return round(
        max(
            settings.sm2_min_ease_factor,
            min(
                settings.sm2_max_ease_factor,
                updated,
            ),
        ),
        4,
    )


def _difficulty_multiplier(
    difficulty: int,
) -> float:
    multipliers = {
        1: 1.15,
        2: 1.08,
        3: 1.00,
        4: 0.92,
        5: 0.85,
    }

    return multipliers.get(
        difficulty,
        1.00,
    )


def _mastery_multiplier(
    mastery_score: float,
) -> float:
    """
    Lower mastery produces shorter intervals.

    Mastery 0.0 -> 0.80
    Mastery 0.5 -> 1.00
    Mastery 1.0 -> 1.20
    """

    score = max(
        0.0,
        min(1.0, mastery_score),
    )

    return 0.80 + (score * 0.40)


def update_sm2_schedule(
    *,
    current_state: dict[str, Any],
    quality_band: int,
    mastery_score: float,
    difficulty: int,
    reviewed_at: datetime,
    settings: Settings,
) -> ScheduleUpdate:
    current_ease = float(
        current_state.get(
            "ease_factor",
            settings.sm2_initial_ease_factor,
        )
    )

    current_interval = int(
        current_state.get(
            "interval_days",
            0,
        )
    )

    current_repetition = int(
        current_state.get(
            "repetition_count",
            0,
        )
    )

    current_lapses = int(
        current_state.get(
            "lapse_count",
            0,
        )
    )

    updated_ease = _updated_ease_factor(
        current_ease=current_ease,
        quality_band=quality_band,
        settings=settings,
    )

    previous_interval = current_interval

    if quality_band < 3:
        interval_days = 1
        repetition_count = 0
        lapse_count = current_lapses + 1
        reason = "lapse"

    else:
        lapse_count = current_lapses

        if current_repetition == 0:
            base_interval = 1
            reason = "first_success"

        elif current_repetition == 1:
            base_interval = 6
            reason = "second_success"

        else:
            base_interval = max(
                1,
                round(
                    max(
                        current_interval,
                        1,
                    )
                    * updated_ease
                ),
            )

            if quality_band == 3:
                reason = "assisted_success"
            elif quality_band == 5:
                reason = "strong_success"
            else:
                reason = "successful_review"

        interval_modifier = (
            _difficulty_multiplier(
                difficulty
            )
            * _mastery_multiplier(
                mastery_score
            )
        )

        if quality_band == 3:
            interval_modifier *= 0.75

        interval_days = max(
            1,
            round(
                base_interval
                * interval_modifier
            ),
        )

        interval_days = min(
            interval_days,
            settings.sm2_max_interval_days,
        )

        repetition_count = (
            current_repetition + 1
        )

    next_review_at = (
        reviewed_at
        + timedelta(
            days=interval_days
        )
    )

    return ScheduleUpdate(
        ease_factor=updated_ease,
        previous_interval_days=(
            previous_interval
        ),
        interval_days=interval_days,
        repetition_count=(
            repetition_count
        ),
        lapse_count=lapse_count,
        next_review_at=next_review_at,
        last_scheduled_at=reviewed_at,
        last_schedule_reason=reason,
    )


def apply_schedule_to_state(
    *,
    state: dict[str, Any],
    schedule: ScheduleUpdate,
) -> dict[str, Any]:
    state["ease_factor"] = (
        schedule.ease_factor
    )

    state["previous_interval_days"] = (
        schedule.previous_interval_days
    )

    state["interval_days"] = (
        schedule.interval_days
    )

    state["repetition_count"] = (
        schedule.repetition_count
    )

    state["lapse_count"] = (
        schedule.lapse_count
    )

    state["next_review_at"] = (
        schedule.next_review_at
    )

    state["last_scheduled_at"] = (
        schedule.last_scheduled_at
    )

    state["last_schedule_reason"] = (
        schedule.last_schedule_reason
    )

    return state