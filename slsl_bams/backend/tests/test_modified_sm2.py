from datetime import datetime, timezone

from app.algorithms.mastery_update import initial_direction_state
from app.algorithms.modified_sm2 import update_sm2_schedule
from app.core.config import Settings


def settings() -> Settings:
    return Settings(
        app_name="Test",
        app_version="0.7.0",
        environment="test",
        api_v1_prefix="/api/v1",
        mongo_uri="mongodb://127.0.0.1:27017",
        mongo_database="test",
        jwt_access_secret="a" * 128,
        jwt_refresh_secret="b" * 128,
        sm2_algorithm_version="bams-sm2-v1",
    )


def test_first_success_schedules_one_day() -> None:
    result = update_sm2_schedule(
        current_state=initial_direction_state(),
        quality_band=5,
        mastery_score=0.45,
        difficulty=3,
        reviewed_at=datetime.now(timezone.utc),
        settings=settings(),
    )
    assert result.interval_days == 1
    assert result.repetition_count == 1
    assert result.last_schedule_reason == "first_success"


def test_second_success_expands_interval() -> None:
    state = initial_direction_state()
    state.update(interval_days=1, repetition_count=1)
    result = update_sm2_schedule(
        current_state=state,
        quality_band=5,
        mastery_score=0.70,
        difficulty=3,
        reviewed_at=datetime.now(timezone.utc),
        settings=settings(),
    )
    assert result.interval_days >= 6
    assert result.repetition_count == 2


def test_lapse_resets_repetition() -> None:
    state = initial_direction_state()
    state.update(interval_days=20, repetition_count=4, lapse_count=1)
    result = update_sm2_schedule(
        current_state=state,
        quality_band=1,
        mastery_score=0.40,
        difficulty=3,
        reviewed_at=datetime.now(timezone.utc),
        settings=settings(),
    )
    assert result.interval_days == 1
    assert result.repetition_count == 0
    assert result.lapse_count == 2


def test_assistance_and_difficulty_shorten_intervals() -> None:
    state = initial_direction_state()
    state.update(interval_days=10, repetition_count=3)
    now = datetime.now(timezone.utc)
    assisted = update_sm2_schedule(current_state=state, quality_band=3, mastery_score=0.70, difficulty=3, reviewed_at=now, settings=settings())
    strong = update_sm2_schedule(current_state=state, quality_band=5, mastery_score=0.70, difficulty=3, reviewed_at=now, settings=settings())
    easy = update_sm2_schedule(current_state=state, quality_band=5, mastery_score=0.70, difficulty=1, reviewed_at=now, settings=settings())
    difficult = update_sm2_schedule(current_state=state, quality_band=5, mastery_score=0.70, difficulty=5, reviewed_at=now, settings=settings())
    assert assisted.interval_days < strong.interval_days
    assert difficult.interval_days < easy.interval_days
