from datetime import datetime, timezone

import pytest

from app.algorithms.mastery_update import (
    calculate_combined_state,
    initial_direction_state,
    mastery_status,
    update_direction_state,
)
from app.algorithms.quality_score import (
    calculate_quality_score,
)
from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_name="Test",
        app_version="0.7.0",
        environment="test",
        api_v1_prefix="/api/v1",
        mongo_uri=(
            "mongodb://127.0.0.1:27017"
        ),
        mongo_database="test",
        jwt_access_secret="a" * 128,
        jwt_refresh_secret="b" * 128,
        mastery_algorithm_version=(
            "bams-mastery-v1"
        ),
    )


def test_skipped_question_quality_is_zero() -> None:
    result = calculate_quality_score(
        direction="receptive",
        question_status="skipped",
        final_correct=None,
        retry_count=0,
        hint_count=0,
        response_time_ms=None,
        difficulty=1,
        recognition_confidence=None,
        productive_acceptance_threshold=0.70,
    )

    assert result.score == 0.0
    assert result.band == 0


def test_independent_correct_answer_has_high_quality() -> None:
    result = calculate_quality_score(
        direction="receptive",
        question_status="correct",
        final_correct=True,
        retry_count=0,
        hint_count=0,
        response_time_ms=3000,
        difficulty=1,
        recognition_confidence=None,
        productive_acceptance_threshold=0.70,
    )

    assert result.score == 5.0
    assert result.band == 5


def test_assistance_reduces_quality() -> None:
    result = calculate_quality_score(
        direction="receptive",
        question_status="correct",
        final_correct=True,
        retry_count=2,
        hint_count=1,
        response_time_ms=20000,
        difficulty=1,
        recognition_confidence=None,
        productive_acceptance_threshold=0.70,
    )

    assert result.score < 5.0
    assert result.retry_penalty > 0
    assert result.hint_penalty > 0


def test_productive_confidence_affects_quality() -> None:
    high = calculate_quality_score(
        direction="productive",
        question_status="correct",
        final_correct=True,
        retry_count=0,
        hint_count=0,
        response_time_ms=5000,
        difficulty=1,
        recognition_confidence=0.95,
        productive_acceptance_threshold=0.70,
    )

    lower = calculate_quality_score(
        direction="productive",
        question_status="correct",
        final_correct=True,
        retry_count=0,
        hint_count=0,
        response_time_ms=5000,
        difficulty=1,
        recognition_confidence=0.72,
        productive_acceptance_threshold=0.70,
    )

    assert high.score > lower.score


def test_first_high_quality_review_increases_mastery(
    settings: Settings,
) -> None:
    state = initial_direction_state()

    updated = update_direction_state(
        current_state=state,
        quality_score=5.0,
        quality_band=5,
        final_correct=True,
        retry_count=0,
        hint_count=0,
        response_time_ms=3000,
        recognition_confidence=None,
        reviewed_at=datetime.now(
            timezone.utc
        ),
        session_id=None,
        question_id=(
            "00000000-0000-4000-8000-000000000001"
        ),
        settings=settings,
    )

    assert updated["score"] == 0.45
    assert updated["total_reviews"] == 1
    assert (
        updated["independent_successes"]
        == 1
    )


def test_failure_reduces_existing_mastery(
    settings: Settings,
) -> None:
    state = initial_direction_state()
    state["score"] = 0.80
    state["status"] = "proficient"
    state["total_reviews"] = 6

    updated = update_direction_state(
        current_state=state,
        quality_score=1.0,
        quality_band=1,
        final_correct=False,
        retry_count=2,
        hint_count=0,
        response_time_ms=10000,
        recognition_confidence=None,
        reviewed_at=datetime.now(
            timezone.utc
        ),
        session_id=None,
        question_id=(
            "00000000-0000-4000-8000-000000000002"
        ),
        settings=settings,
    )

    assert updated["score"] < 0.80
    assert updated["failure_count"] == 1


def test_combined_mastery_requires_both_directions() -> None:
    receptive = initial_direction_state()
    productive = initial_direction_state()

    receptive["score"] = 0.90
    receptive["total_reviews"] = 8
    receptive["status"] = "mastered"

    combined, status, balance = (
        calculate_combined_state(
            receptive=receptive,
            productive=productive,
        )
    )

    assert combined == 0.0
    assert status == "new"
    assert balance == (
        "productive_unassessed"
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.10, "very_weak"),
        (0.35, "weak"),
        (0.55, "learning"),
        (0.75, "proficient"),
        (0.90, "mastered"),
    ],
)
def test_mastery_status_thresholds(
    score: float,
    expected: str,
) -> None:
    assert mastery_status(
        score=score,
        total_reviews=1,
    ) == expected
