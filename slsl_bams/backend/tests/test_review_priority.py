from datetime import datetime, timedelta, timezone

from app.algorithms.mastery_update import initial_direction_state
from app.algorithms.review_priority import calculate_review_priority


def test_new_direction_is_due() -> None:
    result = calculate_review_priority(
        direction_state=initial_direction_state(),
        opposite_state=initial_direction_state(),
        difficulty=1,
        curriculum_weight=1.0,
        now=datetime.now(timezone.utc),
    )
    assert result.is_due is True
    assert result.reason == "new_direction"


def test_overdue_review_has_high_priority() -> None:
    now = datetime.now(timezone.utc)
    state = initial_direction_state()
    state.update(total_reviews=3, score=0.40, next_review_at=now - timedelta(days=5))
    opposite = initial_direction_state()
    opposite["score"] = 0.80
    result = calculate_review_priority(direction_state=state, opposite_state=opposite, difficulty=4, curriculum_weight=1.5, now=now)
    assert result.is_due is True
    assert result.days_overdue >= 5
    assert result.score > 60


def test_future_review_is_not_due_and_gap_is_prioritized() -> None:
    now = datetime.now(timezone.utc)
    weaker = initial_direction_state()
    weaker.update(total_reviews=3, score=0.30, next_review_at=now + timedelta(days=2))
    stronger = initial_direction_state()
    stronger.update(total_reviews=3, score=0.90, next_review_at=now + timedelta(days=2))
    result = calculate_review_priority(direction_state=weaker, opposite_state=stronger, difficulty=3, curriculum_weight=1.0, now=now)
    assert result.is_due is False
    assert result.direction_gap_component > 0
