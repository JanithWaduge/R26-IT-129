from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.evaluate_scheduler_experiment import (
    bootstrap_difference, build_delayed_labels, prediction_metrics, summarize_arm,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def event(identifier: str, *, hours: int, arm: str = "hybrid_ml", correct: int = 1):
    return {
        "event_id": identifier, "student_id": "student-1", "sign_id": "sign-1",
        "arm": arm, "completed_at": NOW + timedelta(hours=hours), "correct": correct,
        "quality_score": 4.0, "response_time_ms": 3000,
        "scheduled_delay_hours": 48.0,
    }


def labels(events):
    return build_delayed_labels(
        events, minimum_delay=timedelta(hours=24), maximum_delay=timedelta(days=30)
    )


def test_attempts_under_24_hours_are_ignored() -> None:
    assert not labels([event("a", hours=0), event("b", hours=5)])


def test_attempts_after_30_days_are_ignored() -> None:
    assert not labels([event("a", hours=0), event("b", hours=31 * 24)])


def test_first_eligible_attempt_is_label() -> None:
    result = labels([event("a", hours=0), event("early", hours=2),
                     event("target", hours=30, correct=0), event("later", hours=50)])
    assert result[0]["target_event_id"] == "target"
    assert result[0]["future_recall_success"] == 0


def test_arm_changes_are_excluded() -> None:
    assert not labels([event("a", hours=0, arm="hybrid_ml"),
                       event("b", hours=30, arm="modified_sm2")])


def test_retention_summary_and_bootstrap() -> None:
    frame = pd.DataFrame([
        {**event("a", hours=0, arm="hybrid_ml"), "future_recall_success": 1},
        {**event("b", hours=0, arm="hybrid_ml"), "student_id": "student-2",
         "future_recall_success": 0},
        {**event("c", hours=0, arm="modified_sm2"), "student_id": "student-3",
         "future_recall_success": 0},
    ])
    summary = summarize_arm(frame[frame.arm == "hybrid_ml"], frame)
    assert summary["retention_rate"] == 0.5
    comparison = bootstrap_difference(
        frame, first_arm="hybrid_ml", second_arm="modified_sm2", iterations=100
    )
    assert comparison["ci_95_low"] is not None
    assert comparison["ci_95_high"] is not None


def test_brier_uses_probabilities_and_synthetic_is_identifiable() -> None:
    frame = pd.DataFrame({
        "future_recall_success": [1, 0], "recall_probability": [0.8, 0.2],
        "synthetic_model": [True, True],
    })
    metrics = prediction_metrics(frame)
    assert round(metrics["brier_score"], 2) == 0.04
    assert metrics["synthetic_predictions"] is True
