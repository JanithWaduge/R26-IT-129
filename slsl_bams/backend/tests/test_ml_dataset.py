import json
from datetime import datetime, timedelta, timezone

from app.ml.dataset_schema import FEATURE_COLUMNS, TARGET_COLUMN
from app.ml.pseudonym import Pseudonymizer
from app.ml.recall_dataset import RecallDatasetBuilder

SECRET = "test-dataset-pseudonym-secret-with-more-than-thirty-two-chars"


def make_event(*, event_id: str, student_id: str, sign_id: str,
               completed_at: datetime, correct: bool | None,
               status: str, quality: float) -> dict:
    return {
        "event_id": event_id, "student_id": student_id, "sign_id": sign_id,
        "session_id": f"session-{event_id}",
        "question_id": f"question-{event_id}", "completed_at": completed_at,
        "question_status": status, "final_correct": correct,
        "quality_score": quality, "quality_band": round(quality),
        "response_time_ms": 4000, "retry_count": 0, "hint_count": 0,
        "difficulty": 2, "category_code": "GREETINGS",
        "prompt_language": "english", "selection_strategy": "adaptive",
        "selection_priority": 64.0, "selection_reason": "due_today",
        "was_due": True, "days_overdue": 0.0,
        "before_state": {
            "score": 0.30, "total_reviews": 1, "successful_reviews": 1,
            "failure_count": 0, "average_quality_score": 4.0,
            "average_response_time_ms": 4500,
            "last_reviewed_at": completed_at - timedelta(days=2),
            "ease_factor": 2.5, "interval_days": 1,
            "repetition_count": 1, "lapse_count": 0,
        },
        "after_state": {
            "score": 0.50, "total_reviews": 2,
            "successful_reviews": 2 if correct else 1,
            "failure_count": 0 if correct else 1,
            "average_quality_score": quality,
            "average_response_time_ms": 4200, "last_reviewed_at": completed_at,
            "ease_factor": 2.6, "interval_days": 6,
            "repetition_count": 2, "lapse_count": 0,
            "next_review_at": completed_at + timedelta(days=6),
        },
    }


def build_dataset(events: list[dict], *, now: datetime):
    return RecallDatasetBuilder(
        pseudonymizer=Pseudonymizer(SECRET),
        min_delay=timedelta(hours=24), max_delay=timedelta(days=30),
    ).build(events, now=now)


def test_first_eligible_delayed_attempt_is_label() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        make_event(event_id="anchor", student_id="raw-student-1",
                   sign_id="raw-sign-1", completed_at=start,
                   correct=True, status="correct", quality=5.0),
        make_event(event_id="too-soon", student_id="raw-student-1",
                   sign_id="raw-sign-1", completed_at=start + timedelta(hours=2),
                   correct=True, status="correct", quality=5.0),
        make_event(event_id="eligible-target", student_id="raw-student-1",
                   sign_id="raw-sign-1", completed_at=start + timedelta(hours=30),
                   correct=False, status="incorrect", quality=1.0),
    ]
    result = build_dataset(events, now=start + timedelta(days=40))
    pseudonymizer = Pseudonymizer(SECRET)
    row_id = pseudonymizer.row("anchor")
    row = next(item for item in result.labeled_rows if item["row_id"] == row_id)
    assert row[TARGET_COLUMN] == 0
    audit = next(item for item in result.audit_rows if item["row_id"] == row_id)
    assert audit["label_delay_hours"] == 30.0
    assert audit["target_event_id"] == pseudonymizer.event("eligible-target")


def test_future_target_never_appears_in_features() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert all(not column.startswith("target_") for column in FEATURE_COLUMNS)
    assert all(not column.startswith("future_") for column in FEATURE_COLUMNS)


def test_raw_database_ids_are_not_exported() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    student, sign = "raw-sensitive-student-id", "raw-database-sign-id"
    events = [
        make_event(event_id="privacy-anchor", student_id=student, sign_id=sign,
                   completed_at=start, correct=True, status="correct", quality=5.0),
        make_event(event_id="privacy-target", student_id=student, sign_id=sign,
                   completed_at=start + timedelta(days=2), correct=True,
                   status="correct", quality=5.0),
    ]
    result = build_dataset(events, now=start + timedelta(days=40))
    serialized = json.dumps({"labeled": result.labeled_rows,
                             "audit": result.audit_rows,
                             "unlabeled": result.unlabeled_rows}, default=str)
    assert student not in serialized
    assert sign not in serialized


def test_same_learner_stays_in_one_split() -> None:
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    events = []
    for sign, delay in (("sign-1", 2), ("sign-2", 3)):
        events.extend([
            make_event(event_id=f"{sign}-a", student_id="same-student", sign_id=sign,
                       completed_at=start, correct=True, status="correct", quality=5.0),
            make_event(event_id=f"{sign}-b", student_id="same-student", sign_id=sign,
                       completed_at=start + timedelta(days=delay), correct=True,
                       status="correct", quality=4.0),
        ])
    result = build_dataset(events, now=start + timedelta(days=40))
    assert len({row["split"] for row in result.labeled_rows}) == 1


def test_recent_anchor_is_not_matured() -> None:
    now = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)
    event = make_event(event_id="recent-anchor", student_id="student-recent",
                       sign_id="sign-recent", completed_at=now - timedelta(hours=5),
                       correct=True, status="correct", quality=5.0)
    result = build_dataset([event], now=now)
    assert not result.labeled_rows
    assert result.unlabeled_rows[0]["unlabeled_reason"] == "not_matured"


def test_old_anchor_without_followup_expires() -> None:
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    event = make_event(event_id="expired-anchor", student_id="student-expired",
                       sign_id="sign-expired", completed_at=now - timedelta(days=45),
                       correct=True, status="correct", quality=5.0)
    result = build_dataset([event], now=now)
    assert result.unlabeled_rows[0]["unlabeled_reason"] == "expired_without_followup"
