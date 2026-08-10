from datetime import datetime, timezone
from typing import Any


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def build_live_recall_features(
    *, event: dict, session: dict, question: dict, category_code: str,
    learner_history: dict,
) -> dict[str, Any]:
    before, after = event.get("before_state") or {}, event.get("after_state") or {}
    completed = (_datetime(question.get("completed_at"))
                 or _datetime(event.get("applied_at")) or datetime.now(timezone.utc))
    previous = _datetime(before.get("last_reviewed_at"))
    hours_since = (None if previous is None else round(
        max((completed - previous).total_seconds() / 3600, 0.0), 4
    ))
    base_interval = int(after.get("interval_days", 1) or 1)
    status = str(event.get("question_status", question.get("status", "incorrect")))
    prompt = question.get("prompt_snapshot", {})
    return {
        "category_code": category_code,
        "prompt_language": str(session.get("prompt_language", "english")),
        "selection_strategy": str(session.get("selection_strategy", "legacy_random")),
        "selection_reason": str(question.get("selection_reason", "legacy_random")),
        "anchor_outcome": status,
        "anchor_correct": int(event.get("final_correct") is True),
        "anchor_was_skipped": int(status == "skipped"),
        "anchor_quality_score": _number(event.get("quality_score")),
        "anchor_quality_band": int(event.get("quality_band", 0) or 0),
        "response_time_ms": event.get("response_time_ms"),
        "retry_count": int(event.get("retry_count", 0)),
        "hint_count": int(event.get("hint_count", 0)),
        "sign_difficulty": int(event.get("difficulty", prompt.get("difficulty", 1))),
        "was_due": int(bool(question.get("was_due", False))),
        "days_overdue": _number(question.get("days_overdue")),
        "selection_priority": _number(question.get("selection_priority")),
        "pre_mastery_score": _number(before.get("score")),
        "post_mastery_score": _number(after.get("score")),
        "pre_total_reviews": int(before.get("total_reviews", 0)),
        "pre_successful_reviews": int(before.get("successful_reviews", 0)),
        "pre_failure_count": int(before.get("failure_count", 0)),
        "pre_average_quality_score": _number(before.get("average_quality_score")),
        "pre_average_response_time_ms": before.get("average_response_time_ms"),
        "hours_since_previous_review": hours_since,
        "pre_ease_factor": _number(before.get("ease_factor"), 2.5),
        "post_ease_factor": _number(after.get("ease_factor"), 2.5),
        "pre_interval_days": int(before.get("interval_days", 0)),
        "post_interval_days": base_interval,
        "pre_repetition_count": int(before.get("repetition_count", 0)),
        "post_repetition_count": int(after.get("repetition_count", 0)),
        "pre_lapse_count": int(before.get("lapse_count", 0)),
        "post_lapse_count": int(after.get("lapse_count", 0)),
        "scheduled_delay_hours": float(base_interval * 24),
        "learner_prior_attempts": int(learner_history.get("attempts", 0)),
        "learner_prior_accuracy": _number(learner_history.get("accuracy")),
        "learner_prior_average_quality": _number(learner_history.get("average_quality")),
    }
