from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.ml.dataset_schema import TARGET_COLUMN
from app.ml.pseudonym import Pseudonymizer


@dataclass(frozen=True)
class DatasetBuildResult:
    labeled_rows: list[dict]
    audit_rows: list[dict]
    unlabeled_rows: list[dict]
    source_event_count: int
    eligible_event_count: int


def _utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _float(state: dict, key: str, default: float = 0.0) -> float:
    try:
        value = state.get(key, default)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int(state: dict, key: str, default: int = 0) -> int:
    try:
        value = state.get(key, default)
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


class RecallDatasetBuilder:
    def __init__(self, *, pseudonymizer: Pseudonymizer,
                 min_delay: timedelta, max_delay: timedelta) -> None:
        if min_delay <= timedelta(0):
            raise ValueError("Minimum recall delay must be positive.")
        if max_delay <= min_delay:
            raise ValueError("Maximum recall delay must be greater than the minimum.")
        self.pseudonymizer = pseudonymizer
        self.min_delay = min_delay
        self.max_delay = max_delay

    def build(self, events: list[dict], *, now: datetime | None = None) -> DatasetBuildResult:
        resolved_now = _utc_datetime(now or datetime.now(timezone.utc))
        assert resolved_now is not None
        valid = [dict(event) for event in events if event.get("event_id")
                 and event.get("student_id") and event.get("sign_id")
                 and _utc_datetime(event.get("completed_at")) is not None]
        for event in valid:
            event["completed_at"] = _utc_datetime(event["completed_at"])
        enriched = self._add_learner_history(valid)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for event in enriched:
            groups[(event["student_id"], event["sign_id"])].append(event)

        labeled, audit, unlabeled = [], [], []
        for group in groups.values():
            group.sort(key=lambda item: (item["completed_at"], item["event_id"]))
            for index, anchor in enumerate(group):
                target = self._find_target(anchor, group[index + 1:])
                metadata = self._metadata(anchor)
                features = self._features(anchor)
                if target is None:
                    unlabeled.append({
                        **metadata, **features,
                        "anchor_event_id": self.pseudonymizer.event(anchor["event_id"]),
                        "anchor_completed_at": anchor["completed_at"].isoformat(),
                        "unlabeled_reason": self._unlabeled_reason(anchor, resolved_now),
                    })
                    continue
                success = int(target.get("final_correct") is True)
                labeled.append({**metadata, **features, TARGET_COLUMN: success})
                delay = (target["completed_at"] - anchor["completed_at"]).total_seconds() / 3600
                audit.append({
                    **metadata,
                    "anchor_event_id": self.pseudonymizer.event(anchor["event_id"]),
                    "anchor_session_id": self.pseudonymizer.session(anchor["session_id"]),
                    "anchor_question_id": self.pseudonymizer.question(anchor["question_id"]),
                    "anchor_completed_at": anchor["completed_at"].isoformat(),
                    "target_event_id": self.pseudonymizer.event(target["event_id"]),
                    "target_session_id": self.pseudonymizer.session(target["session_id"]),
                    "target_question_id": self.pseudonymizer.question(target["question_id"]),
                    "target_completed_at": target["completed_at"].isoformat(),
                    "label_delay_hours": round(delay, 4),
                    "target_quality_score": _optional_float(target.get("quality_score")),
                    TARGET_COLUMN: success,
                })
        return DatasetBuildResult(labeled, audit, unlabeled, len(events), len(enriched))

    def _add_learner_history(self, events: list[dict]) -> list[dict]:
        history: dict[str, dict] = {}
        enriched = []
        for event in sorted(events, key=lambda item: (item["completed_at"], item["event_id"])):
            stats = history.setdefault(event["student_id"], {
                "attempts": 0, "successes": 0, "quality_sum": 0.0, "quality_count": 0,
            })
            item = dict(event)
            item["learner_prior_attempts"] = stats["attempts"]
            item["learner_prior_accuracy"] = (
                stats["successes"] / stats["attempts"] if stats["attempts"] else 0.0
            )
            item["learner_prior_average_quality"] = (
                stats["quality_sum"] / stats["quality_count"] if stats["quality_count"] else 0.0
            )
            enriched.append(item)
            stats["attempts"] += 1
            stats["successes"] += int(event.get("final_correct") is True)
            quality = _optional_float(event.get("quality_score"))
            if quality is not None:
                stats["quality_sum"] += quality
                stats["quality_count"] += 1
        return enriched

    def _find_target(self, anchor: dict, later: list[dict]) -> dict | None:
        for candidate in later:
            delay = candidate["completed_at"] - anchor["completed_at"]
            if delay < self.min_delay:
                continue
            if delay > self.max_delay:
                return None
            return candidate
        return None

    def _metadata(self, anchor: dict) -> dict:
        learner = self.pseudonymizer.learner(anchor["student_id"])
        return {
            "row_id": self.pseudonymizer.row(anchor["event_id"]),
            "learner_group_id": learner,
            "sign_group_id": self.pseudonymizer.sign(anchor["sign_id"]),
            "split": self.pseudonymizer.split_for_learner(learner),
        }

    def _features(self, anchor: dict) -> dict:
        before, after = anchor.get("before_state") or {}, anchor.get("after_state") or {}
        anchor_time = anchor["completed_at"]
        previous = _utc_datetime(before.get("last_reviewed_at"))
        since_previous = None if previous is None else round(
            max((anchor_time - previous).total_seconds() / 3600, 0.0), 4
        )
        next_review = _utc_datetime(after.get("next_review_at"))
        scheduled = (round(max((next_review - anchor_time).total_seconds() / 3600, 0.0), 4)
                     if next_review else _int(after, "interval_days") * 24.0)
        status = str(anchor.get("question_status", "incorrect"))
        return {
            "category_code": str(anchor.get("category_code", "UNKNOWN")),
            "prompt_language": str(anchor.get("prompt_language", "english")),
            "selection_strategy": str(anchor.get("selection_strategy", "legacy_random")),
            "selection_reason": str(anchor.get("selection_reason", "legacy_random")),
            "anchor_outcome": status,
            "anchor_correct": int(anchor.get("final_correct") is True),
            "anchor_was_skipped": int(status == "skipped"),
            "anchor_quality_score": _optional_float(anchor.get("quality_score")),
            "anchor_quality_band": int(anchor.get("quality_band", 0) or 0),
            "response_time_ms": _optional_float(anchor.get("response_time_ms")),
            "retry_count": int(anchor.get("retry_count", 0)),
            "hint_count": int(anchor.get("hint_count", 0)),
            "sign_difficulty": int(anchor.get("difficulty", 1)),
            "was_due": int(bool(anchor.get("was_due", False))),
            "days_overdue": float(anchor.get("days_overdue", 0.0) or 0.0),
            "selection_priority": float(anchor.get("selection_priority", 0.0) or 0.0),
            "pre_mastery_score": _float(before, "score"),
            "post_mastery_score": _float(after, "score"),
            "pre_total_reviews": _int(before, "total_reviews"),
            "pre_successful_reviews": _int(before, "successful_reviews"),
            "pre_failure_count": _int(before, "failure_count"),
            "pre_average_quality_score": _float(before, "average_quality_score"),
            "pre_average_response_time_ms": _optional_float(before.get("average_response_time_ms")),
            "hours_since_previous_review": since_previous,
            "pre_ease_factor": _float(before, "ease_factor", 2.5),
            "post_ease_factor": _float(after, "ease_factor", 2.5),
            "pre_interval_days": _int(before, "interval_days"),
            "post_interval_days": _int(after, "interval_days"),
            "pre_repetition_count": _int(before, "repetition_count"),
            "post_repetition_count": _int(after, "repetition_count"),
            "pre_lapse_count": _int(before, "lapse_count"),
            "post_lapse_count": _int(after, "lapse_count"),
            "scheduled_delay_hours": scheduled,
            "learner_prior_attempts": int(anchor.get("learner_prior_attempts", 0)),
            "learner_prior_accuracy": round(float(anchor.get("learner_prior_accuracy", 0.0)), 6),
            "learner_prior_average_quality": round(float(anchor.get("learner_prior_average_quality", 0.0)), 6),
        }

    def _unlabeled_reason(self, anchor: dict, now: datetime) -> str:
        if now < anchor["completed_at"] + self.min_delay:
            return "not_matured"
        if now <= anchor["completed_at"] + self.max_delay:
            return "awaiting_followup"
        return "expired_without_followup"
