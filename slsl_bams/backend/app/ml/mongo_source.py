from datetime import datetime
from typing import Any

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    MASTERY_EVENTS_COLLECTION,
    QUIZ_SESSIONS_COLLECTION,
    SIGNS_COLLECTION,
)


async def load_receptive_recall_events(database: Any) -> list[dict]:
    """Load finalized, applied receptive events into a normalized stream."""
    event_documents = await (
        database[MASTERY_EVENTS_COLLECTION]
        .find({"status": "applied", "direction": "receptive"})
        .sort([("created_at", 1), ("event_id", 1)])
        .to_list(length=None)
    )
    if not event_documents:
        return []

    session_ids = list({e["session_id"] for e in event_documents
                        if e.get("session_id") is not None})
    sign_ids = list({e["sign_id"] for e in event_documents
                     if e.get("sign_id") is not None})
    sessions = await database[QUIZ_SESSIONS_COLLECTION].find(
        {"_id": {"$in": session_ids}}
    ).to_list(length=None)
    signs = await database[SIGNS_COLLECTION].find(
        {"_id": {"$in": sign_ids}}
    ).to_list(length=None)
    category_ids = list({s["category_id"] for s in signs
                         if s.get("category_id") is not None})
    categories = await database[CURRICULUM_CATEGORIES_COLLECTION].find(
        {"_id": {"$in": category_ids}}, {"code": 1}
    ).to_list(length=None)

    category_codes = {str(c["_id"]): c.get("code", "UNKNOWN")
                      for c in categories}
    sign_map = {str(sign["_id"]): sign for sign in signs}
    session_map = {str(session["_id"]): session for session in sessions}
    question_map: dict[tuple[str, str], dict] = {}
    for session in sessions:
        for question in session.get("questions", []):
            if question.get("question_id"):
                question_map[(str(session["_id"]), question["question_id"])] = question

    normalized: list[dict] = []
    for event in event_documents:
        student_id = event.get("student_id")
        sign_id = event.get("sign_id")
        session_id = event.get("session_id")
        question_id = event.get("question_id")
        if student_id is None or sign_id is None or session_id is None or not question_id:
            continue
        session = session_map.get(str(session_id))
        question = question_map.get((str(session_id), question_id))
        if session is None or question is None:
            continue
        completed_at = (question.get("completed_at") or event.get("applied_at")
                        or event.get("created_at"))
        if not isinstance(completed_at, datetime):
            continue
        sign = sign_map.get(str(sign_id), {})
        prompt = question.get("prompt_snapshot", {})
        category_id = prompt.get("category_id") or sign.get("category_id")
        normalized.append({
            "event_id": str(event["event_id"]),
            "student_id": str(student_id),
            "sign_id": str(sign_id),
            "session_id": str(session_id),
            "question_id": question_id,
            "completed_at": completed_at,
            "question_status": event.get("question_status", question.get("status", "incorrect")),
            "final_correct": event.get("final_correct"),
            "quality_score": event.get("quality_score", question.get("quality_score")),
            "quality_band": event.get("quality_band"),
            "response_time_ms": event.get("response_time_ms"),
            "retry_count": int(event.get("retry_count", 0)),
            "hint_count": int(event.get("hint_count", 0)),
            "difficulty": int(event.get("difficulty", prompt.get("difficulty", sign.get("difficulty", 1)))),
            "category_code": category_codes.get(str(category_id), "UNKNOWN"),
            "prompt_language": str(session.get("prompt_language", "english")),
            "selection_strategy": str(session.get("selection_strategy", "legacy_random")),
            "selection_priority": float(question.get("selection_priority", 0.0)),
            "selection_reason": str(question.get("selection_reason", "legacy_random")),
            "was_due": bool(question.get("was_due", False)),
            "days_overdue": float(question.get("days_overdue", 0.0)),
            "before_state": event.get("before_state") or {},
            "after_state": event.get("after_state") or {},
        })
    normalized.sort(key=lambda item: (item["completed_at"], item["event_id"]))
    return normalized
