from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import AsyncMongoClient
from sklearn.metrics import (
    average_precision_score, brier_score_loss, mean_absolute_error,
    mean_squared_error, roc_auc_score,
)

from app.core.config import get_settings
from app.db.collections import (
    MASTERY_EVENTS_COLLECTION, ML_RECALL_PREDICTIONS_COLLECTION,
    QUIZ_SESSIONS_COLLECTION,
)

VALID_ARMS = ("random_control", "modified_sm2", "hybrid_ml")


def utc_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def build_delayed_labels(events: list[dict], *, minimum_delay: timedelta,
                         maximum_delay: timedelta) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        grouped[(event["student_id"], event["sign_id"])].append(event)
    labels = []
    for group in grouped.values():
        group.sort(key=lambda item: (item["completed_at"], item["event_id"]))
        for index, anchor in enumerate(group):
            target = None
            for candidate in group[index + 1:]:
                delay = candidate["completed_at"] - anchor["completed_at"]
                if delay < minimum_delay:
                    continue
                if delay > maximum_delay:
                    break
                target = candidate
                break
            if target is None or target["arm"] != anchor["arm"]:
                continue
            delay_hours = (target["completed_at"] - anchor["completed_at"]).total_seconds() / 3600
            labels.append({
                **anchor, "future_recall_success": target["correct"],
                "label_delay_hours": delay_hours,
                "schedule_error_days": abs(delay_hours - anchor["scheduled_delay_hours"]) / 24,
                "target_event_id": target["event_id"],
            })
    return labels


def summarize_arm(arm_frame: pd.DataFrame, all_events: pd.DataFrame) -> dict:
    arm = str(arm_frame["arm"].iloc[0])
    events = all_events.loc[all_events["arm"] == arm]
    retained = int(arm_frame["future_recall_success"].sum())
    response_hours = events["response_time_ms"].sum() / 3_600_000
    return {
        "arm": arm, "learners": int(arm_frame["student_id"].nunique()),
        "delayed_labels": len(arm_frame), "retained_recalls": retained,
        "retention_rate": float(arm_frame["future_recall_success"].mean()),
        "mean_quality_score": float(arm_frame["quality_score"].mean()),
        "mean_response_time_ms": float(arm_frame["response_time_ms"].mean()),
        "total_question_events": len(events),
        "retained_recalls_per_100_events": retained / max(len(events), 1) * 100,
        "retained_recalls_per_response_hour": retained / max(response_hours, 0.000001),
    }


def bootstrap_difference(frame: pd.DataFrame, *, first_arm: str, second_arm: str,
                         iterations: int, seed: int = 42) -> dict:
    generator = np.random.default_rng(seed)
    groups = {arm: {learner: data["future_recall_success"].to_numpy()
                    for learner, data in frame.loc[frame["arm"] == arm].groupby("student_id")}
              for arm in (first_arm, second_arm)}
    empty = {"first_arm": first_arm, "second_arm": second_arm,
             "difference": None, "ci_95_low": None, "ci_95_high": None}
    if any(not values for values in groups.values()):
        return empty
    differences = []
    for _ in range(iterations):
        rates = {}
        for arm, learners in groups.items():
            ids = list(learners)
            sampled = generator.choice(ids, size=len(ids), replace=True)
            rates[arm] = float(np.concatenate([learners[item] for item in sampled]).mean())
        differences.append(rates[first_arm] - rates[second_arm])
    observed = (frame.loc[frame["arm"] == first_arm, "future_recall_success"].mean()
                - frame.loc[frame["arm"] == second_arm, "future_recall_success"].mean())
    return {"first_arm": first_arm, "second_arm": second_arm,
            "difference": float(observed),
            "ci_95_low": float(np.percentile(differences, 2.5)),
            "ci_95_high": float(np.percentile(differences, 97.5))}


def prediction_metrics(frame: pd.DataFrame) -> dict:
    truth = frame["future_recall_success"].astype(int)
    probability = frame["recall_probability"].astype(float)
    result = {
        "rows": len(frame),
        "mae": float(mean_absolute_error(truth, probability)),
        "rmse": float(math.sqrt(mean_squared_error(truth, probability))),
        "brier_score": float(brier_score_loss(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
        "synthetic_predictions": bool(frame.get("synthetic_model", pd.Series(False)).any()),
    }
    result["roc_auc"] = float(roc_auc_score(truth, probability)) if truth.nunique() == 2 else None
    return result


async def load_events(database, *, experiment_name: str) -> list[dict]:
    sessions = await database[QUIZ_SESSIONS_COLLECTION].find({
        "experiment_name": experiment_name, "experiment_arm": {"$in": list(VALID_ARMS)}
    }).to_list(length=None)
    session_map = {str(item["_id"]): item for item in sessions}
    events = await database[MASTERY_EVENTS_COLLECTION].find({
        "session_id": {"$in": [item["_id"] for item in sessions]},
        "direction": "receptive", "status": "applied",
    }).sort("created_at", 1).to_list(length=None)
    normalized = []
    for event in events:
        session = session_map.get(str(event["session_id"]))
        if session is None:
            continue
        question = next((q for q in session.get("questions", [])
                         if q.get("question_id") == event["question_id"]), None)
        if question is None:
            continue
        completed = utc_datetime(question.get("completed_at") or event.get("applied_at"))
        if completed is None:
            continue
        after = event.get("after_state") or {}
        normalized.append({
            "event_id": event["event_id"], "student_id": str(event["student_id"]),
            "sign_id": str(event["sign_id"]), "arm": session["experiment_arm"],
            "completed_at": completed, "correct": int(event.get("final_correct") is True),
            "quality_score": float(event.get("quality_score", 0) or 0),
            "response_time_ms": float(event.get("response_time_ms", 0) or 0),
            "scheduled_delay_hours": float(after.get("interval_days", 1) or 1) * 24,
        })
    return normalized


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    settings = get_settings()
    name = args.experiment_name or settings.experiment_name
    client = AsyncMongoClient(settings.mongo_uri)
    try:
        events = await load_events(client[settings.mongo_database], experiment_name=name)
    finally:
        await client.close()
    labels = build_delayed_labels(
        events, minimum_delay=timedelta(hours=settings.experiment_min_recall_delay_hours),
        maximum_delay=timedelta(days=settings.experiment_max_recall_delay_days),
    )
    output = Path(args.output_dir or settings.experiment_output_directory) / (
        f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output.mkdir(parents=True, exist_ok=True)
    label_columns = [
        "event_id", "student_id", "sign_id", "arm", "completed_at", "correct",
        "quality_score", "response_time_ms", "scheduled_delay_hours",
        "future_recall_success", "label_delay_hours", "schedule_error_days", "target_event_id",
    ]
    label_frame = pd.DataFrame(labels, columns=label_columns)
    label_frame.to_csv(output / "delayed_recall_labels.csv", index=False)

    summaries = [
        summarize_arm(group, pd.DataFrame(events))
        for _, group in label_frame.groupby("arm")
    ] if labels else []
    summary_columns = [
        "arm", "learners", "delayed_labels", "retained_recalls", "retention_rate",
        "mean_quality_score", "mean_response_time_ms", "total_question_events",
        "retained_recalls_per_100_events", "retained_recalls_per_response_hour",
    ]
    pd.DataFrame(summaries, columns=summary_columns).to_csv(
        output / "scheduler_arm_summary.csv", index=False
    )

    comparisons = [
        bootstrap_difference(label_frame, first_arm=first, second_arm=second,
                             iterations=settings.experiment_bootstrap_iterations)
        for first, second in (("modified_sm2", "random_control"),
                              ("hybrid_ml", "random_control"),
                              ("hybrid_ml", "modified_sm2"))
    ]
    pd.DataFrame(comparisons).to_csv(
        output / "pairwise_retention_comparisons.csv", index=False
    )
    report = {
        "experiment_name": name, "events": len(events), "delayed_labels": len(labels),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_definition": {
            "minimum_delay_hours": settings.experiment_min_recall_delay_hours,
            "maximum_delay_days": settings.experiment_max_recall_delay_days,
        },
        "limitations": (["No eligible delayed-recall labels were available."] if not labels else []),
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
