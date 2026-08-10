import hashlib
import json
import logging
from datetime import datetime, timezone

from app.algorithms.hybrid_interval import calculate_hybrid_interval
from app.core.config import Settings
from app.ml.live_feature_builder import build_live_recall_features
from app.ml.model_runtime import RecallModelRuntime
from app.repositories.hybrid_scheduler_repository import HybridSchedulerRepository

logger = logging.getLogger(__name__)


class HybridSchedulerService:
    def __init__(self, *, repository: HybridSchedulerRepository,
                 runtime: RecallModelRuntime, settings: Settings) -> None:
        self.repository = repository
        self.runtime = runtime
        self.settings = settings

    async def reconcile_session(self, session: dict) -> None:
        if (self.settings.experiment_enabled
                and session.get("experiment_arm", "not_enrolled") != "hybrid_ml"):
            return
        for question in session.get("questions", []):
            event_id = question.get("mastery_event_id")
            if (not event_id or question.get("mastery_applied_at") is None
                    or question.get("direction") != "receptive"):
                continue
            try:
                await self.apply_event(event_id)
            except Exception:
                logger.exception("Hybrid scheduler failed for mastery event %s", event_id)

    async def apply_event(self, event_id: str) -> dict | None:
        event = await self.repository.get_event(event_id)
        if event is None or event.get("direction") != "receptive":
            return None
        session = await self.repository.get_session(event["session_id"])
        if session is None:
            return None
        question = next((item for item in session.get("questions", [])
                         if item.get("question_id") == event["question_id"]), None)
        if question is None:
            return None
        mastery = await self.repository.get_mastery_record(
            student_id=event["student_id"], sign_id=event["sign_id"]
        )
        if mastery is None:
            return None
        state = mastery["receptive"]
        base = max(1, int((event.get("after_state") or {}).get(
            "interval_days", state.get("interval_days", 1)
        ) or 1))
        if state.get("last_question_id") != event["question_id"]:
            return await self._record_stale(event, question, base)
        created_at = event.get("created_at") or datetime.now(timezone.utc)
        history = await self.repository.get_learner_history(
            student_id=event["student_id"], before_at=created_at,
            excluded_event_id=event_id,
        )
        features = build_live_recall_features(
            event=event, session=session, question=question,
            category_code=await self.repository.get_category_code(event["sign_id"]),
            learner_history=history,
        )
        features_hash = hashlib.sha256(json.dumps(
            features, sort_keys=True, default=str, separators=(",", ":")
        ).encode()).hexdigest()
        now = datetime.now(timezone.utc)
        pending = await self.repository.create_or_get_prediction({
            "event_id": event_id, "student_id": event["student_id"],
            "sign_id": event["sign_id"], "session_id": event["session_id"],
            "question_id": event["question_id"], "direction": "receptive",
            "status": "pending", "model_name": None, "model_version": None,
            "synthetic_model": False, "recall_probability": None,
            "risk_band": None, "base_interval_days": base,
            "adjustment_factor": None, "final_interval_days": base,
            "features_hash": features_hash, "feature_snapshot": features,
            "error_message": None, "created_at": now, "applied_at": None,
        })
        if pending["status"] != "pending":
            return pending
        prediction = self.runtime.predict(features)
        if prediction.status != "predicted" or prediction.probability is None:
            return await self._fallback(
                mastery, event, prediction.status, base, prediction.model_name,
                prediction.model_version, prediction.synthetic_model,
                features_hash, prediction.message,
            )
        reviewed_at = event.get("applied_at") or event.get("created_at") or now
        decision = calculate_hybrid_interval(
            recall_probability=prediction.probability, base_interval_days=base,
            reviewed_at=reviewed_at,
            minimum_factor=self.settings.ml_min_interval_factor,
            maximum_factor=self.settings.ml_max_interval_factor,
            maximum_interval_days=self.settings.sm2_max_interval_days,
        )
        update = {
            "base_sm2_interval_days": decision.base_interval_days,
            "interval_days": decision.final_interval_days,
            "next_review_at": decision.final_next_review_at,
            "last_ml_probability": decision.recall_probability,
            "last_ml_adjustment_factor": decision.adjustment_factor,
            "last_ml_model_version": prediction.model_version,
            "last_ml_status": "applied", "last_ml_event_id": event_id,
            "last_ml_applied_at": now,
        }
        applied = await self.repository.apply_mastery_adjustment(
            mastery_id=mastery["_id"], expected_version=mastery["version"],
            event_id=event_id, question_id=event["question_id"], direction_update=update,
        )
        if not applied:
            existing = await self.repository.get_mastery_record(
                student_id=event["student_id"], sign_id=event["sign_id"]
            )
            applied = bool(existing and existing["receptive"].get("last_ml_event_id") == event_id)
        status = "applied" if applied else "stale_event"
        return await self.repository.update_prediction(event_id=event_id, update={
            "status": status, "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "synthetic_model": prediction.synthetic_model,
            "recall_probability": decision.recall_probability,
            "risk_band": decision.risk_band,
            "adjustment_factor": decision.adjustment_factor,
            "final_interval_days": decision.final_interval_days,
            "error_message": None if applied else "Mastery state changed before ML adjustment.",
            "applied_at": now if applied else None,
        })

    async def _fallback(self, mastery: dict, event: dict, status: str, base: int,
                        model_name: str | None, model_version: str | None,
                        synthetic: bool, features_hash: str,
                        error: str | None) -> dict:
        now = datetime.now(timezone.utc)
        await self.repository.apply_mastery_adjustment(
            mastery_id=mastery["_id"], expected_version=mastery["version"],
            event_id=event["event_id"], question_id=event["question_id"],
            direction_update={
                "base_sm2_interval_days": base, "interval_days": base,
                "last_ml_probability": None, "last_ml_adjustment_factor": None,
                "last_ml_model_version": model_version, "last_ml_status": status,
                "last_ml_event_id": event["event_id"], "last_ml_applied_at": now,
            },
        )
        return await self.repository.update_prediction(event_id=event["event_id"], update={
            "status": status, "model_name": model_name, "model_version": model_version,
            "synthetic_model": synthetic, "recall_probability": None,
            "risk_band": None, "adjustment_factor": None,
            "final_interval_days": base, "features_hash": features_hash,
            "error_message": error, "applied_at": now,
        })

    async def _record_stale(self, event: dict, question: dict, base: int) -> dict:
        now = datetime.now(timezone.utc)
        return await self.repository.create_or_get_prediction({
            "event_id": event["event_id"], "student_id": event["student_id"],
            "sign_id": event["sign_id"], "session_id": event["session_id"],
            "question_id": question["question_id"], "direction": "receptive",
            "status": "stale_event", "model_name": None, "model_version": None,
            "synthetic_model": False, "recall_probability": None,
            "risk_band": None, "base_interval_days": base,
            "adjustment_factor": None, "final_interval_days": base,
            "features_hash": hashlib.sha256((event["event_id"] + ":stale").encode()).hexdigest(),
            "feature_snapshot": {},
            "error_message": "A newer review already updated this mastery state.",
            "created_at": now, "applied_at": now,
        })
