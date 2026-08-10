from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION, MASTERY_EVENTS_COLLECTION,
    MASTERY_RECORDS_COLLECTION, ML_RECALL_PREDICTIONS_COLLECTION,
    QUIZ_SESSIONS_COLLECTION, SIGNS_COLLECTION,
)


class HybridSchedulerRepository:
    def __init__(self, database: Any) -> None:
        self.mastery_events = database[MASTERY_EVENTS_COLLECTION]
        self.mastery_records = database[MASTERY_RECORDS_COLLECTION]
        self.predictions = database[ML_RECALL_PREDICTIONS_COLLECTION]
        self.sessions = database[QUIZ_SESSIONS_COLLECTION]
        self.signs = database[SIGNS_COLLECTION]
        self.categories = database[CURRICULUM_CATEGORIES_COLLECTION]

    async def get_event(self, event_id: str) -> dict | None:
        return await self.mastery_events.find_one({"event_id": event_id, "status": "applied"})

    async def get_session(self, session_id: ObjectId) -> dict | None:
        return await self.sessions.find_one({"_id": session_id})

    async def get_mastery_record(self, *, student_id: ObjectId, sign_id: ObjectId) -> dict | None:
        return await self.mastery_records.find_one({"student_id": student_id, "sign_id": sign_id})

    async def get_category_code(self, sign_id: ObjectId) -> str:
        sign = await self.signs.find_one({"_id": sign_id}, {"category_id": 1})
        if sign is None:
            return "UNKNOWN"
        category = await self.categories.find_one({"_id": sign["category_id"]}, {"code": 1})
        return "UNKNOWN" if category is None else str(category.get("code", "UNKNOWN"))

    async def get_learner_history(
        self, *, student_id: ObjectId, before_at: datetime, excluded_event_id: str,
    ) -> dict:
        cursor = await self.mastery_events.aggregate([
            {"$match": {"student_id": student_id, "direction": "receptive",
                        "status": "applied", "event_id": {"$ne": excluded_event_id},
                        "created_at": {"$lt": before_at}}},
            {"$group": {"_id": None, "attempts": {"$sum": 1},
                        "successes": {"$sum": {"$cond": [
                            {"$eq": ["$final_correct", True]}, 1, 0]}},
                        "average_quality": {"$avg": "$quality_score"}}},
        ])
        results = await cursor.to_list(length=1)
        if not results:
            return {"attempts": 0, "accuracy": 0.0, "average_quality": 0.0}
        result = results[0]
        attempts = int(result.get("attempts", 0))
        return {"attempts": attempts,
                "accuracy": int(result.get("successes", 0)) / attempts if attempts else 0.0,
                "average_quality": float(result.get("average_quality", 0.0) or 0.0)}

    async def create_or_get_prediction(self, document: dict) -> dict:
        try:
            result = await self.predictions.insert_one(document)
            created = await self.predictions.find_one({"_id": result.inserted_id})
            if created is None:
                raise RuntimeError("Prediction could not be retrieved.")
            return created
        except DuplicateKeyError:
            existing = await self.predictions.find_one({"event_id": document["event_id"]})
            if existing is None:
                raise
            return existing

    async def update_prediction(self, *, event_id: str, update: dict) -> dict:
        await self.predictions.update_one({"event_id": event_id}, {"$set": update})
        document = await self.predictions.find_one({"event_id": event_id})
        if document is None:
            raise RuntimeError("Prediction disappeared.")
        return document

    async def apply_mastery_adjustment(
        self, *, mastery_id: ObjectId, expected_version: int, event_id: str,
        question_id: str, direction_update: dict,
    ) -> bool:
        result = await self.mastery_records.update_one(
            {"_id": mastery_id, "version": expected_version,
             "receptive.last_question_id": question_id,
             "receptive.last_ml_event_id": {"$ne": event_id}},
            {"$set": {**{f"receptive.{key}": value for key, value in direction_update.items()},
                       "updated_at": direction_update["last_ml_applied_at"]},
             "$inc": {"version": 1}},
        )
        return result.modified_count == 1

    async def list_student_predictions(self, *, student_id: ObjectId, limit: int) -> list[dict]:
        return await self.predictions.find({"student_id": student_id}).sort(
            "created_at", -1
        ).limit(limit).to_list(length=limit)
