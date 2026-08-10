from typing import Any

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.db.collections import (
    EXPERIMENT_ASSIGNMENTS_COLLECTION, RESEARCH_CONSENTS_COLLECTION, STUDENTS_COLLECTION,
)


class ExperimentRepository:
    def __init__(self, database: Any) -> None:
        self.database = database
        self.assignments = database[EXPERIMENT_ASSIGNMENTS_COLLECTION]
        self.students = database[STUDENTS_COLLECTION]

    async def get_student_context(self, student_id: str) -> dict:
        if not ObjectId.is_valid(student_id):
            raise ValueError("Invalid student identifier.")
        student = await self.students.find_one(
            {"_id": ObjectId(student_id)}, {"preferred_language": 1, "grade_level": 1}
        )
        if student is None:
            raise ValueError("Student profile was not found.")
        return student

    async def get_assignment(self, *, experiment_name: str, student_id: str) -> dict | None:
        return await self.assignments.find_one({
            "experiment_name": experiment_name, "student_id": ObjectId(student_id),
            "is_active": True,
        })

    async def has_active_consent(self, *, experiment_name: str, student_id: str) -> bool:
        if not ObjectId.is_valid(student_id):
            return False
        document = await self.database[RESEARCH_CONSENTS_COLLECTION].find_one(
            {
                "experiment_name": experiment_name,
                "student_id": ObjectId(student_id),
                "status": "active",
                "participant_assent": {"$ne": False},
                "guardian_consent": {"$ne": False},
            },
            {"_id": 1},
        )
        return document is not None

    async def create_or_get_assignment(self, document: dict) -> dict:
        try:
            result = await self.assignments.insert_one(document)
            created = await self.assignments.find_one({"_id": result.inserted_id})
            if created is None:
                raise RuntimeError("Experiment assignment could not be retrieved.")
            return created
        except DuplicateKeyError:
            existing = await self.assignments.find_one({
                "experiment_name": document["experiment_name"],
                "student_id": document["student_id"],
            })
            if existing is None:
                raise
            return existing
