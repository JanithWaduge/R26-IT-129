from typing import Any

from bson import ObjectId

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    CURRICULUM_COMPETENCIES_COLLECTION,
    MASTERY_RECORDS_COLLECTION,
)


class ReviewRepository:
    def __init__(
        self,
        database: Any,
    ) -> None:
        self.mastery_records = database[
            MASTERY_RECORDS_COLLECTION
        ]

        self.categories = database[
            CURRICULUM_CATEGORIES_COLLECTION
        ]

        self.competencies = database[
            CURRICULUM_COMPETENCIES_COLLECTION
        ]

    async def get_mastery_map(
        self,
        *,
        student_id: str,
        sign_ids: list[ObjectId],
    ) -> dict[str, dict]:
        if not ObjectId.is_valid(student_id):
            return {}

        cursor = self.mastery_records.find(
            {
                "student_id": ObjectId(
                    student_id
                ),
                "sign_id": {
                    "$in": sign_ids,
                },
            }
        )

        records = await cursor.to_list(
            length=None
        )

        return {
            str(record["sign_id"]): record
            for record in records
        }

    async def get_category_weight_map(
        self,
        category_ids: list[ObjectId],
    ) -> dict[str, float]:
        cursor = self.categories.find(
            {
                "_id": {
                    "$in": category_ids,
                },
                "is_active": True,
            },
            {
                "adaptive_priority_weight": 1,
            },
        )

        documents = await cursor.to_list(
            length=None
        )

        return {
            str(document["_id"]): float(
                document.get(
                    "adaptive_priority_weight",
                    1.0,
                )
            )
            for document in documents
        }

    async def get_competency_weight_map(
        self,
        competency_ids: list[ObjectId],
    ) -> dict[str, float]:
        cursor = self.competencies.find(
            {
                "_id": {
                    "$in": competency_ids,
                },
                "is_active": True,
            },
            {
                "adaptive_priority_weight": 1,
            },
        )

        documents = await cursor.to_list(
            length=None
        )

        return {
            str(document["_id"]): float(
                document.get(
                    "adaptive_priority_weight",
                    1.0,
                )
            )
            for document in documents
        }