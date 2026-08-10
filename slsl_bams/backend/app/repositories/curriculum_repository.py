from typing import Any

from bson import ObjectId
from pymongo import ASCENDING

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    CURRICULUM_COMPETENCIES_COLLECTION,
)
from app.schemas.curriculum import (
    CurriculumCategoryResponse,
    CurriculumCompetencyResponse,
)


class CurriculumCategoryNotFoundError(Exception):
    pass


class CurriculumRepository:
    def __init__(self, database: Any) -> None:
        self.categories = database[
            CURRICULUM_CATEGORIES_COLLECTION
        ]

        self.competencies = database[
            CURRICULUM_COMPETENCIES_COLLECTION
        ]

    @staticmethod
    def _parse_object_id(
        value: str,
    ) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise CurriculumCategoryNotFoundError(
                "Curriculum category was not found."
            )

        return ObjectId(value)

    @staticmethod
    def category_to_response(
        document: dict,
    ) -> CurriculumCategoryResponse:
        return CurriculumCategoryResponse(
            id=str(document["_id"]),
            code=document["code"],
            name=document["name"],
            description=document["description"],
            display_order=document[
                "display_order"
            ],
            icon_key=document["icon_key"],
            adaptive_priority_weight=float(
                document[
                    "adaptive_priority_weight"
                ]
            ),
            validation_status=document[
                "validation_status"
            ],
            is_active=document["is_active"],
            created_at=document["created_at"],
            updated_at=document["updated_at"],
        )

    @staticmethod
    def competency_to_response(
        document: dict,
    ) -> CurriculumCompetencyResponse:
        return CurriculumCompetencyResponse(
            id=str(document["_id"]),
            category_id=str(
                document["category_id"]
            ),
            code=document["code"],
            title=document["title"],
            description=document["description"],
            grade_levels=document[
                "grade_levels"
            ],
            display_order=document[
                "display_order"
            ],
            receptive_mastery_threshold=float(
                document[
                    "receptive_mastery_threshold"
                ]
            ),
            productive_mastery_threshold=float(
                document[
                    "productive_mastery_threshold"
                ]
            ),
            validation_status=document[
                "validation_status"
            ],
            is_active=document["is_active"],
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            adaptive_priority_weight=float(
                document["adaptive_priority_weight"]
            ),
        )

    async def list_categories(
        self,
    ) -> list[CurriculumCategoryResponse]:
        cursor = (
            self.categories
            .find(
                {
                    "is_active": True,
                }
            )
            .sort(
                "display_order",
                ASCENDING,
            )
        )

        documents = await cursor.to_list(
            length=None
        )

        return [
            self.category_to_response(
                document
            )
            for document in documents
        ]

    async def get_category(
        self,
        category_id: str,
    ) -> CurriculumCategoryResponse:
        object_id = self._parse_object_id(
            category_id
        )

        document = await self.categories.find_one(
            {
                "_id": object_id,
                "is_active": True,
            }
        )

        if document is None:
            raise CurriculumCategoryNotFoundError(
                "Curriculum category was not found."
            )

        return self.category_to_response(
            document
        )

    async def list_competencies(
        self,
        *,
        category_id: str | None = None,
    ) -> list[
        CurriculumCompetencyResponse
    ]:
        query: dict[str, Any] = {
            "is_active": True,
        }

        if category_id is not None:
            query["category_id"] = (
                self._parse_object_id(
                    category_id
                )
            )

        cursor = (
            self.competencies
            .find(query)
            .sort(
                [
                    (
                        "category_id",
                        ASCENDING,
                    ),
                    (
                        "display_order",
                        ASCENDING,
                    ),
                ]
            )
        )

        documents = await cursor.to_list(
            length=None
        )

        return [
            self.competency_to_response(
                document
            )
            for document in documents
        ]

    async def get_competencies_by_ids(
        self,
        competency_ids: list[ObjectId],
    ) -> list[
        CurriculumCompetencyResponse
    ]:
        if not competency_ids:
            return []

        cursor = self.competencies.find(
            {
                "_id": {
                    "$in": competency_ids,
                },
                "is_active": True,
            }
        )

        documents = await cursor.to_list(
            length=None
        )

        response_by_id = {
            str(document["_id"]):
                self.competency_to_response(
                    document
                )
            for document in documents
        }

        return [
            response_by_id[str(object_id)]
            for object_id in competency_ids
            if str(object_id) in response_by_id
        ]
