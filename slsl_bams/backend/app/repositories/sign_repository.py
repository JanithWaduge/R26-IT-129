import re
from math import ceil
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING

from app.db.collections import SIGNS_COLLECTION
from app.repositories.curriculum_repository import (
    CurriculumRepository,
)
from app.schemas.sign import (
    PaginatedSignsResponse,
    SignDetailResponse,
    SignSummaryResponse,
)


class SignNotFoundError(Exception):
    pass


class SignRepository:
    def __init__(
        self,
        database: Any,
    ) -> None:
        self.collection = database[
            SIGNS_COLLECTION
        ]

        self.curriculum_repository = (
            CurriculumRepository(database)
        )

    @staticmethod
    def _parse_object_id(
        value: str,
        error_message: str,
    ) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise SignNotFoundError(
                error_message
            )

        return ObjectId(value)

    @staticmethod
    def _summary_from_document(
        document: dict,
    ) -> SignSummaryResponse:
        return SignSummaryResponse(
            id=str(document["_id"]),
            code=document["code"],
            gloss=document["gloss"],
            meanings=document["meanings"],
            category_id=str(
                document["category_id"]
            ),
            difficulty=document[
                "difficulty"
            ],
            media=document["media"],
            content_status=document[
                "content_status"
            ],
            validation_status=document[
                "validation_status"
            ],
        )

    async def list_signs(
        self,
        *,
        visible_statuses: list[str],
        page: int,
        page_size: int,
        category_id: str | None,
        competency_id: str | None,
        difficulty: int | None,
        search: str | None,
    ) -> PaginatedSignsResponse:
        query: dict[str, Any] = {
            "is_active": True,
            "content_status": {
                "$in": visible_statuses,
            },
        }

        if category_id is not None:
            query["category_id"] = (
                self._parse_object_id(
                    category_id,
                    "Category filter is invalid.",
                )
            )

        if competency_id is not None:
            query["competency_ids"] = (
                self._parse_object_id(
                    competency_id,
                    "Competency filter is invalid.",
                )
            )

        if difficulty is not None:
            query["difficulty"] = difficulty

        if search:
            cleaned_search = search.strip()

            if cleaned_search:
                pattern = re.escape(
                    cleaned_search
                )

                query["$or"] = [
                    {
                        "gloss": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "code": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "meanings.english": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                    {
                        "meanings.sinhala": {
                            "$regex": pattern,
                        }
                    },
                    {
                        "meanings.tamil": {
                            "$regex": pattern,
                        }
                    },
                    {
                        "tags": {
                            "$regex": pattern,
                            "$options": "i",
                        }
                    },
                ]

        total_items = await (
            self.collection.count_documents(
                query
            )
        )

        skip = (page - 1) * page_size

        cursor = (
            self.collection
            .find(query)
            .sort(
                [
                    (
                        "difficulty",
                        ASCENDING,
                    ),
                    (
                        "gloss",
                        ASCENDING,
                    ),
                ]
            )
            .skip(skip)
            .limit(page_size)
        )

        documents = await cursor.to_list(
            length=page_size
        )

        total_pages = (
            ceil(total_items / page_size)
            if total_items > 0
            else 0
        )

        return PaginatedSignsResponse(
            items=[
                self._summary_from_document(
                    document
                )
                for document in documents
            ],
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
        )

    async def get_sign(
        self,
        *,
        sign_id: str,
        visible_statuses: list[str],
    ) -> SignDetailResponse:
        object_id = self._parse_object_id(
            sign_id,
            "Sign was not found.",
        )

        document = await self.collection.find_one(
            {
                "_id": object_id,
                "is_active": True,
                "content_status": {
                    "$in": visible_statuses,
                },
            }
        )

        if document is None:
            raise SignNotFoundError(
                "Sign was not found."
            )

        category = (
            await self.curriculum_repository
            .get_category(
                str(document["category_id"])
            )
        )

        competencies = (
            await self.curriculum_repository
            .get_competencies_by_ids(
                document["competency_ids"]
            )
        )

        prerequisites = (
            await self._get_sign_summaries_by_ids(
                sign_ids=document[
                    "prerequisite_sign_ids"
                ],
                visible_statuses=(
                    visible_statuses
                ),
            )
        )

        return SignDetailResponse(
            id=str(document["_id"]),
            code=document["code"],
            gloss=document["gloss"],
            meanings=document["meanings"],
            category=category,
            competencies=competencies,
            difficulty=document[
                "difficulty"
            ],
            tags=document["tags"],
            prerequisites=prerequisites,
            media=document["media"],
            content_status=document[
                "content_status"
            ],
            validation_status=document[
                "validation_status"
            ],
            is_active=document["is_active"],
            created_at=document[
                "created_at"
            ],
            updated_at=document[
                "updated_at"
            ],
        )

    async def _get_sign_summaries_by_ids(
        self,
        *,
        sign_ids: list[ObjectId],
        visible_statuses: list[str],
    ) -> list[SignSummaryResponse]:
        if not sign_ids:
            return []

        cursor = self.collection.find(
            {
                "_id": {
                    "$in": sign_ids,
                },
                "is_active": True,
                "content_status": {
                    "$in": visible_statuses,
                },
            }
        )

        documents = await cursor.to_list(
            length=None
        )

        document_by_id = {
            str(document["_id"]): document
            for document in documents
        }

        return [
            self._summary_from_document(
                document_by_id[
                    str(sign_id)
                ]
            )
            for sign_id in sign_ids
            if str(sign_id) in document_by_id
        ]