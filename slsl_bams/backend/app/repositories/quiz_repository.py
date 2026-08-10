from typing import Any

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    QUIZ_SESSIONS_COLLECTION,
    SIGNS_COLLECTION,
)


class QuizSessionNotFoundError(Exception):
    pass


class QuizConcurrentUpdateError(Exception):
    pass


class QuizRepository:
    def __init__(
        self,
        database: Any,
    ) -> None:
        self.sessions = database[
            QUIZ_SESSIONS_COLLECTION
        ]

        self.signs = database[
            SIGNS_COLLECTION
        ]

        self.categories = database[
            CURRICULUM_CATEGORIES_COLLECTION
        ]

    @staticmethod
    def parse_object_id(
        value: str,
        *,
        message: str,
    ) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise QuizSessionNotFoundError(
                message
            )

        return ObjectId(value)

    async def find_existing_creation(
        self,
        *,
        student_id: str,
        client_request_id: str,
    ) -> dict | None:
        return await self.sessions.find_one(
            {
                "student_id": self.parse_object_id(
                    student_id,
                    message="Student identifier is invalid.",
                ),
                "client_request_id": client_request_id,
            }
        )

    async def create_session(
        self,
        document: dict,
    ) -> dict:
        try:
            result = await self.sessions.insert_one(
                document
            )
        except DuplicateKeyError:
            existing = await self.sessions.find_one(
                {
                    "student_id": document[
                        "student_id"
                    ],
                    "client_request_id": document[
                        "client_request_id"
                    ],
                }
            )

            if existing is None:
                raise

            return existing

        created = await self.sessions.find_one(
            {
                "_id": result.inserted_id,
            }
        )

        if created is None:
            raise RuntimeError(
                "Quiz session was created but "
                "could not be retrieved."
            )

        return created

    async def get_owned_session(
        self,
        *,
        session_id: str,
        student_id: str,
    ) -> dict:
        session_object_id = self.parse_object_id(
            session_id,
            message="Quiz session was not found.",
        )

        student_object_id = self.parse_object_id(
            student_id,
            message="Student identifier is invalid.",
        )

        document = await self.sessions.find_one(
            {
                "_id": session_object_id,
                "student_id": student_object_id,
            }
        )

        if document is None:
            raise QuizSessionNotFoundError(
                "Quiz session was not found."
            )

        return document

    async def replace_with_version(
        self,
        *,
        document: dict,
        expected_version: int,
    ) -> dict:
        replacement = dict(document)

        replacement["version"] = (
            expected_version + 1
        )

        result = await self.sessions.replace_one(
            {
                "_id": replacement["_id"],
                "student_id": replacement[
                    "student_id"
                ],
                "version": expected_version,
            },
            replacement,
        )

        if result.matched_count == 0:
            raise QuizConcurrentUpdateError(
                "The quiz session changed in another "
                "request. Reload the session and retry."
            )

        updated = await self.sessions.find_one(
            {
                "_id": replacement["_id"],
            }
        )

        if updated is None:
            raise RuntimeError(
                "Updated quiz session could not "
                "be retrieved."
            )

        return updated

    async def find_sign_candidates(
        self,
        *,
        visible_statuses: list[str],
        category_id: str | None,
        competency_id: str | None,
        difficulty: int | None,
    ) -> list[dict]:
        query: dict[str, Any] = {
            "is_active": True,
            "content_status": {
                "$in": visible_statuses,
            },
        }

        if category_id is not None:
            query["category_id"] = (
                self.parse_object_id(
                    category_id,
                    message="Category filter is invalid.",
                )
            )

        if competency_id is not None:
            query["competency_ids"] = (
                self.parse_object_id(
                    competency_id,
                    message="Competency filter is invalid.",
                )
            )

        if difficulty is not None:
            query["difficulty"] = difficulty

        cursor = self.signs.find(query)

        return await cursor.to_list(
            length=None
        )

    async def find_all_visible_signs(
        self,
        *,
        visible_statuses: list[str],
    ) -> list[dict]:
        cursor = self.signs.find(
            {
                "is_active": True,
                "content_status": {
                    "$in": visible_statuses,
                },
            }
        )

        return await cursor.to_list(
            length=None
        )

    async def get_category_documents(
        self,
        category_ids: list[ObjectId],
    ) -> dict[str, dict]:
        cursor = self.categories.find(
            {
                "_id": {
                    "$in": category_ids,
                },
                "is_active": True,
            }
        )

        documents = await cursor.to_list(
            length=None
        )

        return {
            str(document["_id"]): document
            for document in documents
        }