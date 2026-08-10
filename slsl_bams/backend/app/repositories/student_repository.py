from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.db.collections import STUDENTS_COLLECTION
from app.schemas.student import (
    StudentCreate,
    StudentResponse,
    StudentUpdate,
)


class StudentNotFoundError(Exception):
    """Raised when a requested student does not exist."""


class DuplicateStudentEmailError(Exception):
    """Raised when a student's email already exists."""


class StudentRepository:
    def __init__(self, database: Any) -> None:
        self.collection = database[STUDENTS_COLLECTION]

    @staticmethod
    def _normalise_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def _parse_object_id(student_id: str) -> ObjectId:
        if not ObjectId.is_valid(student_id):
            raise StudentNotFoundError("Student was not found.")

        return ObjectId(student_id)

    @staticmethod
    def _to_response(document: dict) -> StudentResponse:
        return StudentResponse(
            id=str(document["_id"]),
            full_name=document["full_name"],
            email=document["email"],
            preferred_language=document["preferred_language"],
            grade_level=document["grade_level"],
            is_active=document["is_active"],
            created_at=document["created_at"],
            updated_at=document["updated_at"],
        )

    async def create(
        self,
        student_data: StudentCreate,
    ) -> StudentResponse:
        now = datetime.now(timezone.utc)

        document = student_data.model_dump(mode="python")
        document["email"] = self._normalise_email(
            str(document["email"])
        )
        document["is_active"] = True
        document["created_at"] = now
        document["updated_at"] = now

        try:
            result = await self.collection.insert_one(document)
        except DuplicateKeyError as error:
            raise DuplicateStudentEmailError(
                "A student with this email already exists."
            ) from error

        created_document = await self.collection.find_one(
            {"_id": result.inserted_id}
        )

        if created_document is None:
            raise RuntimeError(
                "Student was inserted but could not be retrieved."
            )

        return self._to_response(created_document)

    async def get_by_id(
        self,
        student_id: str,
    ) -> StudentResponse:
        object_id = self._parse_object_id(student_id)

        document = await self.collection.find_one(
            {"_id": object_id}
        )

        if document is None:
            raise StudentNotFoundError("Student was not found.")

        return self._to_response(document)

    async def list_students(
        self,
        *,
        skip: int,
        limit: int,
        active_only: bool,
    ) -> list[StudentResponse]:
        query: dict[str, Any] = {}

        if active_only:
            query["is_active"] = True

        cursor = (
            self.collection
            .find(query)
            .sort("created_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )

        documents = await cursor.to_list(length=limit)

        return [
            self._to_response(document)
            for document in documents
        ]

    async def update(
        self,
        student_id: str,
        student_data: StudentUpdate,
    ) -> StudentResponse:
        object_id = self._parse_object_id(student_id)

        update_data = student_data.model_dump(
            mode="python",
            exclude_none=True,
        )

        if "email" in update_data:
            update_data["email"] = self._normalise_email(
                str(update_data["email"])
            )

        update_data["updated_at"] = datetime.now(timezone.utc)

        try:
            updated_document = await self.collection.find_one_and_update(
                {"_id": object_id},
                {"$set": update_data},
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError as error:
            raise DuplicateStudentEmailError(
                "A student with this email already exists."
            ) from error

        if updated_document is None:
            raise StudentNotFoundError("Student was not found.")

        return self._to_response(updated_document)

    async def deactivate(
        self,
        student_id: str,
    ) -> None:
        object_id = self._parse_object_id(student_id)

        result = await self.collection.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "is_active": False,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def hard_delete_for_rollback(
        self,
        student_id: str,
    ) -> None:
        """
        Permanently remove a student only when account
        registration fails before completion.
        """

        object_id = self._parse_object_id(
            student_id
        )

        await self.collection.delete_one(
            {"_id": object_id}
        )
        
        if result.matched_count == 0:
            raise StudentNotFoundError("Student was not found.")