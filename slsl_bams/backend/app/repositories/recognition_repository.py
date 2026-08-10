from typing import Any

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.db.collections import (
    RECOGNITION_ATTEMPTS_COLLECTION,
)


class RecognitionRepository:
    def __init__(
        self,
        database: Any,
    ) -> None:
        self.collection = database[
            RECOGNITION_ATTEMPTS_COLLECTION
        ]

    async def get_by_request(
        self,
        *,
        student_id: str,
        request_id: str,
    ) -> dict | None:
        if not ObjectId.is_valid(
            student_id
        ):
            return None

        return await self.collection.find_one(
            {
                "student_id": ObjectId(
                    student_id
                ),
                "request_id": request_id,
            }
        )

    async def create_attempt(
        self,
        document: dict,
    ) -> dict:
        try:
            result = (
                await self.collection.insert_one(
                    document
                )
            )
        except DuplicateKeyError:
            existing = await (
                self.collection.find_one(
                    {
                        "student_id": document[
                            "student_id"
                        ],
                        "request_id": document[
                            "request_id"
                        ],
                    }
                )
            )

            if existing is None:
                raise

            return existing

        created = await self.collection.find_one(
            {
                "_id": result.inserted_id,
            }
        )

        if created is None:
            raise RuntimeError(
                "Recognition attempt was stored "
                "but could not be retrieved."
            )

        return created