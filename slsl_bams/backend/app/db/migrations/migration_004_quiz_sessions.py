from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

from app.db.collections import (
    QUIZ_SESSIONS_COLLECTION,
)
from app.db.schema import QUIZ_SESSION_VALIDATOR

VERSION = 4
NAME = "Create embedded quiz sessions and interaction logs"


async def _create_or_update_collection(
    database: Any,
    collection_name: str,
    validator: dict,
) -> None:
    collection_names = await database.list_collection_names()

    if collection_name not in collection_names:
        await database.create_collection(
            collection_name,
            validator=validator,
            validationLevel="strict",
            validationAction="error",
        )
        return

    await database.command(
        {
            "collMod": collection_name,
            "validator": validator,
            "validationLevel": "strict",
            "validationAction": "error",
        }
    )


async def upgrade(database: Any) -> None:
    await _create_or_update_collection(
        database,
        QUIZ_SESSIONS_COLLECTION,
        QUIZ_SESSION_VALIDATOR,
    )

    sessions = database[
        QUIZ_SESSIONS_COLLECTION
    ]

    await sessions.create_indexes(
        [
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("client_request_id", ASCENDING),
                ],
                name="uq_quiz_student_client_request",
                unique=True,
            ),
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("status", ASCENDING),
                    ("updated_at", DESCENDING),
                ],
                name="ix_quiz_student_status_updated",
            ),
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("started_at", DESCENDING),
                ],
                name="ix_quiz_student_started",
            ),
            IndexModel(
                [
                    ("status", ASCENDING),
                    ("expires_at", ASCENDING),
                ],
                name="ix_quiz_status_expires",
            ),
            IndexModel(
                [
                    ("filters.category_id", ASCENDING),
                    ("status", ASCENDING),
                ],
                name="ix_quiz_category_status",
            ),
        ]
    )