from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

from app.db.collections import STUDENTS_COLLECTION
from app.db.schema import STUDENT_VALIDATOR

VERSION = 1
NAME = "Create students collection and indexes"


async def _create_or_update_collection(
    database: Any,
    collection_name: str,
    validator: dict,
) -> None:
    """
    Create a collection with validation.

    If the collection already exists, update its validator using collMod.
    """

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
    """Apply migration version 1."""

    await _create_or_update_collection(
        database=database,
        collection_name=STUDENTS_COLLECTION,
        validator=STUDENT_VALIDATOR,
    )

    students = database[STUDENTS_COLLECTION]

    await students.create_indexes(
        [
            IndexModel(
                [("email", ASCENDING)],
                name="uq_students_email",
                unique=True,
            ),
            IndexModel(
                [
                    ("is_active", ASCENDING),
                    ("created_at", DESCENDING),
                ],
                name="ix_students_active_created_at",
            ),
            IndexModel(
                [("preferred_language", ASCENDING)],
                name="ix_students_preferred_language",
            ),
        ]
    )