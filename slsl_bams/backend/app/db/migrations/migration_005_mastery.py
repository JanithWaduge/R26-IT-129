from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

from app.db.collections import (
    MASTERY_EVENTS_COLLECTION,
    MASTERY_RECORDS_COLLECTION,
    QUIZ_SESSIONS_COLLECTION,
)
from app.db.schema import (
    MASTERY_EVENT_VALIDATOR,
    MASTERY_RECORD_VALIDATOR,
    QUIZ_SESSION_VALIDATOR,
)

VERSION = 5
NAME = "Create dual mastery records and mastery events"


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


async def _backfill_quiz_questions(
    database: Any,
) -> None:
    sessions = database[
        QUIZ_SESSIONS_COLLECTION
    ]

    await sessions.update_many(
        {},
        [
            {
                "$set": {
                    "questions": {
                        "$map": {
                            "input": "$questions",
                            "as": "question",
                            "in": {
                                "$mergeObjects": [
                                    "$$question",
                                    {
                                        "mastery_event_id": (
                                            "$$question.question_id"
                                        ),
                                        "mastery_applied_at": None,
                                        "quality_score": None,
                                    },
                                ]
                            },
                        }
                    }
                }
            }
        ],
    )


async def upgrade(database: Any) -> None:
    await _backfill_quiz_questions(
        database
    )

    await database.command(
        {
            "collMod": QUIZ_SESSIONS_COLLECTION,
            "validator": QUIZ_SESSION_VALIDATOR,
            "validationLevel": "strict",
            "validationAction": "error",
        }
    )

    await _create_or_update_collection(
        database,
        MASTERY_RECORDS_COLLECTION,
        MASTERY_RECORD_VALIDATOR,
    )

    await _create_or_update_collection(
        database,
        MASTERY_EVENTS_COLLECTION,
        MASTERY_EVENT_VALIDATOR,
    )

    mastery_records = database[
        MASTERY_RECORDS_COLLECTION
    ]

    await mastery_records.create_indexes(
        [
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("sign_id", ASCENDING),
                ],
                name="uq_mastery_student_sign",
                unique=True,
            ),
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("overall_status", ASCENDING),
                    ("updated_at", DESCENDING),
                ],
                name="ix_mastery_student_status_updated",
            ),
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("balance_status", ASCENDING),
                ],
                name="ix_mastery_student_balance",
            ),
        ]
    )

    mastery_events = database[
        MASTERY_EVENTS_COLLECTION
    ]

    await mastery_events.create_indexes(
        [
            IndexModel(
                [("event_id", ASCENDING)],
                name="uq_mastery_event_id",
                unique=True,
            ),
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("created_at", DESCENDING),
                ],
                name="ix_mastery_event_student_created",
            ),
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("sign_id", ASCENDING),
                    ("created_at", DESCENDING),
                ],
                name="ix_mastery_event_student_sign",
            ),
            IndexModel(
                [
                    ("status", ASCENDING),
                    ("created_at", ASCENDING),
                ],
                name="ix_mastery_event_status_created",
            ),
        ]
    )