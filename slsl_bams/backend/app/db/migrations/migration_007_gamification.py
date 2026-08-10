from typing import Any

from pymongo import (
    ASCENDING,
    DESCENDING,
    IndexModel,
)

from app.db.collections import (
    GAMIFICATION_EVENTS_COLLECTION,
    GAMIFICATION_PROFILES_COLLECTION,
)
from app.db.schema import (
    GAMIFICATION_EVENT_VALIDATOR,
    GAMIFICATION_PROFILE_VALIDATOR,
)

VERSION = 7
NAME = "Create gamification profiles and reward events"


async def _create_or_update_collection(
    database: Any,
    *,
    collection_name: str,
    validator: dict,
) -> None:
    collection_names = (
        await database.list_collection_names()
    )

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
        collection_name=(
            GAMIFICATION_PROFILES_COLLECTION
        ),
        validator=(
            GAMIFICATION_PROFILE_VALIDATOR
        ),
    )

    await _create_or_update_collection(
        database,
        collection_name=(
            GAMIFICATION_EVENTS_COLLECTION
        ),
        validator=(
            GAMIFICATION_EVENT_VALIDATOR
        ),
    )

    profiles = database[
        GAMIFICATION_PROFILES_COLLECTION
    ]

    await profiles.create_indexes(
        [
            IndexModel(
                [
                    ("student_id", ASCENDING),
                ],
                name="uq_gamification_student",
                unique=True,
            ),
            IndexModel(
                [
                    ("level", DESCENDING),
                    ("total_xp", DESCENDING),
                ],
                name="ix_gamification_level_xp",
            ),
        ]
    )

    events = database[
        GAMIFICATION_EVENTS_COLLECTION
    ]

    await events.create_indexes(
        [
            IndexModel(
                [
                    ("session_id", ASCENDING),
                ],
                name="uq_gamification_session",
                unique=True,
            ),
            IndexModel(
                [
                    ("student_id", ASCENDING),
                    ("created_at", DESCENDING),
                ],
                name=(
                    "ix_gamification_student_created"
                ),
            ),
        ]
    )
