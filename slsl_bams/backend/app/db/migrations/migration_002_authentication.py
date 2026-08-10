from typing import Any

from pymongo import ASCENDING, IndexModel

from app.db.collections import (
    REFRESH_SESSIONS_COLLECTION,
    USERS_COLLECTION,
)
from app.db.schema import (
    REFRESH_SESSION_VALIDATOR,
    USER_VALIDATOR,
)

VERSION = 2
NAME = "Create users and refresh sessions collections"


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
        USERS_COLLECTION,
        USER_VALIDATOR,
    )

    await _create_or_update_collection(
        database,
        REFRESH_SESSIONS_COLLECTION,
        REFRESH_SESSION_VALIDATOR,
    )

    users = database[USERS_COLLECTION]

    await users.create_indexes(
        [
            IndexModel(
                [("email", ASCENDING)],
                name="uq_users_email",
                unique=True,
            ),
            IndexModel(
                [("student_id", ASCENDING)],
                name="uq_users_student_id",
                unique=True,
            ),
            IndexModel(
                [
                    ("role", ASCENDING),
                    ("is_active", ASCENDING),
                ],
                name="ix_users_role_active",
            ),
        ]
    )

    refresh_sessions = database[REFRESH_SESSIONS_COLLECTION]

    await refresh_sessions.create_indexes(
        [
            IndexModel(
                [("jti_hash", ASCENDING)],
                name="uq_refresh_sessions_jti_hash",
                unique=True,
            ),
            IndexModel(
                [("expires_at", ASCENDING)],
                name="ttl_refresh_sessions_expires_at",
                expireAfterSeconds=0,
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("family_id", ASCENDING),
                ],
                name="ix_refresh_sessions_user_family",
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("revoked_at", ASCENDING),
                ],
                name="ix_refresh_sessions_user_revoked",
            ),
        ]
    )