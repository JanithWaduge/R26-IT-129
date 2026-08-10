from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, IndexModel

from app.db.collections import SCHEMA_MIGRATIONS_COLLECTION
from app.db.migrations import (
    migration_001_initial,
    migration_002_authentication,
    migration_003_learning_content,
    migration_004_quiz_sessions,
    migration_005_mastery,
    migration_006_scheduler,
    migration_007_gamification,
    migration_008_hybrid_scheduler,
    migration_009_scheduler_experiment,
    migration_010_governance,
)
from app.db.schema import MIGRATION_VALIDATOR

MIGRATIONS = (
    migration_001_initial,
    migration_002_authentication,
    migration_003_learning_content,
    migration_004_quiz_sessions,
    migration_005_mastery,
    migration_006_scheduler,
    migration_007_gamification,
    migration_008_hybrid_scheduler,
    migration_009_scheduler_experiment,
    migration_010_governance,
)


async def _ensure_migration_collection(
    database: Any,
) -> None:
    collection_names = await database.list_collection_names()

    if SCHEMA_MIGRATIONS_COLLECTION not in collection_names:
        await database.create_collection(
            SCHEMA_MIGRATIONS_COLLECTION,
            validator=MIGRATION_VALIDATOR,
            validationLevel="strict",
            validationAction="error",
        )
    else:
        await database.command(
            {
                "collMod": SCHEMA_MIGRATIONS_COLLECTION,
                "validator": MIGRATION_VALIDATOR,
                "validationLevel": "strict",
                "validationAction": "error",
            }
        )

    await database[
        SCHEMA_MIGRATIONS_COLLECTION
    ].create_indexes(
        [
            IndexModel(
                [("version", ASCENDING)],
                name="uq_schema_migration_version",
                unique=True,
            )
        ]
    )


async def run_migrations(database: Any) -> None:
    await _ensure_migration_collection(database)

    collection = database[
        SCHEMA_MIGRATIONS_COLLECTION
    ]

    for migration in MIGRATIONS:
        existing = await collection.find_one(
            {"version": migration.VERSION}
        )

        if existing is not None:
            continue

        await migration.upgrade(database)

        await collection.insert_one(
            {
                "version": migration.VERSION,
                "name": migration.NAME,
                "applied_at": datetime.now(timezone.utc),
            }
        )
