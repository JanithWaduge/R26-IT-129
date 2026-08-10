from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

from app.db.collections import EXPERIMENT_ASSIGNMENTS_COLLECTION, QUIZ_SESSIONS_COLLECTION
from app.db.schema import EXPERIMENT_ASSIGNMENT_VALIDATOR, QUIZ_SESSION_VALIDATOR

VERSION = 9
NAME = "Create scheduler evaluation experiment"


async def upgrade(database: Any) -> None:
    sessions = database[QUIZ_SESSIONS_COLLECTION]
    await sessions.update_many({}, [{"$set": {
        "experiment_name": {"$ifNull": ["$experiment_name", None]},
        "experiment_arm": {"$ifNull": ["$experiment_arm", "legacy_unassigned"]},
    }}], bypass_document_validation=True)
    await database.command({
        "collMod": QUIZ_SESSIONS_COLLECTION, "validator": QUIZ_SESSION_VALIDATOR,
        "validationLevel": "strict", "validationAction": "error",
    })
    names = await database.list_collection_names()
    if EXPERIMENT_ASSIGNMENTS_COLLECTION not in names:
        await database.create_collection(
            EXPERIMENT_ASSIGNMENTS_COLLECTION,
            validator=EXPERIMENT_ASSIGNMENT_VALIDATOR,
            validationLevel="strict", validationAction="error",
        )
    else:
        await database.command({
            "collMod": EXPERIMENT_ASSIGNMENTS_COLLECTION,
            "validator": EXPERIMENT_ASSIGNMENT_VALIDATOR,
            "validationLevel": "strict", "validationAction": "error",
        })
    await database[EXPERIMENT_ASSIGNMENTS_COLLECTION].create_indexes([
        IndexModel([("experiment_name", ASCENDING), ("student_id", ASCENDING)],
                   unique=True, name="uq_experiment_student"),
        IndexModel([("experiment_name", ASCENDING), ("arm", ASCENDING),
                    ("assigned_at", DESCENDING)], name="ix_experiment_arm_assigned"),
    ])
