from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

from app.db.collections import MASTERY_RECORDS_COLLECTION, ML_RECALL_PREDICTIONS_COLLECTION
from app.db.schema import MASTERY_RECORD_VALIDATOR, ML_RECALL_PREDICTION_VALIDATOR

VERSION = 8
NAME = "Add bounded hybrid recall scheduler"


async def upgrade(database: Any) -> None:
    mastery_records = database[MASTERY_RECORDS_COLLECTION]
    defaults = {
        "base_sm2_interval_days": 0,
        "last_ml_probability": None,
        "last_ml_adjustment_factor": None,
        "last_ml_model_version": None,
        "last_ml_status": "not_evaluated",
        "last_ml_event_id": None,
        "last_ml_applied_at": None,
    }
    await mastery_records.update_many({}, [{"$set": {
        "receptive": {"$mergeObjects": [defaults, "$receptive"]},
        "productive": {"$mergeObjects": [defaults, "$productive"]},
    }}], bypass_document_validation=True)
    await database.command({
        "collMod": MASTERY_RECORDS_COLLECTION,
        "validator": MASTERY_RECORD_VALIDATOR,
        "validationLevel": "strict", "validationAction": "error",
    })
    names = await database.list_collection_names()
    if ML_RECALL_PREDICTIONS_COLLECTION not in names:
        await database.create_collection(
            ML_RECALL_PREDICTIONS_COLLECTION,
            validator=ML_RECALL_PREDICTION_VALIDATOR,
            validationLevel="strict", validationAction="error",
        )
    else:
        await database.command({
            "collMod": ML_RECALL_PREDICTIONS_COLLECTION,
            "validator": ML_RECALL_PREDICTION_VALIDATOR,
            "validationLevel": "strict", "validationAction": "error",
        })
    await database[ML_RECALL_PREDICTIONS_COLLECTION].create_indexes([
        IndexModel([("event_id", ASCENDING)], unique=True, name="uq_ml_recall_event"),
        IndexModel([("student_id", ASCENDING), ("created_at", DESCENDING)],
                   name="ix_ml_recall_student_created"),
        IndexModel([("model_version", ASCENDING), ("created_at", DESCENDING)],
                   name="ix_ml_recall_model_created"),
    ])
