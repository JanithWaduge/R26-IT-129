from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

from app.db.collections import RESEARCH_CONSENTS_COLLECTION, SECURITY_AUDIT_EVENTS_COLLECTION

VERSION = 10
NAME = "Add research consent and audit governance"

CONSENT_VALIDATOR = {"$jsonSchema": {
    "bsonType": "object",
    "required": ["experiment_name", "student_id", "consent_version", "status",
                 "participant_assent", "guardian_consent", "source_document_hash",
                 "recorded_at", "withdrawn_at", "created_at", "updated_at"],
    "additionalProperties": False,
    "properties": {
        "_id": {"bsonType": "objectId"}, "experiment_name": {"bsonType": "string"},
        "student_id": {"bsonType": "objectId"}, "consent_version": {"bsonType": "string"},
        "status": {"enum": ["active", "declined", "withdrawn", "expired"]},
        "participant_assent": {"bsonType": ["bool", "null"]},
        "guardian_consent": {"bsonType": ["bool", "null"]},
        "source_document_hash": {"bsonType": ["string", "null"]},
        "recorded_at": {"bsonType": "date"},
        "withdrawn_at": {"bsonType": ["date", "null"]},
        "created_at": {"bsonType": "date"}, "updated_at": {"bsonType": "date"},
    },
}}

AUDIT_VALIDATOR = {"$jsonSchema": {
    "bsonType": "object",
    "required": ["event_type", "actor_id", "request_id", "resource_type", "resource_id",
                 "outcome", "details", "created_at", "expires_at"],
    "additionalProperties": False,
    "properties": {
        "_id": {"bsonType": "objectId"}, "event_type": {"bsonType": "string"},
        "actor_id": {"bsonType": ["objectId", "null"]},
        "request_id": {"bsonType": ["string", "null"]},
        "resource_type": {"bsonType": ["string", "null"]},
        "resource_id": {"bsonType": ["string", "null"]},
        "outcome": {"enum": ["success", "failure", "denied"]},
        "details": {"bsonType": "object"}, "created_at": {"bsonType": "date"},
        "expires_at": {"bsonType": "date"},
    },
}}


async def upgrade(database: Any) -> None:
    existing = await database.list_collection_names()
    for name, validator in ((RESEARCH_CONSENTS_COLLECTION, CONSENT_VALIDATOR),
                            (SECURITY_AUDIT_EVENTS_COLLECTION, AUDIT_VALIDATOR)):
        if name not in existing:
            await database.create_collection(name, validator=validator,
                                             validationLevel="strict", validationAction="error")
        else:
            await database.command({"collMod": name, "validator": validator,
                                    "validationLevel": "strict", "validationAction": "error"})
    await database[RESEARCH_CONSENTS_COLLECTION].create_indexes([
        IndexModel([("experiment_name", ASCENDING), ("student_id", ASCENDING)], unique=True,
                   name="uq_consent_experiment_student"),
        IndexModel([("status", ASCENDING), ("updated_at", DESCENDING)],
                   name="ix_consent_status_updated"),
    ])
    await database[SECURITY_AUDIT_EVENTS_COLLECTION].create_indexes([
        IndexModel([("created_at", DESCENDING)], name="ix_audit_created"),
        IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0, name="ttl_audit_expiry"),
    ])
