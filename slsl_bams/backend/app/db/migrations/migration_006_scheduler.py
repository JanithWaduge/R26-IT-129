from typing import Any

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    CURRICULUM_COMPETENCIES_COLLECTION,
    MASTERY_EVENTS_COLLECTION,
    MASTERY_RECORDS_COLLECTION,
    QUIZ_SESSIONS_COLLECTION,
)
from app.db.schema import (
    CURRICULUM_CATEGORY_VALIDATOR,
    CURRICULUM_COMPETENCY_VALIDATOR,
    MASTERY_EVENT_VALIDATOR,
    MASTERY_RECORD_VALIDATOR,
    QUIZ_SESSION_VALIDATOR,
)

VERSION = 6
NAME = "Add modified SM-2 schedules and adaptive priority"


async def _apply_validator(
    database: Any,
    *,
    collection_name: str,
    validator: dict,
) -> None:
    """
    Apply the final strict validator only after all
    existing documents have been backfilled.
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


async def _backfill_curriculum_weights(
    database: Any,
) -> None:
    """
    Add default adaptive weights to existing category
    and competency documents.

    Validation is bypassed because some documents may
    temporarily not satisfy the latest schema.
    """

    categories = database[
        CURRICULUM_CATEGORIES_COLLECTION
    ]

    await categories.update_many(
        {
            "$or": [
                {
                    "adaptive_priority_weight": {
                        "$exists": False,
                    }
                },
                {
                    "adaptive_priority_weight": None,
                },
            ]
        },
        {
            "$set": {
                "adaptive_priority_weight": 1.0,
            }
        },
        bypass_document_validation=True,
    )

    competencies = database[
        CURRICULUM_COMPETENCIES_COLLECTION
    ]

    await competencies.update_many(
        {
            "$or": [
                {
                    "adaptive_priority_weight": {
                        "$exists": False,
                    }
                },
                {
                    "adaptive_priority_weight": None,
                },
            ]
        },
        {
            "$set": {
                "adaptive_priority_weight": 1.0,
            }
        },
        bypass_document_validation=True,
    )


async def _backfill_mastery_schedules(
    database: Any,
) -> None:
    """
    Add independent SM-2 schedule fields to both
    receptive and productive mastery states.
    """

    mastery_records = database[
        MASTERY_RECORDS_COLLECTION
    ]

    schedule_defaults = {
        "ease_factor": 2.50,
        "previous_interval_days": 0,
        "interval_days": 0,
        "repetition_count": 0,
        "lapse_count": 0,
        "next_review_at": None,
        "last_scheduled_at": None,
        "last_schedule_reason": "unassessed",
    }

    await mastery_records.update_many(
        {},
        [
            {
                "$set": {
                    "receptive": {
                        "$mergeObjects": [
                            schedule_defaults,
                            "$receptive",
                        ]
                    },
                    "productive": {
                        "$mergeObjects": [
                            schedule_defaults,
                            "$productive",
                        ]
                    },
                }
            }
        ],
        bypass_document_validation=True,
    )


async def _backfill_mastery_events(
    database: Any,
) -> None:
    """
    Existing mastery events were created before the
    scheduler-version field was introduced.
    """

    mastery_events = database[
        MASTERY_EVENTS_COLLECTION
    ]

    await mastery_events.update_many(
        {
            "$or": [
                {
                    "scheduler_algorithm_version": {
                        "$exists": False,
                    }
                },
                {
                    "scheduler_algorithm_version": None,
                },
            ]
        },
        {
            "$set": {
                "scheduler_algorithm_version": (
                    "bams-sm2-v1"
                ),
            }
        },
        bypass_document_validation=True,
    )


async def _backfill_quiz_selection(
    database: Any,
) -> None:
    """
    Add the session strategy and all question-level
    selection fields in one atomic update pipeline.

    Adding them together avoids an intermediate state
    that fails the latest strict validator.
    """

    sessions = database[
        QUIZ_SESSIONS_COLLECTION
    ]

    await sessions.update_many(
        {},
        [
            {
                "$set": {
                    "selection_strategy": {
                        "$ifNull": [
                            "$selection_strategy",
                            "legacy_random",
                        ]
                    },
                    "questions": {
                        "$map": {
                            "input": "$questions",
                            "as": "question",
                            "in": {
                                "$mergeObjects": [
                                    {
                                        "selection_priority": 0.0,
                                        "selection_reason": (
                                            "legacy_random"
                                        ),
                                        "was_due": False,
                                        "days_overdue": 0.0,
                                    },
                                    "$$question",
                                ]
                            },
                        }
                    },
                }
            }
        ],
        bypass_document_validation=True,
    )


async def upgrade(database: Any) -> None:
    # --------------------------------------------------
    # 1. Backfill all existing documents first
    # --------------------------------------------------

    await _backfill_curriculum_weights(
        database
    )

    await _backfill_mastery_schedules(
        database
    )

    await _backfill_mastery_events(
        database
    )

    await _backfill_quiz_selection(
        database
    )

    # --------------------------------------------------
    # 2. Apply the latest strict validators only after
    #    all documents satisfy the new structures
    # --------------------------------------------------

    await _apply_validator(
        database,
        collection_name=(
            CURRICULUM_CATEGORIES_COLLECTION
        ),
        validator=(
            CURRICULUM_CATEGORY_VALIDATOR
        ),
    )

    await _apply_validator(
        database,
        collection_name=(
            CURRICULUM_COMPETENCIES_COLLECTION
        ),
        validator=(
            CURRICULUM_COMPETENCY_VALIDATOR
        ),
    )

    await _apply_validator(
        database,
        collection_name=(
            MASTERY_RECORDS_COLLECTION
        ),
        validator=MASTERY_RECORD_VALIDATOR,
    )

    await _apply_validator(
        database,
        collection_name=(
            MASTERY_EVENTS_COLLECTION
        ),
        validator=MASTERY_EVENT_VALIDATOR,
    )

    await _apply_validator(
        database,
        collection_name=(
            QUIZ_SESSIONS_COLLECTION
        ),
        validator=QUIZ_SESSION_VALIDATOR,
    )