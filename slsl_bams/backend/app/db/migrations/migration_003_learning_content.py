from typing import Any

from pymongo import ASCENDING, IndexModel

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    CURRICULUM_COMPETENCIES_COLLECTION,
    SIGNS_COLLECTION,
)
from app.db.schema import (
    CURRICULUM_CATEGORY_VALIDATOR,
    CURRICULUM_COMPETENCY_VALIDATOR,
    SIGN_VALIDATOR,
)

VERSION = 3
NAME = "Create curriculum categories, competencies and signs"


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
        CURRICULUM_CATEGORIES_COLLECTION,
        CURRICULUM_CATEGORY_VALIDATOR,
    )

    await _create_or_update_collection(
        database,
        CURRICULUM_COMPETENCIES_COLLECTION,
        CURRICULUM_COMPETENCY_VALIDATOR,
    )

    await _create_or_update_collection(
        database,
        SIGNS_COLLECTION,
        SIGN_VALIDATOR,
    )

    categories = database[
        CURRICULUM_CATEGORIES_COLLECTION
    ]

    await categories.create_indexes(
        [
            IndexModel(
                [("code", ASCENDING)],
                name="uq_curriculum_category_code",
                unique=True,
            ),
            IndexModel(
                [
                    ("is_active", ASCENDING),
                    ("display_order", ASCENDING),
                ],
                name="ix_category_active_order",
            ),
            IndexModel(
                [("validation_status", ASCENDING)],
                name="ix_category_validation_status",
            ),
        ]
    )

    competencies = database[
        CURRICULUM_COMPETENCIES_COLLECTION
    ]

    await competencies.create_indexes(
        [
            IndexModel(
                [("code", ASCENDING)],
                name="uq_curriculum_competency_code",
                unique=True,
            ),
            IndexModel(
                [
                    ("category_id", ASCENDING),
                    ("is_active", ASCENDING),
                    ("display_order", ASCENDING),
                ],
                name="ix_competency_category_active_order",
            ),
            IndexModel(
                [("grade_levels", ASCENDING)],
                name="ix_competency_grade_levels",
            ),
        ]
    )

    signs = database[SIGNS_COLLECTION]

    await signs.create_indexes(
        [
            IndexModel(
                [("code", ASCENDING)],
                name="uq_sign_code",
                unique=True,
            ),
            IndexModel(
                [
                    ("category_id", ASCENDING),
                    ("is_active", ASCENDING),
                    ("content_status", ASCENDING),
                    ("difficulty", ASCENDING),
                ],
                name="ix_sign_category_visibility_difficulty",
            ),
            IndexModel(
                [
                    ("competency_ids", ASCENDING),
                    ("is_active", ASCENDING),
                ],
                name="ix_sign_competency_active",
            ),
            IndexModel(
                [("search_terms", ASCENDING)],
                name="ix_sign_search_terms",
            ),
            IndexModel(
                [("tags", ASCENDING)],
                name="ix_sign_tags",
            ),
            IndexModel(
                [("prerequisite_sign_ids", ASCENDING)],
                name="ix_sign_prerequisites",
            ),
        ]
    )