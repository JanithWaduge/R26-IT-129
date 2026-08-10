import asyncio

from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi

from app.core.config import get_settings
from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    CURRICULUM_COMPETENCIES_COLLECTION,
    SIGNS_COLLECTION,
)
from app.db.migrations.runner import run_migrations
from app.db.seed_data import build_seed_documents


async def _upsert_documents(
    collection,
    documents: list[dict],
) -> None:
    for document in documents:
        await collection.replace_one(
            {
                "_id": document["_id"],
            },
            document,
            upsert=True,
        )


async def main() -> None:
    settings = get_settings()

    client = AsyncMongoClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=(
            settings
            .mongo_server_selection_timeout_ms
        ),
        server_api=ServerApi("1"),
    )

    try:
        await client.admin.command("ping")

        database = client[
            settings.mongo_database
        ]

        await run_migrations(database)

        seed_data = build_seed_documents()

        await _upsert_documents(
            database[
                CURRICULUM_CATEGORIES_COLLECTION
            ],
            seed_data["categories"],
        )

        await _upsert_documents(
            database[
                CURRICULUM_COMPETENCIES_COLLECTION
            ],
            seed_data["competencies"],
        )

        await _upsert_documents(
            database[SIGNS_COLLECTION],
            seed_data["signs"],
        )

        print("Learning-content seed completed.")
        print(
            "Categories:",
            len(seed_data["categories"]),
        )
        print(
            "Competencies:",
            len(seed_data["competencies"]),
        )
        print(
            "Development signs:",
            len(seed_data["signs"]),
        )
        print(
            "Important: all seeded learning "
            "records remain provisional."
        )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())