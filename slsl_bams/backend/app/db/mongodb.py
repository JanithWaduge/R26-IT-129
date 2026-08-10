import logging

from fastapi import FastAPI
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi

from app.db.migrations.runner import run_migrations

logger = logging.getLogger(__name__)


async def connect_to_mongodb(app: FastAPI) -> None:
    """
    Connect to MongoDB, verify the connection and apply migrations.
    """

    settings = app.state.settings

    client = AsyncMongoClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=(
            settings.mongo_server_selection_timeout_ms
        ),
        server_api=ServerApi("1"),
    )

    try:
        await client.admin.command("ping")

        database = client[settings.mongo_database]

        app.state.mongodb_client = client
        app.state.database = database

        if settings.run_migrations_on_startup:
            await run_migrations(database)

        logger.info(
            "Connected to MongoDB database '%s'.",
            settings.mongo_database,
        )

    except Exception:
        await client.close()
        logger.exception("MongoDB startup connection failed.")
        raise


async def close_mongodb_connection(app: FastAPI) -> None:
    """Close the MongoDB client during application shutdown."""

    client = getattr(app.state, "mongodb_client", None)

    if client is None:
        return

    settings = app.state.settings

    if settings.drop_database_on_shutdown:
        await client.drop_database(settings.mongo_database)

    await client.close()

    logger.info("MongoDB connection closed.")
