import asyncio

from pymongo import AsyncMongoClient

from app.core.config import get_settings
from app.db.migrations.runner import run_migrations


async def main() -> None:
    settings = get_settings()
    client = AsyncMongoClient(settings.mongo_uri, serverSelectionTimeoutMS=10000)
    try:
        await client.admin.command("ping")
        await run_migrations(client[settings.mongo_database])
        print("Database migrations completed.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
