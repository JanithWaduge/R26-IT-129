import asyncio

from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi

from app.core.config import get_settings


async def main() -> None:
    settings = get_settings()

    client = AsyncMongoClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=(
            settings.mongo_server_selection_timeout_ms
        ),
        server_api=ServerApi("1"),
    )

    try:
        result = await client.admin.command("ping")

        print("MongoDB connection successful")
        print(f"Database: {settings.mongo_database}")
        print(f"Ping response: {result}")

    except Exception as error:
        print("MongoDB connection failed")
        print(f"Reason: {error}")
        raise

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())