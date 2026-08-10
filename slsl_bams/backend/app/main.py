from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.security import configure_security
from app.db.mongodb import (
    close_mongodb_connection,
    connect_to_mongodb,
)


def create_app(
    settings: Settings | None = None,
) -> FastAPI:
    resolved_settings = (
        settings or get_settings()
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        app.state.settings = (
            resolved_settings
        )

        await connect_to_mongodb(app)

        try:
            yield
        finally:
            await close_mongodb_connection(
                app
            )

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        debug=resolved_settings.debug,
        docs_url="/docs" if resolved_settings.docs_enabled else None,
        redoc_url="/redoc" if resolved_settings.docs_enabled else None,
        openapi_url="/openapi.json" if resolved_settings.docs_enabled else None,
        lifespan=lifespan,
    )

    app.state.settings = resolved_settings
    configure_security(app, resolved_settings)

    @app.get("/", summary="Get API status")
    async def root() -> dict[str, str]:
        return {
            "message": "SLSL-BAMS API is running",
            "version": resolved_settings.app_version,
            "database": "MongoDB",
        }

    video_directory = Path(
        resolved_settings
        .sign_video_directory
    )

    video_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    app.mount(
        resolved_settings
        .sign_video_url_prefix,
        StaticFiles(
            directory=str(
                video_directory
            ),
        ),
        name="sign-videos",
    )

    app.include_router(
        api_router,
        prefix=(
            resolved_settings
            .api_v1_prefix
        ),
    )

    return app


app = create_app()
