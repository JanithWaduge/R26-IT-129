from app.core.config import Settings
from app.repositories.sign_repository import (
    SignRepository,
)
from app.schemas.sign import (
    PaginatedSignsResponse,
    SignDetailResponse,
)


class SignService:
    def __init__(
        self,
        *,
        repository: SignRepository,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.settings = settings

    def _visible_statuses(
        self,
    ) -> list[str]:
        if self.settings.environment in {
            "development",
            "test",
        }:
            return [
                "approved",
                "development",
            ]

        return ["approved"]

    async def list_signs(
        self,
        *,
        page: int,
        page_size: int,
        category_id: str | None,
        competency_id: str | None,
        difficulty: int | None,
        search: str | None,
    ) -> PaginatedSignsResponse:
        return await self.repository.list_signs(
            visible_statuses=(
                self._visible_statuses()
            ),
            page=page,
            page_size=page_size,
            category_id=category_id,
            competency_id=competency_id,
            difficulty=difficulty,
            search=search,
        )

    async def get_sign(
        self,
        sign_id: str,
    ) -> SignDetailResponse:
        return await self.repository.get_sign(
            sign_id=sign_id,
            visible_statuses=(
                self._visible_statuses()
            ),
        )
