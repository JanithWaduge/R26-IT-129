from app.repositories.curriculum_repository import (
    CurriculumRepository,
)
from app.schemas.curriculum import (
    CurriculumCategoryResponse,
    CurriculumCompetencyResponse,
)


class CurriculumService:
    def __init__(
        self,
        repository: CurriculumRepository,
    ) -> None:
        self.repository = repository

    async def list_categories(
        self,
    ) -> list[
        CurriculumCategoryResponse
    ]:
        return await (
            self.repository.list_categories()
        )

    async def get_category(
        self,
        category_id: str,
    ) -> CurriculumCategoryResponse:
        return await (
            self.repository.get_category(
                category_id
            )
        )

    async def list_competencies(
        self,
        category_id: str | None,
    ) -> list[
        CurriculumCompetencyResponse
    ]:
        return await (
            self.repository
            .list_competencies(
                category_id=category_id
            )
        )