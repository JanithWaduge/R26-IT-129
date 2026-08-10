from app.repositories.student_repository import StudentRepository
from app.schemas.student import (
    StudentCreate,
    StudentResponse,
    StudentUpdate,
)


class StudentService:
    def __init__(
        self,
        repository: StudentRepository,
    ) -> None:
        self.repository = repository

    async def create_student(
        self,
        student_data: StudentCreate,
    ) -> StudentResponse:
        return await self.repository.create(student_data)

    async def get_student(
        self,
        student_id: str,
    ) -> StudentResponse:
        return await self.repository.get_by_id(student_id)

    async def list_students(
        self,
        *,
        skip: int,
        limit: int,
        active_only: bool,
    ) -> list[StudentResponse]:
        return await self.repository.list_students(
            skip=skip,
            limit=limit,
            active_only=active_only,
        )

    async def update_student(
        self,
        student_id: str,
        student_data: StudentUpdate,
    ) -> StudentResponse:
        return await self.repository.update(
            student_id=student_id,
            student_data=student_data,
        )

    async def deactivate_student(
        self,
        student_id: str,
    ) -> None:
        await self.repository.deactivate(student_id)