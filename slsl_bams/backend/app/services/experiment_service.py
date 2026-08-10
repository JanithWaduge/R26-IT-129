from dataclasses import dataclass
from datetime import datetime, timezone

from bson import ObjectId

from app.algorithms.experiment_assignment import assign_experiment_arm
from app.core.config import Settings
from app.repositories.experiment_repository import ExperimentRepository
from app.services.audit_service import AuditService


@dataclass(frozen=True)
class ResolvedExperimentAssignment:
    experiment_name: str | None
    arm: str


class ExperimentService:
    def __init__(self, *, repository: ExperimentRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings

    async def resolve_assignment(self, *, student_id: str) -> ResolvedExperimentAssignment:
        if not self.settings.experiment_enabled:
            return ResolvedExperimentAssignment(None, "not_enrolled")
        if not await self.repository.has_active_consent(
            experiment_name=self.settings.experiment_name, student_id=student_id
        ):
            return ResolvedExperimentAssignment(None, "not_enrolled")
        secret = self.settings.experiment_assignment_secret
        if secret is None:
            raise RuntimeError(
                "EXPERIMENT_ASSIGNMENT_SECRET is required when the experiment is enabled."
            )
        existing = await self.repository.get_assignment(
            experiment_name=self.settings.experiment_name, student_id=student_id
        )
        if existing is not None:
            return ResolvedExperimentAssignment(existing["experiment_name"], existing["arm"])
        student = await self.repository.get_student_context(student_id)
        assignment = assign_experiment_arm(
            experiment_name=self.settings.experiment_name, student_id=student_id,
            preferred_language=str(student.get("preferred_language", "unknown")),
            grade_level=student.get("grade_level"), secret=secret.get_secret_value(),
        )
        now = datetime.now(timezone.utc)
        document = await self.repository.create_or_get_assignment({
            "experiment_name": self.settings.experiment_name,
            "student_id": ObjectId(student_id), "arm": assignment.arm,
            "stratum": assignment.stratum, "assignment_hash": assignment.assignment_hash,
            "is_active": True, "assigned_at": now, "created_at": now, "updated_at": now,
        })
        await AuditService(
            database=self.repository.database,
            retention_days=self.settings.audit_retention_days,
        ).record(
            event_type="experiment_assignment", outcome="success", actor_id=student_id,
            resource_type="experiment", resource_id=self.settings.experiment_name,
            details={"arm": document["arm"]},
        )
        return ResolvedExperimentAssignment(document["experiment_name"], document["arm"])
