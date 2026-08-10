from dataclasses import dataclass
import hashlib
import hmac

EXPERIMENT_ARMS = ("random_control", "modified_sm2", "hybrid_ml")


@dataclass(frozen=True)
class ExperimentAssignment:
    arm: str
    stratum: str
    assignment_hash: str


def assign_experiment_arm(
    *, experiment_name: str, student_id: str, preferred_language: str,
    grade_level: str | None, secret: str,
) -> ExperimentAssignment:
    if len(secret) < 32:
        raise ValueError("Experiment assignment secret must contain at least 32 characters.")
    language = preferred_language.strip().lower() or "unknown"
    grade = grade_level.strip().lower() if grade_level else "unknown"
    stratum = f"{language}|{grade}"
    message = f"{experiment_name}:{stratum}:{student_id}".encode()
    assignment_hash = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    arm = EXPERIMENT_ARMS[int(assignment_hash[:16], 16) % len(EXPERIMENT_ARMS)]
    return ExperimentAssignment(arm, stratum, assignment_hash)
