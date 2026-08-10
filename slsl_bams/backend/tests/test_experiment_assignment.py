import pytest

from app.algorithms.experiment_assignment import assign_experiment_arm

SECRET = "test-experiment-assignment-secret-longer-than-thirty-two-characters"


def assignment(student: str):
    return assign_experiment_arm(
        experiment_name="experiment-v1", student_id=student,
        preferred_language="english", grade_level="Grade 8", secret=SECRET,
    )


def test_assignment_is_deterministic() -> None:
    assert assignment("student-1") == assignment("student-1")


def test_assignment_uses_valid_arm() -> None:
    assert assignment("student-2").arm in {
        "random_control", "modified_sm2", "hybrid_ml",
    }


def test_secret_must_be_long() -> None:
    with pytest.raises(ValueError):
        assign_experiment_arm(
            experiment_name="experiment-v1", student_id="student-3",
            preferred_language="english", grade_level=None, secret="short",
        )
