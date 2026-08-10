from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class QualityScoreResult:
    score: float
    band: int
    base_score: float
    retry_penalty: float
    hint_penalty: float
    response_time_penalty: float
    confidence_penalty: float
    response_target_ms: int

    def to_components(self) -> dict:
        return {
            "base_score": self.base_score,
            "retry_penalty": self.retry_penalty,
            "hint_penalty": self.hint_penalty,
            "response_time_penalty": (
                self.response_time_penalty
            ),
            "confidence_penalty": (
                self.confidence_penalty
            ),
            "response_target_ms": (
                self.response_target_ms
            ),
        }


def _response_target_ms(
    *,
    direction: str,
    difficulty: int,
) -> int:
    if direction == "productive":
        return 8000 + (difficulty * 2500)

    return 5000 + (difficulty * 1500)


def _response_time_penalty(
    *,
    response_time_ms: int | None,
    target_ms: int,
) -> float:
    if response_time_ms is None:
        return 1.0

    if response_time_ms <= target_ms:
        return 0.0

    if response_time_ms <= target_ms * 1.5:
        return 0.25

    if response_time_ms <= target_ms * 2:
        return 0.50

    return 1.0


def _confidence_penalty(
    *,
    direction: str,
    confidence: float | None,
    acceptance_threshold: float,
) -> float:
    if direction != "productive":
        return 0.0

    if confidence is None:
        return 1.5

    if confidence >= 0.90:
        return 0.0

    if confidence >= 0.80:
        return 0.25

    if confidence >= acceptance_threshold:
        return 0.50

    return 1.50


def calculate_quality_score(
    *,
    direction: str,
    question_status: str,
    final_correct: bool | None,
    retry_count: int,
    hint_count: int,
    response_time_ms: int | None,
    difficulty: int,
    recognition_confidence: float | None,
    productive_acceptance_threshold: float,
) -> QualityScoreResult:
    """
    Convert one finalized quiz question into a
    transparent performance-quality score from 0 to 5.

    The weights are provisional research parameters.
    They must be reviewed during pilot evaluation.
    """

    target_ms = _response_target_ms(
        direction=direction,
        difficulty=difficulty,
    )

    if question_status == "skipped":
        return QualityScoreResult(
            score=0.0,
            band=0,
            base_score=0.0,
            retry_penalty=0.0,
            hint_penalty=0.0,
            response_time_penalty=0.0,
            confidence_penalty=0.0,
            response_target_ms=target_ms,
        )

    if final_correct is not True:
        return QualityScoreResult(
            score=1.0,
            band=1,
            base_score=1.0,
            retry_penalty=0.0,
            hint_penalty=0.0,
            response_time_penalty=0.0,
            confidence_penalty=0.0,
            response_target_ms=target_ms,
        )

    base_score = 5.0

    retry_penalty = min(
        retry_count * 0.75,
        2.0,
    )

    hint_penalty = min(
        hint_count * 0.50,
        1.0,
    )

    response_penalty = (
        _response_time_penalty(
            response_time_ms=(
                response_time_ms
            ),
            target_ms=target_ms,
        )
    )

    confidence_penalty = (
        _confidence_penalty(
            direction=direction,
            confidence=(
                recognition_confidence
            ),
            acceptance_threshold=(
                productive_acceptance_threshold
            ),
        )
    )

    score = (
        base_score
        - retry_penalty
        - hint_penalty
        - response_penalty
        - confidence_penalty
    )

    score = round(
        max(0.0, min(5.0, score)),
        2,
    )

    band = max(
        0,
        min(
            5,
            floor(score + 0.5),
        ),
    )

    return QualityScoreResult(
        score=score,
        band=band,
        base_score=base_score,
        retry_penalty=retry_penalty,
        hint_penalty=hint_penalty,
        response_time_penalty=(
            response_penalty
        ),
        confidence_penalty=(
            confidence_penalty
        ),
        response_target_ms=target_ms,
    )