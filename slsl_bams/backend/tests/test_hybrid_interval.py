from datetime import datetime, timezone

from app.algorithms.hybrid_interval import calculate_hybrid_interval

NOW = datetime(2026, 8, 2, tzinfo=timezone.utc)


def calculate(probability: float, interval: int, maximum: int = 365):
    return calculate_hybrid_interval(
        recall_probability=probability, base_interval_days=interval,
        reviewed_at=NOW, minimum_factor=0.50, maximum_factor=1.15,
        maximum_interval_days=maximum,
    )


def test_high_risk_shortens_interval() -> None:
    result = calculate(0.20, 10)
    assert result.adjustment_factor == 0.50
    assert result.final_interval_days == 5


def test_good_retention_keeps_interval() -> None:
    result = calculate(0.75, 10)
    assert result.adjustment_factor == 1.00
    assert result.final_interval_days == 10


def test_strong_retention_has_bounded_extension() -> None:
    result = calculate(0.95, 20)
    assert result.adjustment_factor == 1.15
    assert result.final_interval_days == 23


def test_interval_never_becomes_zero() -> None:
    assert calculate(0.01, 1).final_interval_days == 1


def test_maximum_interval_is_enforced() -> None:
    assert calculate(0.99, 365).final_interval_days == 365
