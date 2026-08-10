from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class HybridIntervalDecision:
    recall_probability: float
    risk_band: str
    base_interval_days: int
    adjustment_factor: float
    final_interval_days: int
    base_next_review_at: datetime
    final_next_review_at: datetime


def recall_adjustment_factor(probability: float) -> tuple[float, str]:
    probability = max(0.0, min(1.0, probability))
    if probability < 0.30:
        return 0.50, "very_high_risk"
    if probability < 0.50:
        return 0.70, "high_risk"
    if probability < 0.70:
        return 0.90, "moderate_risk"
    if probability < 0.85:
        return 1.00, "good_retention"
    return 1.15, "strong_retention"


def calculate_hybrid_interval(
    *, recall_probability: float, base_interval_days: int,
    reviewed_at: datetime, minimum_factor: float, maximum_factor: float,
    maximum_interval_days: int,
) -> HybridIntervalDecision:
    base = max(1, int(base_interval_days))
    raw_factor, risk_band = recall_adjustment_factor(recall_probability)
    factor = max(minimum_factor, min(maximum_factor, raw_factor))
    final = max(1, min(maximum_interval_days, round(base * factor)))
    return HybridIntervalDecision(
        round(max(0.0, min(1.0, recall_probability)), 6), risk_band, base,
        round(factor, 4), final, reviewed_at + timedelta(days=base),
        reviewed_at + timedelta(days=final),
    )
