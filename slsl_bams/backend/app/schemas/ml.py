from datetime import datetime

from pydantic import BaseModel


class MLRuntimeStatusResponse(BaseModel):
    enabled: bool
    status: str
    model_name: str | None
    model_version: str | None
    synthetic_model: bool
    message: str | None


class MLPredictionHistoryItemResponse(BaseModel):
    event_id: str
    sign_id: str
    status: str
    recall_probability: float | None
    risk_band: str | None
    base_interval_days: int
    adjustment_factor: float | None
    final_interval_days: int
    model_version: str | None
    synthetic_model: bool
    created_at: datetime
