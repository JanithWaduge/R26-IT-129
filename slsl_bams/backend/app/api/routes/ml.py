from bson import ObjectId
from fastapi import APIRouter, Depends, Query, Request

from app.api.dependencies.auth import CurrentUserDependency
from app.db.dependencies import get_database
from app.ml.model_runtime import RecallModelRuntime
from app.repositories.hybrid_scheduler_repository import HybridSchedulerRepository
from app.schemas.ml import MLPredictionHistoryItemResponse, MLRuntimeStatusResponse

router = APIRouter()


@router.get("/status", response_model=MLRuntimeStatusResponse)
async def get_ml_status(
    request: Request, current_user: CurrentUserDependency,
) -> MLRuntimeStatusResponse:
    del current_user
    settings = request.app.state.settings
    status = RecallModelRuntime(settings).status()
    return MLRuntimeStatusResponse(
        enabled=settings.ml_scheduler_enabled, status=status.status,
        model_name=status.model_name, model_version=status.model_version,
        synthetic_model=status.synthetic_model, message=status.message,
    )


@router.get("/predictions", response_model=list[MLPredictionHistoryItemResponse])
async def list_my_predictions(
    current_user: CurrentUserDependency, database=Depends(get_database),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[MLPredictionHistoryItemResponse]:
    documents = await HybridSchedulerRepository(database).list_student_predictions(
        student_id=ObjectId(current_user.student_id), limit=limit
    )
    return [MLPredictionHistoryItemResponse(
        event_id=item["event_id"], sign_id=str(item["sign_id"]),
        status=item["status"], recall_probability=item["recall_probability"],
        risk_band=item["risk_band"], base_interval_days=item["base_interval_days"],
        adjustment_factor=item["adjustment_factor"],
        final_interval_days=item["final_interval_days"],
        model_version=item["model_version"], synthetic_model=item["synthetic_model"],
        created_at=item["created_at"],
    ) for item in documents]
