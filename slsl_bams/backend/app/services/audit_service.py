from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId

from app.db.collections import SECURITY_AUDIT_EVENTS_COLLECTION

BLOCKED_DETAIL_KEYS = {
    "password", "token", "access_token", "refresh_token", "authorization",
    "jwt", "secret", "mongo_uri",
}


class AuditService:
    def __init__(self, *, database: Any, retention_days: int) -> None:
        self.collection = database[SECURITY_AUDIT_EVENTS_COLLECTION]
        self.retention_days = retention_days

    async def record(
        self, *, event_type: str, outcome: str, actor_id: str | None = None,
        request_id: str | None = None, resource_type: str | None = None,
        resource_id: str | None = None, details: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        safe_details = {
            str(key): value for key, value in (details or {}).items()
            if str(key).lower() not in BLOCKED_DETAIL_KEYS
        }
        object_id = ObjectId(actor_id) if actor_id and ObjectId.is_valid(actor_id) else None
        await self.collection.insert_one({
            "event_type": event_type, "actor_id": object_id, "request_id": request_id,
            "resource_type": resource_type, "resource_id": resource_id,
            "outcome": outcome, "details": safe_details, "created_at": now,
            "expires_at": now + timedelta(days=self.retention_days),
        })
