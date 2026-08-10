from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from app.algorithms.mastery_update import (
    calculate_combined_state,
    initial_direction_state,
)
from app.db.collections import (
    MASTERY_EVENTS_COLLECTION,
    MASTERY_RECORDS_COLLECTION,
    SIGNS_COLLECTION,
)


class MasteryRecordNotFoundError(Exception):
    pass


class MasteryConcurrentUpdateError(Exception):
    pass


class MasteryRepository:
    def __init__(
        self,
        database: Any,
    ) -> None:
        self.records = database[
            MASTERY_RECORDS_COLLECTION
        ]

        self.events = database[
            MASTERY_EVENTS_COLLECTION
        ]

        self.signs = database[
            SIGNS_COLLECTION
        ]

    @staticmethod
    def parse_object_id(
        value: str,
        *,
        message: str,
    ) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise MasteryRecordNotFoundError(
                message
            )

        return ObjectId(value)

    async def create_or_get_event(
        self,
        event_document: dict,
    ) -> dict:
        try:
            await self.events.insert_one(
                event_document
            )
        except DuplicateKeyError:
            pass

        event = await self.events.find_one(
            {
                "event_id": event_document[
                    "event_id"
                ]
            }
        )

        if event is None:
            raise RuntimeError(
                "Mastery event could not "
                "be created or retrieved."
            )

        return event

    async def mark_event_applied(
        self,
        *,
        event_id: str,
        before_state: dict,
        after_state: dict,
        applied_at: datetime,
    ) -> dict:
        await self.events.update_one(
            {
                "event_id": event_id,
            },
            {
                "$set": {
                    "status": "applied",
                    "before_state": before_state,
                    "after_state": after_state,
                    "applied_at": applied_at,
                }
            },
        )

        event = await self.events.find_one(
            {
                "event_id": event_id,
            }
        )

        if event is None:
            raise RuntimeError(
                "Applied mastery event "
                "could not be retrieved."
            )

        return event

    async def ensure_mastery_record(
        self,
        *,
        student_id: ObjectId,
        sign_id: ObjectId,
    ) -> dict:
        now = datetime.now(timezone.utc)

        receptive = initial_direction_state()
        productive = initial_direction_state()

        combined_score, overall_status, balance = (
            calculate_combined_state(
                receptive=receptive,
                productive=productive,
            )
        )

        await self.records.update_one(
            {
                "student_id": student_id,
                "sign_id": sign_id,
            },
            {
                "$setOnInsert": {
                    "student_id": student_id,
                    "sign_id": sign_id,
                    "receptive": receptive,
                    "productive": productive,
                    "combined_score": (
                        combined_score
                    ),
                    "overall_status": (
                        overall_status
                    ),
                    "balance_status": balance,
                    "processed_events": [],
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

        record = await self.records.find_one(
            {
                "student_id": student_id,
                "sign_id": sign_id,
            }
        )

        if record is None:
            raise RuntimeError(
                "Mastery record could not "
                "be created or retrieved."
            )

        return record

    async def apply_mastery_update(
        self,
        *,
        current_record: dict,
        event_id: str,
        direction: str,
        updated_direction_state: dict,
        quality_score: float,
        quality_band: int,
        applied_at: datetime,
    ) -> dict | None:
        replacement = deepcopy(
            current_record
        )

        if any(
            event["event_id"] == event_id
            for event in replacement[
                "processed_events"
            ]
        ):
            return replacement

        before_score = float(
            replacement[direction][
                "score"
            ]
        )

        after_score = float(
            updated_direction_state[
                "score"
            ]
        )

        replacement[direction] = (
            updated_direction_state
        )

        (
            combined_score,
            overall_status,
            balance_status,
        ) = calculate_combined_state(
            receptive=replacement[
                "receptive"
            ],
            productive=replacement[
                "productive"
            ],
        )

        replacement["combined_score"] = (
            combined_score
        )

        replacement["overall_status"] = (
            overall_status
        )

        replacement["balance_status"] = (
            balance_status
        )

        replacement[
            "processed_events"
        ].append(
            {
                "event_id": event_id,
                "direction": direction,
                "quality_score": (
                    quality_score
                ),
                "quality_band": (
                    quality_band
                ),
                "before_score": before_score,
                "after_score": after_score,
                "applied_at": applied_at,
            }
        )

        expected_version = current_record[
            "version"
        ]

        replacement["version"] = (
            expected_version + 1
        )

        replacement["updated_at"] = (
            applied_at
        )

        result = await self.records.replace_one(
            {
                "_id": current_record["_id"],
                "version": expected_version,
                "processed_events.event_id": {
                    "$ne": event_id,
                },
            },
            replacement,
        )

        if result.matched_count == 0:
            return None

        return replacement

    async def get_record(
        self,
        *,
        student_id: ObjectId,
        sign_id: ObjectId,
    ) -> dict | None:
        return await self.records.find_one(
            {
                "student_id": student_id,
                "sign_id": sign_id,
            }
        )

    async def get_records_for_student(
        self,
        *,
        student_id: ObjectId,
    ) -> list[dict]:
        cursor = self.records.find(
            {
                "student_id": student_id,
            }
        )

        return await cursor.to_list(
            length=None
        )

    async def list_visible_signs(
        self,
        *,
        visible_statuses: list[str],
        category_id: str | None = None,
    ) -> list[dict]:
        query: dict[str, Any] = {
            "is_active": True,
            "content_status": {
                "$in": visible_statuses,
            },
        }

        if category_id is not None:
            query["category_id"] = (
                self.parse_object_id(
                    category_id,
                    message=(
                        "Category identifier "
                        "is invalid."
                    ),
                )
            )

        cursor = (
            self.signs
            .find(query)
            .sort(
                [
                    (
                        "difficulty",
                        ASCENDING,
                    ),
                    (
                        "gloss",
                        ASCENDING,
                    ),
                ]
            )
        )

        return await cursor.to_list(
            length=None
        )

    async def get_visible_sign(
        self,
        *,
        sign_id: str,
        visible_statuses: list[str],
    ) -> dict:
        object_id = self.parse_object_id(
            sign_id,
            message="Sign was not found.",
        )

        sign = await self.signs.find_one(
            {
                "_id": object_id,
                "is_active": True,
                "content_status": {
                    "$in": visible_statuses,
                },
            }
        )

        if sign is None:
            raise MasteryRecordNotFoundError(
                "Sign was not found."
            )

        return sign