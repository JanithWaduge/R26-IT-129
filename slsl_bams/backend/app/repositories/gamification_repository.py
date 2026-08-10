from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.errors import (
    DuplicateKeyError,
)

from app.db.collections import (
    GAMIFICATION_EVENTS_COLLECTION,
    GAMIFICATION_PROFILES_COLLECTION,
)


class GamificationRepository:
    def __init__(
        self,
        database: Any,
    ) -> None:
        self.profiles = database[
            GAMIFICATION_PROFILES_COLLECTION
        ]

        self.events = database[
            GAMIFICATION_EVENTS_COLLECTION
        ]

    @staticmethod
    def parse_object_id(
        value: str,
    ) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise ValueError(
                "Invalid identifier."
            )

        return ObjectId(value)

    async def ensure_profile(
        self,
        *,
        student_id: ObjectId,
        xp_per_level: int,
    ) -> dict:
        now = datetime.now(timezone.utc)

        await self.profiles.update_one(
            {
                "student_id": student_id,
            },
            {
                "$setOnInsert": {
                    "student_id": student_id,
                    "total_xp": 0,
                    "level": 1,
                    "xp_into_level": 0,
                    "xp_to_next_level": (
                        xp_per_level
                    ),
                    "current_streak_days": 0,
                    "longest_streak_days": 0,
                    "last_activity_date": None,
                    "total_quizzes": 0,
                    "total_questions": 0,
                    "total_correct": 0,
                    "total_incorrect": 0,
                    "total_skipped": 0,
                    "total_hints": 0,
                    "total_retries": 0,
                    "perfect_quizzes": 0,
                    "unlocked_badges": [],
                    "processed_session_ids": [],
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

        profile = await self.profiles.find_one(
            {
                "student_id": student_id,
            }
        )

        if profile is None:
            raise RuntimeError(
                "Gamification profile could "
                "not be created."
            )

        return profile

    async def create_or_get_event(
        self,
        document: dict,
    ) -> dict:
        try:
            result = await self.events.insert_one(
                document
            )

            created = await self.events.find_one(
                {
                    "_id": result.inserted_id,
                }
            )

            if created is None:
                raise RuntimeError(
                    "Reward event could not "
                    "be retrieved."
                )

            return created

        except DuplicateKeyError:
            existing = await self.events.find_one(
                {
                    "session_id": document[
                        "session_id"
                    ],
                }
            )

            if existing is None:
                raise

            return existing

    async def replace_profile(
        self,
        *,
        profile: dict,
        expected_version: int,
        session_id: ObjectId,
    ) -> dict | None:
        replacement = dict(profile)

        replacement["version"] = (
            expected_version + 1
        )

        result = await self.profiles.replace_one(
            {
                "_id": replacement["_id"],
                "version": expected_version,
                "processed_session_ids": {
                    "$ne": session_id,
                },
            },
            replacement,
        )

        if result.matched_count == 0:
            return None

        return replacement

    async def mark_event_applied(
        self,
        *,
        session_id: ObjectId,
        badges_awarded: list[str],
        total_xp_after: int,
        level_after: int,
        streak_after: int,
        applied_at: datetime,
    ) -> dict:
        await self.events.update_one(
            {
                "session_id": session_id,
            },
            {
                "$set": {
                    "status": "applied",
                    "badges_awarded": (
                        badges_awarded
                    ),
                    "total_xp_after": (
                        total_xp_after
                    ),
                    "level_after": level_after,
                    "streak_after": (
                        streak_after
                    ),
                    "applied_at": applied_at,
                }
            },
        )

        event = await self.events.find_one(
            {
                "session_id": session_id,
            }
        )

        if event is None:
            raise RuntimeError(
                "Applied reward event could "
                "not be retrieved."
            )

        return event

    async def get_profile(
        self,
        *,
        student_id: ObjectId,
    ) -> dict | None:
        return await self.profiles.find_one(
            {
                "student_id": student_id,
            }
        )

    async def get_event(
        self,
        *,
        session_id: ObjectId,
    ) -> dict | None:
        return await self.events.find_one(
            {
                "session_id": session_id,
            }
        )

    async def list_events(
        self,
        *,
        student_id: ObjectId,
        limit: int,
    ) -> list[dict]:
        cursor = (
            self.events.find(
                {
                    "student_id": student_id,
                    "status": "applied",
                }
            )
            .sort("created_at", -1)
            .limit(limit)
        )

        return await cursor.to_list(
            length=limit
        )