from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.db.collections import (
    REFRESH_SESSIONS_COLLECTION,
    USERS_COLLECTION,
)
from app.schemas.auth import (
    AuthUserRecord,
    UserRole,
)


class DuplicateUserEmailError(Exception):
    pass


class AuthUserNotFoundError(Exception):
    pass


class AuthRepository:
    def __init__(self, database: Any) -> None:
        self.users = database[USERS_COLLECTION]
        self.refresh_sessions = database[
            REFRESH_SESSIONS_COLLECTION
        ]

    @staticmethod
    def normalise_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def parse_object_id(
        value: str,
    ) -> ObjectId:
        if not ObjectId.is_valid(value):
            raise AuthUserNotFoundError(
                "Authentication user was not found."
            )

        return ObjectId(value)

    @staticmethod
    def to_user_record(
        document: dict,
    ) -> AuthUserRecord:
        return AuthUserRecord(
            id=str(document["_id"]),
            student_id=str(document["student_id"]),
            email=document["email"],
            password_hash=document["password_hash"],
            role=document["role"],
            is_active=document["is_active"],
            token_version=document["token_version"],
            created_at=document["created_at"],
            updated_at=document["updated_at"],
        )

    async def create_user(
        self,
        *,
        student_id: str,
        email: str,
        password_hash: str,
    ) -> AuthUserRecord:
        now = datetime.now(timezone.utc)

        document = {
            "student_id": self.parse_object_id(
                student_id
            ),
            "email": self.normalise_email(email),
            "password_hash": password_hash,
            "role": UserRole.STUDENT.value,
            "is_active": True,
            "token_version": 0,
            "created_at": now,
            "updated_at": now,
        }

        try:
            result = await self.users.insert_one(
                document
            )
        except DuplicateKeyError as error:
            raise DuplicateUserEmailError(
                "An account with this email already exists."
            ) from error

        created = await self.users.find_one(
            {"_id": result.inserted_id}
        )

        if created is None:
            raise RuntimeError(
                "User was created but could not be retrieved."
            )

        return self.to_user_record(created)

    async def get_user_by_email(
        self,
        email: str,
    ) -> AuthUserRecord | None:
        document = await self.users.find_one(
            {
                "email": self.normalise_email(
                    email
                )
            }
        )

        if document is None:
            return None

        return self.to_user_record(document)

    async def get_user_by_id(
        self,
        user_id: str,
    ) -> AuthUserRecord | None:
        if not ObjectId.is_valid(user_id):
            return None

        document = await self.users.find_one(
            {"_id": ObjectId(user_id)}
        )

        if document is None:
            return None

        return self.to_user_record(document)

    async def create_refresh_session(
        self,
        *,
        user_id: str,
        jti_hash: str,
        family_id: str,
        expires_at: datetime,
    ) -> None:
        await self.refresh_sessions.insert_one(
            {
                "user_id": self.parse_object_id(
                    user_id
                ),
                "jti_hash": jti_hash,
                "family_id": family_id,
                "expires_at": expires_at,
                "revoked_at": None,
                "replaced_by_jti_hash": None,
                "created_at": datetime.now(
                    timezone.utc
                ),
            }
        )

    async def consume_refresh_session(
        self,
        *,
        user_id: str,
        jti_hash: str,
        replacement_jti_hash: str,
    ) -> bool:
        now = datetime.now(timezone.utc)

        consumed = (
            await self.refresh_sessions.find_one_and_update(
                {
                    "user_id": self.parse_object_id(
                        user_id
                    ),
                    "jti_hash": jti_hash,
                    "revoked_at": None,
                    "expires_at": {
                        "$gt": now,
                    },
                },
                {
                    "$set": {
                        "revoked_at": now,
                        "replaced_by_jti_hash": (
                            replacement_jti_hash
                        ),
                    }
                },
                return_document=ReturnDocument.BEFORE,
            )
        )

        return consumed is not None

    async def revoke_refresh_session(
        self,
        *,
        jti_hash: str,
    ) -> None:
        await self.refresh_sessions.update_one(
            {
                "jti_hash": jti_hash,
                "revoked_at": None,
            },
            {
                "$set": {
                    "revoked_at": datetime.now(
                        timezone.utc
                    ),
                }
            },
        )

    async def revoke_token_family(
        self,
        *,
        user_id: str,
        family_id: str,
    ) -> None:
        await self.refresh_sessions.update_many(
            {
                "user_id": self.parse_object_id(
                    user_id
                ),
                "family_id": family_id,
                "revoked_at": None,
            },
            {
                "$set": {
                    "revoked_at": datetime.now(
                        timezone.utc
                    ),
                }
            },
        )

    async def revoke_all_sessions(
        self,
        *,
        user_id: str,
    ) -> None:
        await self.refresh_sessions.update_many(
            {
                "user_id": self.parse_object_id(
                    user_id
                ),
                "revoked_at": None,
            },
            {
                "$set": {
                    "revoked_at": datetime.now(
                        timezone.utc
                    ),
                }
            },
        )

    async def increment_token_version(
        self,
        *,
        user_id: str,
    ) -> AuthUserRecord:
        document = await self.users.find_one_and_update(
            {
                "_id": self.parse_object_id(
                    user_id
                )
            },
            {
                "$inc": {
                    "token_version": 1,
                },
                "$set": {
                    "updated_at": datetime.now(
                        timezone.utc
                    ),
                },
            },
            return_document=ReturnDocument.AFTER,
        )

        if document is None:
            raise AuthUserNotFoundError(
                "Authentication user was not found."
            )

        return self.to_user_record(document)