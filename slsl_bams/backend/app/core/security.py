import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt
from fastapi import FastAPI, Request
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import Settings
from app.schemas.auth import TokenPayload


password_hasher = PasswordHash.recommended()


def configure_security(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    if settings.force_https:
        app.add_middleware(HTTPSRedirectMiddleware)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers.update({
            "X-Request-ID": request_id,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Cache-Control": "no-store",
        })
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class SecurityTokenError(Exception):
    """Raised when a JWT cannot be trusted."""


@dataclass(frozen=True)
class CreatedToken:
    token: str
    token_id: str
    token_id_hash: str
    expires_at: datetime
    family_id: str | None = None


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    try:
        return password_hasher.verify(
            password,
            password_hash,
        )
    except Exception:
        return False


def hash_token_identifier(
    token_id: str,
) -> str:
    return hashlib.sha256(
        token_id.encode("utf-8")
    ).hexdigest()


def create_access_token(
    *,
    settings: Settings,
    user_id: str,
    role: str,
    token_version: int,
) -> CreatedToken:
    now = datetime.now(timezone.utc)

    expires_at = now + timedelta(
        minutes=settings.access_token_expire_minutes
    )

    token_id = str(uuid4())

    payload = {
        "sub": user_id,
        "typ": "access",
        "jti": token_id,
        "ver": token_version,
        "role": role,
        "iat": now,
        "exp": expires_at,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }

    encoded_token = jwt.encode(
        payload,
        settings.jwt_access_secret,
        algorithm=settings.jwt_algorithm,
    )

    return CreatedToken(
        token=encoded_token,
        token_id=token_id,
        token_id_hash=hash_token_identifier(
            token_id
        ),
        expires_at=expires_at,
    )


def create_refresh_token(
    *,
    settings: Settings,
    user_id: str,
    token_version: int,
    family_id: str | None = None,
) -> CreatedToken:
    now = datetime.now(timezone.utc)

    expires_at = now + timedelta(
        days=settings.refresh_token_expire_days
    )

    token_id = str(uuid4())
    resolved_family_id = family_id or str(uuid4())

    payload = {
        "sub": user_id,
        "typ": "refresh",
        "jti": token_id,
        "family_id": resolved_family_id,
        "ver": token_version,
        "iat": now,
        "exp": expires_at,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }

    encoded_token = jwt.encode(
        payload,
        settings.jwt_refresh_secret,
        algorithm=settings.jwt_algorithm,
    )

    return CreatedToken(
        token=encoded_token,
        token_id=token_id,
        token_id_hash=hash_token_identifier(
            token_id
        ),
        expires_at=expires_at,
        family_id=resolved_family_id,
    )


def decode_token(
    *,
    token: str,
    expected_type: str,
    settings: Settings,
) -> TokenPayload:
    if expected_type == "access":
        secret = settings.jwt_access_secret
    elif expected_type == "refresh":
        secret = settings.jwt_refresh_secret
    else:
        raise SecurityTokenError(
            "Unsupported token type."
        )

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={
                "require": [
                    "sub",
                    "typ",
                    "jti",
                    "ver",
                    "iat",
                    "exp",
                    "iss",
                    "aud",
                ]
            },
        )
    except InvalidTokenError as error:
        raise SecurityTokenError(
            "Token is invalid or expired."
        ) from error

    if payload.get("typ") != expected_type:
        raise SecurityTokenError(
            "Incorrect token type."
        )

    subject = payload.get("sub")
    token_id = payload.get("jti")
    token_version = payload.get("ver")
    expires_at = payload.get("exp")

    if (
        not isinstance(subject, str)
        or not isinstance(token_id, str)
        or not isinstance(token_version, int)
        or not isinstance(expires_at, int)
    ):
        raise SecurityTokenError(
            "Token payload is invalid."
        )

    return TokenPayload(
        subject=subject,
        token_type=expected_type,
        token_id=token_id,
        token_version=token_version,
        role=payload.get("role"),
        family_id=payload.get("family_id"),
        expires_at=datetime.fromtimestamp(
            expires_at,
            tz=timezone.utc,
        ),
    )
