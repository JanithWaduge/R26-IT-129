from fastapi.testclient import TestClient


REGISTER_PAYLOAD = {
    "full_name": "Research Student",
    "email": "research.student@example.com",
    "password": "SecurePass123",
    "preferred_language": "sinhala",
    "grade_level": "Grade 8",
}


def register_student(
    client: TestClient,
) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json=REGISTER_PAYLOAD,
    )

    assert response.status_code == 201

    return response.json()


def test_registration_returns_token_pair(
    client: TestClient,
) -> None:
    tokens = register_student(client)

    assert tokens["token_type"] == "bearer"
    assert tokens["access_token"]
    assert tokens["refresh_token"]
    assert tokens["access_token"] != (
        tokens["refresh_token"]
    )
    assert tokens["expires_in"] == 900

    assert "password" not in tokens
    assert "password_hash" not in tokens


def test_duplicate_registration_is_rejected(
    client: TestClient,
) -> None:
    register_student(client)

    duplicate = client.post(
        "/api/v1/auth/register",
        json=REGISTER_PAYLOAD,
    )

    assert duplicate.status_code == 409


def test_invalid_password_is_rejected(
    client: TestClient,
) -> None:
    register_student(client)

    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": (
                REGISTER_PAYLOAD["email"]
            ),
            "password": "WrongPassword999",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "Invalid email or password."
    )


def test_login_and_protected_profile(
    client: TestClient,
) -> None:
    register_student(client)

    login = client.post(
        "/api/v1/auth/login",
        data={
            "username": (
                REGISTER_PAYLOAD["email"]
            ),
            "password": (
                REGISTER_PAYLOAD["password"]
            ),
        },
    )

    assert login.status_code == 200

    access_token = login.json()[
        "access_token"
    ]

    profile = client.get(
        "/api/v1/auth/me",
        headers={
            "Authorization": (
                f"Bearer {access_token}"
            )
        },
    )

    assert profile.status_code == 200

    body = profile.json()

    assert body["email"] == (
        REGISTER_PAYLOAD["email"]
    )
    assert body["full_name"] == (
        REGISTER_PAYLOAD["full_name"]
    )
    assert body["role"] == "student"


def test_protected_route_requires_token(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/auth/me"
    )

    assert response.status_code == 401


def test_refresh_token_rotation_and_reuse_detection(
    client: TestClient,
) -> None:
    original_tokens = register_student(
        client
    )

    old_refresh = original_tokens[
        "refresh_token"
    ]

    refresh_response = client.post(
        "/api/v1/auth/refresh",
        json={
            "refresh_token": old_refresh,
        },
    )

    assert refresh_response.status_code == 200

    new_tokens = refresh_response.json()

    assert new_tokens["refresh_token"] != (
        old_refresh
    )

    reuse_response = client.post(
        "/api/v1/auth/refresh",
        json={
            "refresh_token": old_refresh,
        },
    )

    assert reuse_response.status_code == 401

    new_refresh_after_reuse = client.post(
        "/api/v1/auth/refresh",
        json={
            "refresh_token": (
                new_tokens["refresh_token"]
            ),
        },
    )

    assert new_refresh_after_reuse.status_code == 401


def test_logout_revokes_refresh_token(
    client: TestClient,
) -> None:
    tokens = register_student(client)

    logout = client.post(
        "/api/v1/auth/logout",
        json={
            "refresh_token": (
                tokens["refresh_token"]
            ),
        },
    )

    assert logout.status_code == 204

    refresh = client.post(
        "/api/v1/auth/refresh",
        json={
            "refresh_token": (
                tokens["refresh_token"]
            ),
        },
    )

    assert refresh.status_code == 401


def test_logout_all_invalidates_access_token(
    client: TestClient,
) -> None:
    tokens = register_student(client)

    headers = {
        "Authorization": (
            f"Bearer {tokens['access_token']}"
        )
    }

    logout_all = client.post(
        "/api/v1/auth/logout-all",
        headers=headers,
    )

    assert logout_all.status_code == 204

    profile = client.get(
        "/api/v1/auth/me",
        headers=headers,
    )

    assert profile.status_code == 401