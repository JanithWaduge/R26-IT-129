from fastapi.testclient import TestClient


def create_authenticated_student(
    client: TestClient,
) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Profile Student",
            "email": "profile@example.com",
            "password": "ProfilePass123",
            "preferred_language": "english",
            "grade_level": "Grade 7",
        },
    )

    assert response.status_code == 201

    return response.json()


def test_student_can_get_own_profile(
    client: TestClient,
) -> None:
    tokens = create_authenticated_student(
        client
    )

    response = client.get(
        "/api/v1/students/me",
        headers={
            "Authorization": (
                f"Bearer {tokens['access_token']}"
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["full_name"] == (
        "Profile Student"
    )


def test_student_can_update_own_profile(
    client: TestClient,
) -> None:
    tokens = create_authenticated_student(
        client
    )

    response = client.patch(
        "/api/v1/students/me",
        headers={
            "Authorization": (
                f"Bearer {tokens['access_token']}"
            )
        },
        json={
            "full_name": (
                "Updated Profile Student"
            ),
            "preferred_language": "tamil",
            "grade_level": "Grade 8",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["full_name"] == (
        "Updated Profile Student"
    )
    assert body["preferred_language"] == (
        "tamil"
    )
    assert body["grade_level"] == "Grade 8"


def test_profile_update_cannot_change_role(
    client: TestClient,
) -> None:
    tokens = create_authenticated_student(
        client
    )

    response = client.patch(
        "/api/v1/students/me",
        headers={
            "Authorization": (
                f"Bearer {tokens['access_token']}"
            )
        },
        json={
            "role": "admin",
        },
    )

    assert response.status_code == 422