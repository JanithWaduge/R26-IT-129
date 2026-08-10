from uuid import uuid4

from fastapi.testclient import TestClient


def register(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Review Student",
            "email": "review@example.com",
            "password": "ReviewPass123",
            "preferred_language": "english",
            "grade_level": "Grade 8",
        },
    )
    assert response.status_code == 201
    return {
        "Authorization": (
            "Bearer "
            + response.json()["access_token"]
        )
    }


def test_new_directions_are_due(client: TestClient) -> None:
    response = client.get(
        "/api/v1/reviews/overview",
        headers=register(client),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_signs"] == 5
    assert body["receptive_due"] == 5
    assert body["productive_due"] == 5
    assert body["total_due_directions"] == 10


def test_due_only_quiz_uses_scheduler(client: TestClient) -> None:
    response = client.post(
        "/api/v1/quizzes/sessions",
        headers=register(client),
        json={
            "client_request_id": str(uuid4()),
            "mode": "receptive",
            "selection_strategy": "due_only",
            "prompt_language": "english",
            "question_count": 3,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["selection_strategy"] == "due_only"
    assert body["current_question"]["was_due"] is True
    assert body["current_question"]["selection_priority"] > 0
