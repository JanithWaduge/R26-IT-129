from uuid import uuid4

from fastapi.testclient import TestClient


def register(client: TestClient, email: str) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Gamification Student",
            "email": email,
            "password": "GamifyPass123",
            "preferred_language": "english",
            "grade_level": "Grade 8",
        },
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def create_receptive_quiz(
    client: TestClient,
    headers: dict[str, str],
) -> dict:
    response = client.post(
        "/api/v1/quizzes/sessions",
        headers=headers,
        json={
            "client_request_id": str(uuid4()),
            "mode": "receptive",
            "selection_strategy": "adaptive",
            "prompt_language": "english",
            "question_count": 1,
        },
    )
    assert response.status_code == 201
    return response.json()


def complete_quiz_correctly(
    client: TestClient,
    headers: dict[str, str],
    session: dict,
) -> dict:
    question = session["current_question"]
    for option in question["options"]:
        response = client.post(
            f"/api/v1/quizzes/sessions/{session['id']}/answer",
            headers=headers,
            json={
                "submission_id": str(uuid4()),
                "question_id": question["question_id"],
                "direction": "receptive",
                "selected_sign_id": option["sign_id"],
                "client_response_time_ms": 2000,
            },
        )
        assert response.status_code == 200
        body = response.json()
        if body["feedback"]["is_correct"]:
            return body
    raise AssertionError("No correct option was accepted.")


def test_profile_starts_empty(client: TestClient) -> None:
    headers = register(client, "gamification.empty@example.com")
    response = client.get("/api/v1/gamification/profile", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total_xp"] == 0
    assert body["level"] == 1
    assert body["total_quizzes"] == 0
    assert body["current_streak_days"] == 0


def test_completed_quiz_awards_xp(client: TestClient) -> None:
    headers = register(client, "gamification.reward@example.com")
    completed = complete_quiz_correctly(
        client,
        headers,
        create_receptive_quiz(client, headers),
    )
    reward = completed["reward"]
    assert reward is not None
    assert reward["xp_awarded"] > 0
    assert reward["current_streak_days"] == 1
    profile = client.get(
        "/api/v1/gamification/profile", headers=headers
    ).json()
    assert profile["total_quizzes"] == 1
    assert profile["total_correct"] == 1
    assert profile["total_xp"] == reward["total_xp"]


def test_first_quiz_unlocks_badge(client: TestClient) -> None:
    headers = register(client, "gamification.badge@example.com")
    completed = complete_quiz_correctly(
        client,
        headers,
        create_receptive_quiz(client, headers),
    )
    codes = {badge["code"] for badge in completed["reward"]["badges_awarded"]}
    assert "first_quiz" in codes


def test_session_reward_is_idempotent(client: TestClient) -> None:
    headers = register(client, "gamification.idempotent@example.com")
    completed = complete_quiz_correctly(
        client,
        headers,
        create_receptive_quiz(client, headers),
    )
    session_id = completed["session"]["id"]
    first = client.get("/api/v1/gamification/profile", headers=headers).json()
    for _ in range(3):
        response = client.get(
            f"/api/v1/quizzes/sessions/{session_id}", headers=headers
        )
        assert response.status_code == 200
    second = client.get("/api/v1/gamification/profile", headers=headers).json()
    assert first["total_xp"] == second["total_xp"]
    assert second["total_quizzes"] == 1


def test_reward_history_is_private(client: TestClient) -> None:
    first_headers = register(client, "gamification.first@example.com")
    complete_quiz_correctly(
        client,
        first_headers,
        create_receptive_quiz(client, first_headers),
    )
    second_headers = register(client, "gamification.second@example.com")
    history = client.get(
        "/api/v1/gamification/history", headers=second_headers
    )
    assert history.status_code == 200
    assert history.json()["items"] == []
