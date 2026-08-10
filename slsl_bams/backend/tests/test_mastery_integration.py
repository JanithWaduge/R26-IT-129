from uuid import uuid4

from fastapi.testclient import TestClient


def register_student(
    client: TestClient,
    *,
    email: str = "mastery@example.com",
) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Mastery Student",
            "email": email,
            "password": "MasteryPass123",
            "preferred_language": "english",
            "grade_level": "Grade 8",
        },
    )

    assert response.status_code == 201

    return {
        "Authorization": (
            "Bearer "
            + response.json()[
                "access_token"
            ]
        )
    }


def create_session(
    client: TestClient,
    headers: dict[str, str],
    *,
    mode: str,
) -> dict:
    response = client.post(
        "/api/v1/quizzes/sessions",
        headers=headers,
        json={
            "client_request_id": str(
                uuid4()
            ),
            "mode": mode,
            "prompt_language": "english",
            "question_count": 1,
        },
    )

    assert response.status_code == 201

    return response.json()


def answer_receptive_until_correct(
    client: TestClient,
    headers: dict[str, str],
    session: dict,
) -> dict:
    question = session[
        "current_question"
    ]

    for option in question["options"]:
        response = client.post(
            (
                "/api/v1/quizzes/sessions/"
                f"{session['id']}/answer"
            ),
            headers=headers,
            json={
                "submission_id": str(
                    uuid4()
                ),
                "question_id": question[
                    "question_id"
                ],
                "direction": "receptive",
                "selected_sign_id": option[
                    "sign_id"
                ],
                "client_response_time_ms": 2500,
            },
        )

        assert response.status_code == 200

        body = response.json()

        if body["feedback"][
            "is_correct"
        ]:
            return body

    raise AssertionError(
        "Correct option was not found."
    )


def test_mastery_overview_starts_with_new_signs(
    client: TestClient,
) -> None:
    headers = register_student(client)

    response = client.get(
        "/api/v1/mastery/overview",
        headers=headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_signs"] == 5
    assert body["attempted_signs"] == 0
    assert body["statuses"]["new"] == 5


def test_receptive_quiz_updates_receptive_mastery(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_session(
        client,
        headers,
        mode="receptive",
    )

    answer_receptive_until_correct(
        client,
        headers,
        session,
    )

    response = client.get(
        "/api/v1/mastery/signs",
        headers=headers,
    )

    assert response.status_code == 200

    attempted = [
        item
        for item in response.json()[
            "items"
        ]
        if item["attempted"]
    ]

    assert len(attempted) == 1

    mastery = attempted[0]

    assert mastery["receptive"][
        "total_reviews"
    ] == 1

    assert mastery["receptive"][
        "score"
    ] > 0

    assert mastery["productive"][
        "total_reviews"
    ] == 0

    assert mastery["balance_status"] == (
        "productive_unassessed"
    )


def test_skipped_question_creates_zero_quality_mastery(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_session(
        client,
        headers,
        mode="receptive",
    )

    question = session[
        "current_question"
    ]

    response = client.post(
        (
            "/api/v1/quizzes/sessions/"
            f"{session['id']}/skip"
        ),
        headers=headers,
        json={
            "action_id": str(uuid4()),
            "question_id": question[
                "question_id"
            ],
        },
    )

    assert response.status_code == 200

    mastery_response = client.get(
        "/api/v1/mastery/signs",
        headers=headers,
    )

    attempted = [
        item
        for item in (
            mastery_response.json()[
                "items"
            ]
        )
        if item["attempted"]
    ]

    assert len(attempted) == 1

    direction = attempted[0][
        "receptive"
    ]

    assert direction[
        "total_reviews"
    ] == 1

    assert direction["score"] == 0.0
    assert direction[
        "last_quality_score"
    ] == 0.0


def test_repeated_session_reads_do_not_duplicate_mastery(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_session(
        client,
        headers,
        mode="receptive",
    )

    completed = (
        answer_receptive_until_correct(
            client,
            headers,
            session,
        )
    )

    session_id = completed[
        "session"
    ]["id"]

    for _ in range(3):
        response = client.get(
            (
                "/api/v1/quizzes/sessions/"
                f"{session_id}"
            ),
            headers=headers,
        )

        assert response.status_code == 200

    mastery_response = client.get(
        "/api/v1/mastery/signs",
        headers=headers,
    )

    attempted = [
        item
        for item in (
            mastery_response.json()[
                "items"
            ]
        )
        if item["attempted"]
    ]

    assert attempted[0]["receptive"][
        "total_reviews"
    ] == 1


def test_productive_failure_updates_productive_state(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_session(
        client,
        headers,
        mode="productive",
    )

    question = session[
        "current_question"
    ]

    final_body = None

    for _ in range(6):
        response = client.post(
            (
                "/api/v1/quizzes/sessions/"
                f"{session['id']}/answer"
            ),
            headers=headers,
            json={
                "submission_id": str(
                    uuid4()
                ),
                "question_id": question[
                    "question_id"
                ],
                "direction": "productive",
                "selected_sign_id": None,
                "client_response_time_ms": 4000,
                "recognition_result": {
                    "request_id": str(
                        uuid4()
                    ),
                    "model_name": (
                        "test-recognizer"
                    ),
                    "model_version": "0.1",
                    "predicted_sign_id": None,
                    "predicted_code": None,
                    "confidence": 0.15,
                    "inference_time_ms": 50,
                    "top_predictions": [],
                },
            },
        )

        assert response.status_code == 200

        final_body = response.json()

        if final_body["feedback"][
            "question_finalized"
        ]:
            break

    assert final_body is not None
    assert final_body["session"][
        "status"
    ] == "completed"

    mastery_response = client.get(
        "/api/v1/mastery/signs",
        headers=headers,
    )

    attempted = [
        item
        for item in (
            mastery_response.json()[
                "items"
            ]
        )
        if item["attempted"]
    ]

    productive = attempted[0][
        "productive"
    ]

    assert productive[
        "total_reviews"
    ] == 1

    assert productive[
        "failure_count"
    ] == 1

    assert productive[
        "last_recognition_confidence"
    ] == 0.15


def test_mastery_is_private_to_student(
    client: TestClient,
) -> None:
    first_headers = register_student(
        client,
        email="mastery.first@example.com",
    )

    session = create_session(
        client,
        first_headers,
        mode="receptive",
    )

    answer_receptive_until_correct(
        client,
        first_headers,
        session,
    )

    second_headers = register_student(
        client,
        email="mastery.second@example.com",
    )

    second_overview = client.get(
        "/api/v1/mastery/overview",
        headers=second_headers,
    )

    assert second_overview.status_code == 200

    body = second_overview.json()

    assert body["attempted_signs"] == 0
    assert body["statuses"]["new"] == 5