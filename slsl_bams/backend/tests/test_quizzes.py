from uuid import uuid4

from fastapi.testclient import TestClient


def register_student(
    client: TestClient,
    *,
    email: str = "quiz.student@example.com",
) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Quiz Student",
            "email": email,
            "password": "QuizStudent123",
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


def create_receptive_session(
    client: TestClient,
    headers: dict[str, str],
    *,
    client_request_id: str | None = None,
    question_count: int = 3,
) -> dict:
    response = client.post(
        "/api/v1/quizzes/sessions",
        headers=headers,
        json={
            "client_request_id": (
                client_request_id
                or str(uuid4())
            ),
            "mode": "receptive",
            "prompt_language": "english",
            "question_count": question_count,
        },
    )

    assert response.status_code == 201

    return response.json()


def test_quiz_requires_authentication(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/quizzes/sessions",
        json={
            "client_request_id": str(
                uuid4()
            ),
            "mode": "receptive",
            "question_count": 3,
        },
    )

    assert response.status_code == 401


def test_create_receptive_quiz_session(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_receptive_session(
        client,
        headers,
    )

    assert session["status"] == (
        "in_progress"
    )

    assert session["mode"] == (
        "receptive"
    )

    assert session["question_count"] == 3
    assert session["answered_count"] == 0

    question = session[
        "current_question"
    ]

    assert question is not None
    assert question["direction"] == (
        "receptive"
    )

    assert len(question["options"]) == 4

    assert "expected_code" not in question
    assert "sign_id" not in question


def test_session_creation_is_idempotent(
    client: TestClient,
) -> None:
    headers = register_student(client)

    request_id = str(uuid4())

    first = create_receptive_session(
        client,
        headers,
        client_request_id=request_id,
    )

    second = create_receptive_session(
        client,
        headers,
        client_request_id=request_id,
    )

    assert first["id"] == second["id"]


def test_correct_receptive_answer(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_receptive_session(
        client,
        headers,
        question_count=1,
    )

    question = session[
        "current_question"
    ]

    # The expected answer is not exposed through
    # the public response. Test data can infer it by
    # trying the available options until correct.
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
                "client_response_time_ms": 1500,
            },
        )

        assert response.status_code == 200

        body = response.json()

        if body["feedback"]["is_correct"]:
            assert body["session"][
                "status"
            ] == "completed"

            assert body["session"][
                "summary"
            ]["correct_count"] == 1

            return

        if not body["feedback"][
            "retry_allowed"
        ]:
            break

    raise AssertionError(
        "No correct option was accepted."
    )


def test_hint_is_recorded(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_receptive_session(
        client,
        headers,
    )

    question = session[
        "current_question"
    ]

    action_id = str(uuid4())

    response = client.post(
        (
            "/api/v1/quizzes/sessions/"
            f"{session['id']}/hint"
        ),
        headers=headers,
        json={
            "action_id": action_id,
            "question_id": question[
                "question_id"
            ],
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["hint_type"] == "category"
    assert body["hint_count"] == 1
    assert body["session"]["summary"][
        "hints_used"
    ] == 1

    duplicate = client.post(
        (
            "/api/v1/quizzes/sessions/"
            f"{session['id']}/hint"
        ),
        headers=headers,
        json={
            "action_id": action_id,
            "question_id": question[
                "question_id"
            ],
        },
    )

    assert duplicate.status_code == 200
    assert duplicate.json()[
        "hint_count"
    ] == 1


def test_skip_advances_question(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_receptive_session(
        client,
        headers,
        question_count=2,
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

    body = response.json()

    assert body["session"][
        "current_question_index"
    ] == 1

    assert body["session"]["summary"][
        "skipped_count"
    ] == 1


def test_wrong_answer_allows_retry(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_receptive_session(
        client,
        headers,
        question_count=1,
    )

    question = session[
        "current_question"
    ]

    first_option = question["options"][0]

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
            "selected_sign_id": (
                first_option["sign_id"]
            ),
            "client_response_time_ms": 1200,
        },
    )

    assert response.status_code == 200

    body = response.json()

    if not body["feedback"]["is_correct"]:
        assert body["feedback"][
            "retry_allowed"
        ] is True

        assert body["session"][
            "current_question"
        ]["attempt_number"] == 2


def test_productive_recognition_contract(
    client: TestClient,
) -> None:
    headers = register_student(client)

    create_response = client.post(
        "/api/v1/quizzes/sessions",
        headers=headers,
        json={
            "client_request_id": str(
                uuid4()
            ),
            "mode": "productive",
            "prompt_language": "english",
            "question_count": 1,
        },
    )

    assert create_response.status_code == 201

    session = create_response.json()

    question = session[
        "current_question"
    ]

    assert question["direction"] == (
        "productive"
    )

    assert question["prompt_text"]
    assert question["options"] == []

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
            "client_response_time_ms": 3500,
            "recognition_result": {
                "request_id": str(
                    uuid4()
                ),
                "model_name": (
                    "objective-2-test-model"
                ),
                "model_version": "0.1.0",
                "predicted_sign_id": None,
                "predicted_code": None,
                "confidence": 0.12,
                "inference_time_ms": 45,
                "top_predictions": [],
            },
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["feedback"][
        "is_correct"
    ] is False

    assert body["feedback"][
        "recognition_confidence"
    ] == 0.12


def test_student_cannot_access_another_session(
    client: TestClient,
) -> None:
    first_headers = register_student(
        client,
        email="first.quiz@example.com",
    )

    session = create_receptive_session(
        client,
        first_headers,
    )

    second_headers = register_student(
        client,
        email="second.quiz@example.com",
    )

    response = client.get(
        (
            "/api/v1/quizzes/sessions/"
            f"{session['id']}"
        ),
        headers=second_headers,
    )

    assert response.status_code == 404


def test_abandon_session(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_receptive_session(
        client,
        headers,
    )

    response = client.post(
        (
            "/api/v1/quizzes/sessions/"
            f"{session['id']}/abandon"
        ),
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["status"] == (
        "abandoned"
    )