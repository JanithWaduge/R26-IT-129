from uuid import uuid4

from fastapi.testclient import TestClient


def register_student(
    client: TestClient,
    *,
    email: str = (
        "recognition.student@example.com"
    ),
) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": (
                "Recognition Student"
            ),
            "email": email,
            "password": (
                "RecognitionPass123"
            ),
            "preferred_language": (
                "english"
            ),
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


def create_productive_session(
    client: TestClient,
    headers: dict[str, str],
) -> dict:
    response = client.post(
        "/api/v1/quizzes/sessions",
        headers=headers,
        json={
            "client_request_id": str(
                uuid4()
            ),
            "mode": "productive",
            "selection_strategy": (
                "adaptive"
            ),
            "prompt_language": "english",
            "question_count": 1,
        },
    )

    assert response.status_code == 201

    session = response.json()

    assert session["current_question"][
        "direction"
    ] == "productive"

    return session


def upload_mock_video(
    client: TestClient,
    headers: dict[str, str],
    *,
    session: dict,
    request_id: str,
    content_type: str = "video/mp4",
) -> object:
    question = session[
        "current_question"
    ]

    return client.post(
        (
            "/api/v1/quizzes/sessions/"
            f"{session['id']}"
            "/recognize-answer"
        ),
        headers=headers,
        data={
            "request_id": request_id,
            "question_id": question[
                "question_id"
            ],
            "client_response_time_ms": (
                "3500"
            ),
            "duration_ms": "3000",
        },
        files={
            "video": (
                "sign.mp4",
                b"development-mock-video",
                content_type,
            )
        },
    )


def test_mock_recognizer_completes_productive_question(
    client: TestClient,
) -> None:
    headers = register_student(client)

    session = create_productive_session(
        client,
        headers,
    )

    response = upload_mock_video(
        client,
        headers,
        session=session,
        request_id=str(uuid4()),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["feedback"][
        "is_correct"
    ] is True

    assert body["session"]["status"] == (
        "completed"
    )

    assert body["recognition"][
        "confidence"
    ] == 0.92

    assert body["recognition"][
        "model_name"
    ] == (
        "mock-recognizer-do-not-evaluate"
    )

    assert body["capture"][
        "retained"
    ] is False


def test_productive_recognition_updates_productive_mastery(
    client: TestClient,
) -> None:
    headers = register_student(
        client,
        email=(
            "recognition.mastery@example.com"
        ),
    )

    session = create_productive_session(
        client,
        headers,
    )

    response = upload_mock_video(
        client,
        headers,
        session=session,
        request_id=str(uuid4()),
    )

    assert response.status_code == 200

    mastery_response = client.get(
        "/api/v1/mastery/signs",
        headers=headers,
    )

    assert mastery_response.status_code == 200

    attempted = [
        item
        for item in mastery_response.json()[
            "items"
        ]
        if item["productive"][
            "total_reviews"
        ] > 0
    ]

    assert len(attempted) == 1

    productive = attempted[0][
        "productive"
    ]

    assert productive[
        "total_reviews"
    ] == 1

    assert productive[
        "successful_reviews"
    ] == 1

    assert productive[
        "last_recognition_confidence"
    ] == 0.92

    assert productive[
        "next_review_at"
    ] is not None


def test_duplicate_request_does_not_duplicate_mastery(
    client: TestClient,
) -> None:
    headers = register_student(
        client,
        email=(
            "recognition.duplicate@example.com"
        ),
    )

    session = create_productive_session(
        client,
        headers,
    )

    request_id = str(uuid4())

    first = upload_mock_video(
        client,
        headers,
        session=session,
        request_id=request_id,
    )

    assert first.status_code == 200

    duplicate = upload_mock_video(
        client,
        headers,
        session=session,
        request_id=request_id,
    )

    assert duplicate.status_code == 200

    assert duplicate.json()[
        "feedback"
    ]["message"] == (
        "This video submission was "
        "already processed."
    )

    mastery_response = client.get(
        "/api/v1/mastery/signs",
        headers=headers,
    )

    attempted = [
        item
        for item in mastery_response.json()[
            "items"
        ]
        if item["productive"][
            "total_reviews"
        ] > 0
    ]

    assert attempted[0][
        "productive"
    ]["total_reviews"] == 1


def test_receptive_question_rejects_video_recognition(
    client: TestClient,
) -> None:
    headers = register_student(
        client,
        email=(
            "recognition.receptive@example.com"
        ),
    )

    response = client.post(
        "/api/v1/quizzes/sessions",
        headers=headers,
        json={
            "client_request_id": str(
                uuid4()
            ),
            "mode": "receptive",
            "selection_strategy": (
                "adaptive"
            ),
            "prompt_language": "english",
            "question_count": 1,
        },
    )

    assert response.status_code == 201

    session = response.json()

    upload_response = upload_mock_video(
        client,
        headers,
        session=session,
        request_id=str(uuid4()),
    )

    assert (
        upload_response.status_code
        == 409
    )


def test_unsupported_video_type_is_rejected(
    client: TestClient,
) -> None:
    headers = register_student(
        client,
        email=(
            "recognition.media@example.com"
        ),
    )

    session = create_productive_session(
        client,
        headers,
    )

    response = upload_mock_video(
        client,
        headers,
        session=session,
        request_id=str(uuid4()),
        content_type="text/plain",
    )

    assert response.status_code == 415


def test_short_recording_is_rejected(
    client: TestClient,
) -> None:
    headers = register_student(
        client,
        email=(
            "recognition.duration@example.com"
        ),
    )

    session = create_productive_session(
        client,
        headers,
    )

    question = session[
        "current_question"
    ]

    response = client.post(
        (
            "/api/v1/quizzes/sessions/"
            f"{session['id']}"
            "/recognize-answer"
        ),
        headers=headers,
        data={
            "request_id": str(
                uuid4()
            ),
            "question_id": question[
                "question_id"
            ],
            "client_response_time_ms": (
                "1000"
            ),
            "duration_ms": "200",
        },
        files={
            "video": (
                "sign.mp4",
                b"mock-video",
                "video/mp4",
            )
        },
    )

    assert response.status_code == 422