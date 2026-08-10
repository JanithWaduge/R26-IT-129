from fastapi.testclient import TestClient


def register_and_get_headers(
    client: TestClient,
) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "full_name": "Vocabulary Student",
            "email": "vocabulary@example.com",
            "password": "VocabularyPass123",
            "preferred_language": "sinhala",
            "grade_level": "Grade 8",
        },
    )

    assert response.status_code == 201

    access_token = response.json()[
        "access_token"
    ]

    return {
        "Authorization": (
            f"Bearer {access_token}"
        )
    }


def test_curriculum_requires_authentication(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/curriculum/categories"
    )

    assert response.status_code == 401


def test_signs_require_authentication(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/signs"
    )

    assert response.status_code == 401


def test_list_curriculum_categories(
    client: TestClient,
) -> None:
    headers = register_and_get_headers(
        client
    )

    response = client.get(
        "/api/v1/curriculum/categories",
        headers=headers,
    )

    assert response.status_code == 200

    categories = response.json()

    assert len(categories) == 5

    codes = {
        category["code"]
        for category in categories
    }

    assert "GREETINGS" in codes
    assert "SCHOOL_VOCABULARY" in codes
    assert "NUMBERS" in codes
    assert "EMOTIONS" in codes
    assert "DAILY_LIFE" in codes

    assert categories[0][
        "validation_status"
    ] == "provisional"


def test_list_competencies_for_category(
    client: TestClient,
) -> None:
    headers = register_and_get_headers(
        client
    )

    categories_response = client.get(
        "/api/v1/curriculum/categories",
        headers=headers,
    )

    category = (
        categories_response.json()[0]
    )

    response = client.get(
        "/api/v1/curriculum/competencies",
        headers=headers,
        params={
            "category_id": category["id"],
        },
    )

    assert response.status_code == 200

    competencies = response.json()

    assert len(competencies) == 1

    assert competencies[0][
        "category_id"
    ] == category["id"]

    assert competencies[0][
        "receptive_mastery_threshold"
    ] == 0.85

    assert competencies[0][
        "productive_mastery_threshold"
    ] == 0.85


def test_list_development_signs_in_testing(
    client: TestClient,
) -> None:
    headers = register_and_get_headers(
        client
    )

    response = client.get(
        "/api/v1/signs",
        headers=headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_items"] == 5
    assert body["page"] == 1
    assert body["total_pages"] == 1

    assert all(
        item["content_status"]
        == "development"
        for item in body["items"]
    )


def test_filter_signs_by_category(
    client: TestClient,
) -> None:
    headers = register_and_get_headers(
        client
    )

    categories = client.get(
        "/api/v1/curriculum/categories",
        headers=headers,
    ).json()

    greetings_category = next(
        category
        for category in categories
        if category["code"] == "GREETINGS"
    )

    response = client.get(
        "/api/v1/signs",
        headers=headers,
        params={
            "category_id": (
                greetings_category["id"]
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_items"] == 1
    assert body["items"][0][
        "gloss"
    ] == "HELLO"


def test_search_signs(
    client: TestClient,
) -> None:
    headers = register_and_get_headers(
        client
    )

    response = client.get(
        "/api/v1/signs",
        headers=headers,
        params={
            "search": "school",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_items"] == 1
    assert body["items"][0][
        "gloss"
    ] == "SCHOOL"


def test_get_sign_details(
    client: TestClient,
) -> None:
    headers = register_and_get_headers(
        client
    )

    list_response = client.get(
        "/api/v1/signs",
        headers=headers,
    )

    sign_id = list_response.json()[
        "items"
    ][0]["id"]

    response = client.get(
        f"/api/v1/signs/{sign_id}",
        headers=headers,
    )

    assert response.status_code == 200

    sign = response.json()

    assert sign["category"]
    assert len(sign["competencies"]) == 1
    media_source_type = sign["media"]["source_type"]
    assert media_source_type in {
        "pending",
        "video",
    }
    if media_source_type == "pending":
        assert sign["media"]["uri"] is None
    else:
        assert sign["media"]["uri"]
    assert sign[
        "validation_status"
    ] == "provisional"


def test_invalid_category_filter_is_rejected(
    client: TestClient,
) -> None:
    headers = register_and_get_headers(
        client
    )

    response = client.get(
        "/api/v1/signs",
        headers=headers,
        params={
            "category_id": (
                "invalid-object-id"
            ),
        },
    )

    assert response.status_code == 422
