from fastapi.testclient import TestClient


def test_root_endpoint(
    client: TestClient,
) -> None:
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["message"] == (
        "SLSL-BAMS API is running"
    )
    assert body["version"] == "1.0.0"
    assert body["database"] == "MongoDB"


def test_health_endpoint(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"
    assert body["database"] == "connected"
