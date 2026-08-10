from collections.abc import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

from pymongo import MongoClient

from app.db.collections import (
    CURRICULUM_CATEGORIES_COLLECTION,
    CURRICULUM_COMPETENCIES_COLLECTION,
    SIGNS_COLLECTION,
)
from app.db.seed_data import build_seed_documents

@pytest.fixture
def client() -> Generator[
    TestClient,
    None,
    None,
]:
    test_database_name = (
        f"slsl_bams_test_{uuid4().hex}"
    )

    settings = Settings(
        app_name="SLSL-BAMS Test API",
        app_version="1.0.0",
        environment="test",
        trusted_hosts=["testserver", "localhost", "127.0.0.1"],
        api_v1_prefix="/api/v1",
        mongo_uri="mongodb://127.0.0.1:27017",
        mongo_database=test_database_name,
        mongo_server_selection_timeout_ms=5000,
        drop_database_on_shutdown=True,
        jwt_access_secret="a" * 128,
        jwt_refresh_secret="b" * 128,
        jwt_algorithm="HS256",
        jwt_issuer="slsl-bams-test-api",
        jwt_audience="slsl-bams-test-client",
        access_token_expire_minutes=15,
        refresh_token_expire_days=30,
        quiz_session_duration_minutes=60,
        quiz_max_retries=3,
        quiz_option_count=4,
        productive_acceptance_confidence=0.70,
        gamification_xp_per_level=250,
        gamification_timezone="Asia/Colombo",
        ml_scheduler_enabled=False,
        ml_allow_synthetic_model=False,
        experiment_enabled=False,
        recognizer_mode="mock",
        recognizer_base_url=None,
        recognizer_api_key=None,
        recognizer_timeout_seconds=20.0,
        recognition_max_upload_bytes=(
            15 * 1024 * 1024
        ),
        recognition_min_duration_ms=1000,
        recognition_max_duration_ms=10000,
        recognition_temp_directory=(
            "storage/test_recognition_temp"
        ),
        recognition_retain_uploads=False,
        # Retain compatibility with tests that
        # directly submit mocked recognition JSON.
        allow_client_recognition_results=True,
        mastery_algorithm_version="bams-mastery-v1",
        mastery_initial_learning_rate=0.45,
        mastery_intermediate_learning_rate=0.30,
        mastery_stable_learning_rate=0.20,
        mastery_failure_learning_rate=0.45,
        sm2_algorithm_version="bams-sm2-v1",
        sm2_initial_ease_factor=2.50,
        sm2_min_ease_factor=1.30,
        sm2_max_ease_factor=3.20,
        sm2_max_interval_days=365,
    )

    app = create_app(settings)

    with TestClient(app) as test_client:
        sync_client = MongoClient(
            settings.mongo_uri
        )

        try:
            database = sync_client[
                settings.mongo_database
            ]

            seed_data = build_seed_documents()

            database[
                CURRICULUM_CATEGORIES_COLLECTION
            ].insert_many(
                seed_data["categories"]
            )

            database[
                CURRICULUM_COMPETENCIES_COLLECTION
            ].insert_many(
                seed_data["competencies"]
            )

            database[
                SIGNS_COLLECTION
            ].insert_many(
                seed_data["signs"]
            )

        finally:
            sync_client.close()

        yield test_client
