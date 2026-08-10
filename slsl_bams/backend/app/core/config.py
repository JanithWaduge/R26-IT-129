import os
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

ENV_FILE = os.getenv("ENV_FILE", ".env.development")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_name: str = "SLSL-BAMS API"
    app_version: str = "1.0.0"
    environment: Literal["development", "test", "production"] = "development"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    docs_enabled: bool = True
    run_migrations_on_startup: bool = True
    force_https: bool = False
    trusted_hosts: list[str] = ["localhost", "127.0.0.1"]
    cors_allowed_origins: list[str] = ["http://localhost"]
    audit_retention_days: int = Field(default=365, ge=30, le=3650)
    research_retention_days: int = Field(default=730, ge=30, le=3650)

    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_database: str = "slsl_bams_dev"

    mongo_server_selection_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
    )

    drop_database_on_shutdown: bool = False

    jwt_access_secret: str = Field(min_length=64)
    jwt_refresh_secret: str = Field(min_length=64)

    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "slsl-bams-api"
    jwt_audience: str = "slsl-bams-mobile"

    access_token_expire_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
    )

    refresh_token_expire_days: int = Field(
        default=30,
        ge=1,
        le=90,
    )

    quiz_session_duration_minutes: int = Field(
        default=60,
        ge=5,
        le=240,
    )

    quiz_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
    )

    quiz_option_count: int = Field(
        default=4,
        ge=2,
        le=6,
    )

    productive_acceptance_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )

    mastery_algorithm_version: str = "bams-mastery-v1"

    mastery_initial_learning_rate: float = Field(
        default=0.45,
        ge=0.05,
        le=1.0,
    )

    mastery_intermediate_learning_rate: float = Field(
        default=0.30,
        ge=0.05,
        le=1.0,
    )

    mastery_stable_learning_rate: float = Field(
        default=0.20,
        ge=0.05,
        le=1.0,
    )

    mastery_failure_learning_rate: float = Field(
        default=0.45,
        ge=0.05,
        le=1.0,
    )

    sm2_algorithm_version: str = "bams-sm2-v1"

    sm2_initial_ease_factor: float = Field(
        default=2.50,
        ge=1.30,
        le=3.50,
    )

    sm2_min_ease_factor: float = Field(
        default=1.30,
        ge=1.10,
        le=2.50,
    )

    sm2_max_ease_factor: float = Field(
        default=3.20,
        ge=2.00,
        le=4.00,
    )

    sm2_max_interval_days: int = Field(
        default=365,
        ge=30,
        le=1095,
    )

    recognizer_mode: Literal[
        "mock",
        "http",
    ] = "mock"

    recognizer_base_url: str | None = None
    recognizer_api_key: str | None = None

    recognizer_timeout_seconds: float = Field(
        default=20.0,
        ge=1.0,
        le=120.0,
    )

    recognition_max_upload_bytes: int = Field(
        default=15 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
    )

    recognition_min_duration_ms: int = Field(
        default=1000,
        ge=250,
        le=10000,
    )

    recognition_max_duration_ms: int = Field(
        default=10000,
        ge=1000,
        le=60000,
    )

    recognition_temp_directory: str = (
        "storage/recognition_temp"
    )

    recognition_retain_uploads: bool = False

    allow_client_recognition_results: bool = False

    sign_video_directory: str = "storage/sign_videos"

    sign_video_url_prefix: str = (
    "/media/sign-videos"
    )

    gamification_xp_per_level: int = Field(
    default=250,
    ge=50,
    le=5000,
    )

    gamification_timezone: str = "Asia/Colombo"

    ml_pseudonym_secret: SecretStr | None = None
    ml_dataset_schema_version: str = "slsl-receptive-recall-v1"
    ml_min_recall_delay_hours: int = Field(default=24, ge=1, le=720)
    ml_max_recall_delay_days: int = Field(default=30, ge=2, le=365)
    ml_export_directory: str = "outputs/ml_datasets"
    ml_min_training_rows: int = Field(default=100, ge=10, le=1_000_000)
    ml_min_class_rows: int = Field(default=20, ge=5, le=100_000)

    ml_model_output_directory: str = (
        "outputs/models/receptive_recall"
    )

    ml_model_version: str = (
        "synthetic-prototype-1.0.0"
    )

    ml_recall_threshold: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
    )

    ml_scheduler_enabled: bool = False
    ml_model_path: str = "outputs/models/receptive_recall/recall_model_champion.joblib"
    ml_model_metadata_path: str = "outputs/models/receptive_recall/model_metadata.json"
    ml_allow_synthetic_model: bool = False
    ml_require_exact_sklearn_version: bool = True
    ml_min_interval_factor: float = Field(default=0.50, ge=0.25, le=1.00)
    ml_max_interval_factor: float = Field(default=1.15, ge=1.00, le=1.50)

    experiment_enabled: bool = False
    experiment_name: str = "slsl-scheduler-comparison-v1"
    experiment_assignment_secret: SecretStr | None = None
    experiment_min_recall_delay_hours: int = Field(default=24, ge=1, le=720)
    experiment_max_recall_delay_days: int = Field(default=30, ge=2, le=365)
    experiment_bootstrap_iterations: int = Field(default=2000, ge=100, le=100000)
    experiment_min_learners_per_arm: int = Field(default=15, ge=3, le=10000)
    experiment_output_directory: str = "outputs/evaluations"
    research_report_output_directory: str = "outputs/research_reports"
    research_report_title: str = "SLSL-BAMS Scheduler Evaluation"
    research_report_include_raw_labels: bool = False

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug_mode(cls, value):
        if isinstance(value, str) and value.lower() in {"release", "profile"}:
            return False
        if isinstance(value, str) and value.lower() == "debug":
            return True
        return value

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.environment != "production":
            return self
        problems: list[str] = []
        checks = (
            (self.debug, "DEBUG must be false."),
            (self.docs_enabled, "DOCS_ENABLED must be false."),
            (not self.force_https, "FORCE_HTTPS must be true."),
            (self.run_migrations_on_startup, "RUN_MIGRATIONS_ON_STARTUP must be false."),
            ("*" in self.cors_allowed_origins, "Wildcard CORS is prohibited."),
            (self.ml_allow_synthetic_model, "Synthetic ML models cannot be enabled in production."),
            (self.recognition_retain_uploads, "Raw recognition uploads cannot be retained in production."),
        )
        problems.extend(message for unsafe, message in checks if unsafe)
        for name, secret in (("JWT_ACCESS_SECRET", self.jwt_access_secret),
                             ("JWT_REFRESH_SECRET", self.jwt_refresh_secret)):
            if len(secret) < 32:
                problems.append(f"{name} must contain at least 32 characters.")
        if problems:
            raise ValueError("Unsafe production configuration: " + " ".join(problems))
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()
