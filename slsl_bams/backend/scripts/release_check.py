from pathlib import Path

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    errors: list[str] = []
    checks = (
        (settings.environment != "production", "ENVIRONMENT is not production."),
        (settings.debug, "Debug mode is enabled."),
        (settings.docs_enabled, "API documentation is enabled."),
        (not settings.force_https, "HTTPS enforcement is disabled."),
        (settings.run_migrations_on_startup, "Startup migrations are enabled."),
        (settings.ml_allow_synthetic_model, "Synthetic ML model is allowed."),
        (settings.research_report_include_raw_labels, "Raw research labels are enabled."),
        ("*" in settings.cors_allowed_origins, "Wildcard CORS is configured."),
    )
    errors.extend(message for failed, message in checks if failed)
    for path in ("requirements.txt", "app/main.py", "scripts/run_migrations.py"):
        if not Path(path).exists():
            errors.append(f"Required file missing: {path}")
    if errors:
        print("\nRELEASE CHECK FAILED")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"\nRELEASE CHECK PASSED\nVersion     : {settings.app_version}"
          f"\nEnvironment : {settings.environment}")


if __name__ == "__main__":
    main()
