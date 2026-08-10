from app.core.config import Settings
from app.recognition.base import (
    SignRecognizer,
)
from app.recognition.http_recognizer import (
    HttpSignRecognizer,
)
from app.recognition.mock_recognizer import (
    MockSignRecognizer,
)


def build_recognizer(
    settings: Settings,
) -> SignRecognizer:
    if settings.recognizer_mode == "mock":
        return MockSignRecognizer()

    if not settings.recognizer_base_url:
        raise RuntimeError(
            "RECOGNIZER_BASE_URL is required "
            "when RECOGNIZER_MODE=http."
        )

    return HttpSignRecognizer(
        base_url=(
            settings.recognizer_base_url
        ),
        api_key=(
            settings.recognizer_api_key
        ),
        timeout_seconds=(
            settings
            .recognizer_timeout_seconds
        ),
    )