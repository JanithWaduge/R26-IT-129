"""
Stage 1 — Whisper ASR Module
Loads OpenAI Whisper once and transcribes audio files in Sinhala or Tamil.
"""

import whisper

# Maps Whisper language codes to display names and locale tags
_LANGUAGE_NAMES = {
    "si": "Sinhala",
    "ta": "Tamil",
}
_LOCALE_MAP = {
    "si": "si-LK",
    "ta": "ta-LK",
}


class WhisperASR:
    """
    Singleton wrapper around OpenAI Whisper.
    Load once at server startup, then call transcribe() per request.
    """

    def __init__(self, model_size: str = "small"):
        """
        Load Whisper model into memory.

        Args:
            model_size: "tiny" | "base" | "small" | "medium" | "large"
                        "small" is the minimum recommended for Sinhala/Tamil —
                        both are low-resource languages that tiny/base handle poorly.
        """
        self.model_size = model_size
        print(f"[Whisper] Loading '{model_size}' model — this takes ~10 s the first time...")
        self.model = whisper.load_model(model_size)
        print("[Whisper] Model ready.")

    def transcribe(self, audio_path: str, language: str | None = None) -> dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Absolute path to audio file (WAV, M4A, MP3, etc.)
            language:   ISO 639-1 code to force language ("si", "ta"),
                        or None to let Whisper auto-detect.

        Returns:
            {
                "text":          "transcribed text",
                "language":      "si",
                "language_name": "Sinhala",
                "locale":        "si-LK",
                "confidence":    0.82        # 0.0 – 1.0, derived from segment log-probs
            }
        """
        decode_options: dict = {}
        if language:
            decode_options["language"] = language

        result = self.model.transcribe(audio_path, **decode_options)

        detected = result.get("language", "unknown")

        # Convert mean segment log-probability to a human-readable 0–1 confidence.
        # log-prob of 0.0 → 100 %, –1.0 → ~37 %; clamp below 0 and above 1.
        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
            confidence = round(max(0.0, min(1.0, avg_logprob + 1.0)), 2)
        else:
            confidence = 0.0

        return {
            "text":          result["text"].strip(),
            "language":      detected,
            "language_name": _LANGUAGE_NAMES.get(detected, detected.upper()),
            "locale":        _LOCALE_MAP.get(detected, detected),
            "confidence":    confidence,
        }
