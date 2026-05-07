"""
sinhala_asr.py
Pretrained Sinhala Whisper from HuggingFace — no fine-tuning needed.
Model: Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2
"""

import os
import io
import torch
import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline

# Ensure ffmpeg is on PATH for M4A/MP3 decoding (phone audio formats)
_ffmpeg_dir = r'C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Links'
if _ffmpeg_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')


# ── Model loader ──────────────────────────────────────────────────────────────
def load_sinhala_model():
    """
    Load pretrained Sinhala Whisper from HuggingFace.
    Uses GPU if available, falls back to CPU.
    Called ONCE at server startup — result is cached in main.py.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ASR] Loading model on: {device.upper()}")

    asr = pipeline(
        "automatic-speech-recognition",
        model="Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2",
        device=device,
    )
    print("[ASR] Model ready.")
    return asr


def _load_audio(audio_path: str) -> np.ndarray:
    """
    Load audio file to 16kHz numpy array.
    Tries librosa first (handles FLAC/WAV natively),
    falls back to soundfile, then ffmpeg subprocess for M4A/MP3.
    """
    ext = os.path.splitext(audio_path)[1].lower()

    try:
        # librosa handles FLAC, WAV, OGG natively via soundfile
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        return audio
    except Exception:
        pass

    # Fallback: use ffmpeg to decode to raw PCM, then load
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1",
            "-f", "wav", "pipe:1"
        ]
        proc = subprocess.run(cmd, capture_output=True)
        audio, _ = sf.read(io.BytesIO(proc.stdout))
        audio = audio.astype(np.float32)
        return audio
    except Exception as e:
        raise RuntimeError(
            f"Could not decode audio '{audio_path}': {e}\n"
            "Install ffmpeg: winget install ffmpeg"
        )


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe_sinhala(
    audio_path: str,
    language: str = "si",
    asr=None,           # pass cached pipeline from main.py to avoid reload
) -> dict:
    """
    Transcribe Sinhala audio.

    Args:
        audio_path : Path to audio file (FLAC, WAV, M4A, MP3, etc.)
        language   : "si" | "ta" | "auto"  (informational — model is Sinhala-only)
        asr        : Pre-loaded pipeline (optional). Loads fresh if None.

    Returns:
        {"text": "...", "language": "si"}
    """
    if asr is None:
        asr = load_sinhala_model()

    audio_array = _load_audio(audio_path)

    # NOTE: Ransaka model is Sinhala-only — no generate_kwargs needed.
    # Passing language= breaks models with outdated generation configs.
    result = asr({"array": audio_array, "sampling_rate": 16000})

    # Return all fields expected by AsrResult.fromJson in asr_service.dart
    _language_names = {"si": "Sinhala", "ta": "Tamil"}
    _locale_map     = {"si": "si-LK",   "ta": "ta-LK"}

    return {
        "text":          result["text"].strip(),
        "language":      language,
        "language_name": _language_names.get(language, "Sinhala"),
        "locale":        _locale_map.get(language, "si-LK"),
        "confidence":    0.85,
    }


# ── Quick test (run directly: python sinhala_asr.py) ─────────────────────────
if __name__ == "__main__":
    import pandas as pd
    import sys, io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=== Testing Sinhala ASR ===")
    df       = pd.read_csv("./sinhala_data/test.csv")
    sample   = df["audio_path"].iloc[0]
    expected = df["transcription"].iloc[0]

    print(f"Audio file : {sample}")
    print(f"Expected   : {expected}")
    print("Transcribing...")

    result = transcribe_sinhala(sample)
    print(f"Result     : {result['text']}")
