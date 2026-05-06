"""
SLSL Objective 1 — FastAPI Server
Runs on port 8000 (separate from Janith's Flask server on port 5000).

Endpoints:
  GET  /health            → liveness check
  POST /asr/transcribe    → audio file → transcribed text + language
"""

import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from asr.whisper_asr import WhisperASR

app = FastAPI(
    title="SLSL Voice-to-Sign API",
    description="Stage 1: Voice → Text (Sinhala / Tamil via Whisper)",
    version="1.0.0",
)

# Allow the Flutter app (any origin on LAN) to reach this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Whisper once at startup ─────────────────────────────────────────────
# Change model_size to "medium" or "large" for better Sinhala/Tamil accuracy
# at the cost of more RAM and slower first load.
_asr = WhisperASR(model_size="small")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness check — Flutter app pings this before showing the screen."""
    return {"status": "ok", "model": _asr.model_size}


@app.post("/asr/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Query(
        default=None,
        description="Force language: 'si' = Sinhala, 'ta' = Tamil. Omit for auto-detect.",
    ),
):
    """
    Accept an audio file from the Flutter app and return transcribed text.

    - Saves upload to a temp file (Whisper needs a file path, not a stream).
    - Deletes the temp file after transcription regardless of success/failure.
    - Returns JSON matching AsrResult in asr_service.dart.
    """
    # Preserve original extension so ffmpeg can sniff the format correctly.
    original_name = file.filename or "audio.m4a"
    suffix = os.path.splitext(original_name)[1] or ".m4a"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = _asr.transcribe(tmp_path, language=language)
        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # host="0.0.0.0" makes the server reachable from the phone on the same Wi-Fi.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
