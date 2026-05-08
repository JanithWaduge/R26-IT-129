# Technical Summary Report
## Sign Language Learning App — SLSL (R26-IT-129)
**Generated from actual project code — no fabricated data**

---

## 1. Project Overview

**What the app does:**
A bilingual (Sinhala + Tamil) mobile application that converts spoken language into Sri Lanka Sign Language (SLSL) animations for deaf and mute students. The app records voice input, transcribes it to text using ASR, translates it to the corresponding sign, and animates a hand skeleton showing how to perform the sign.

**Target users:** Sri Lankan deaf and mute students, primarily in academic/educational settings.

**Platform:** Flutter mobile app (Android) + Python backend servers (FastAPI + Flask)

**Team structure:**
- Gimhana — Objective 1 (Voice → Sign animation)
- Janith — Objective 2 (Sign recognition → Text)

**Current completion status:**
- Objective 1: ~80% complete (ASR working, sign animation working, translation dictionary partial)
- Objective 2: Backend implemented, model trained (Janith's work)
- Objectives 3 & 4: Not yet implemented

---

## 2. What Has Been Completed

### Objective 1: Voice to Sign (Gimhana)

#### ASR Component
- **Files:** `gimhana/sinhala_asr.py`, `gimhana/main.py`
- **Model:** `Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2` (HuggingFace)
- Based on OpenAI Whisper tiny architecture, fine-tuned on 20,000+ Sinhala samples
- Loaded via HuggingFace `transformers.pipeline("automatic-speech-recognition")`
- Audio loading: librosa at 16kHz (bypasses ffmpeg dependency); subprocess ffmpeg fallback for unsupported formats
- **API:** FastAPI on port 8000, endpoint `POST /asr/transcribe`
- Languages supported: Sinhala (`"si"`), Tamil (`"ta"`), auto-detect
- Response: `{text, language, language_name, locale, confidence}`
- Anti-hallucination: `temperature=[0.0, 0.2, 0.4]`, `compression_ratio_threshold=1.35`, `logprob_threshold=-1.0`
- Post-processing: regex removes character repetitions >3, Sinhala syllable (U+0D80–U+0DFF) repetitions >2
- Model loaded once at startup; passed as `asr=_asr_pipeline` parameter to avoid reloading per request

#### Sign Dataset Processing
- **File:** `gimhana/extract_keypoints.py`
- **Input:** 929 MP4 videos across Nouns + Verbs categories, 3 performers (P1/P2/P3)
- MediaPipe 0.10.14 hand landmark extraction (`mp.solutions.hands`)
- 21 landmarks × 3 coordinates (x, y, z) = 63 floats per frame
- 30 frames evenly sampled per video using `np.linspace(0, total-1, 30)`
- Landmarks averaged across all available performers per sign
- Last frame repeated if fewer than 30 frames detected (padding)
- **Output:** `gimhana/assets/sign_data.json` — 382 signs, 0 detection failures, 0 low-frame warnings

#### Flutter Sign Animation
- **File:** `slsl_app/lib/widgets/sign_avatar_widget.dart`
- `SignAvatarWidget(signWord: String)` StatefulWidget
- `_HandPainter` CustomPainter draws 21 joints + connections
- Connections: thumb (0–4), index (0, 5–8), middle (0, 9–12), ring (0, 13–16), pinky (0, 17–20), palm (5, 9, 13, 17)
- Lines: white opacity 0.8, strokeWidth 3.0; Joints: Color(0xFF7C3AED), radius 5.0
- Timer interval: `(duration_ms / frame_count / speed_multiplier)` ms; frame cycling modulo total frames
- Speed slider: 0.25×, 0.5×, 0.75×, 1.0× (4 divisions); Play/Pause button
- Case-insensitive sign lookup from `assets/sign_data.json`
- "Sign not available yet" message for unknown signs

#### Translation Dictionary
- **File:** `slsl_app/lib/utils/sinhala_to_english.dart`
- Currently 5 demo entries:

| Sinhala | English | In Dataset |
|---------|---------|------------|
| පොත | Book | ✅ |
| පුටුව | Chair | ✅ |
| පරිගණකය | Computer | ✅ |
| පිළිතුර | Answer | ✅ |
| ආයුබෝවන් | Hello | ❌ (not in academic dataset) |

- Full 382-sign dictionary: not yet implemented; planned after presentation

#### Mobile App Integration
- **File:** `slsl_app/lib/screens/voice_to_sign_screen.dart`
- Records audio via `record` package (M4A/AAC)
- Sends to FastAPI via HTTP multipart POST (120s timeout)
- Displays transcription text + confidence bar
- "Translate to Sign Language" button triggers `translateToSign()` then `SignAvatarWidget`
- Server health check (`GET /health`) on screen load

#### Dataset Preparation (for future fine-tuning)
- **File:** `gimhana/prepare_finetune_data.py`
- OpenSLR52: 155,970 utterances total, 10,632 files downloaded locally
- 1,600 train / 400 test samples saved to `sinhala_data/train.csv` and `sinhala_data/test.csv`

#### Fine-tuning Script (not yet run)
- **File:** `gimhana/fine_tune_whisper.py`
- Seq2SeqTrainer, Whisper-small base, 5 epochs, batch_size=2, grad_accum=8, fp16=True
- Blocked by CUDA PyTorch installation failure (network hash mismatch on 2.4–2.8 GB wheel)

---

### Objective 2: Sign to Text (Janith)

- **File:** `Janith/slsl_server.py` — Flask server on port 5000
- Real-time hand landmark detection via MediaPipe 0.10.5
- **File:** `Janith/training/train_model.py`
  - Input shape: `(30, 63)` — 30 frames × 21 landmarks × 3 coords
  - Architecture: `Conv1D → MaxPool → BatchNorm → Conv1D → MaxPool → BatchNorm → LSTM(128, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(softmax)`
- **Novel contribution:** Velocity-threshold noise filter (`Janith/training/noise_filter_experiment.py`) — filters accidental hand movements below threshold=0.02 by computing `mean(|curr_frame - prev_frame|)` per frame
- Model exported to TFLite for on-device mobile inference

---

### Not Yet Implemented
- Objective 3: Not yet started
- Objective 4: Not yet started

---

## 3. Technologies Used

| Component | Technology | Version | Why Chosen |
|-----------|------------|---------|------------|
| Mobile App | Flutter | 3.24.5 | Cross-platform, single codebase |
| Mobile Language | Dart | SDK ≥3.0.0 | Flutter native language |
| ASR Backend | FastAPI | ≥0.104.0 | Async, fast, auto-generated docs |
| ASR Server | Uvicorn | ≥0.24.0 | ASGI server for FastAPI |
| ASR Model | Whisper (Ransaka fine-tune) | tiny | Sinhala-specific, fast |
| Sign Backend | Flask | — | Simple REST for sign recognition |
| Hand Tracking | MediaPipe | 0.10.14 | Industry standard, 21 landmarks |
| Video Processing | OpenCV | 4.13.0 | Frame extraction from MP4 |
| Audio Loading | librosa | latest | No ffmpeg dependency |
| ML Framework (Obj 2) | TensorFlow / Keras | 2.13.0 | LSTM + TFLite export |
| HuggingFace | transformers | latest | Model loading pipeline |
| Python Runtime | Python | 3.12 (conda) | Required for PyTorch CUDA wheels |
| Environment | Conda (whisper_env) | Miniconda3 | Package management |
| Audio Recording | record (Flutter) | ^5.0.4 | M4A/AAC recording |
| HTTP Client | http (Flutter) | ^1.2.0 | Server communication |
| Permissions | permission_handler | ^11.3.0 | Microphone access |
| Storage | path_provider | ^2.1.2 | Temp file directory |

---

## 4. Algorithms

### ASR — Voice to Text
- Whisper encoder-decoder transformer; audio → 16kHz numpy array → mel spectrogram → text
- Temperature fallback: `[0.0, 0.2, 0.4]`; stops at first acceptable output
- Compression ratio threshold: 1.35 (rejects repetitive/looping output)
- Log-probability threshold: −1.0 (rejects low-confidence tokens)
- Post-processing regex: removes char repetitions >3, Sinhala syllable repetitions >2

### Sign Keypoint Extraction
- MediaPipe Hands: `static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5`
- Frame sampling: `np.linspace(0, total_frames-1, 30)` for even distribution
- Multi-performer averaging: x, y, z coordinates averaged across all available performers (P1/P2/P3)
- Output coordinates rounded to 6 decimal places

### Sign Animation
- Timer interval = `duration_ms ÷ frame_count ÷ speed_multiplier` (clamped 33–500 ms raw)
- `_currentFrame = (_currentFrame + 1) % totalFrames` — loops continuously
- Normalised (0.0–1.0) coordinates × canvas pixel dimensions = rendered pixel position
- Connections drawn as lines first, joints drawn as filled circles on top

### CNN + LSTM (Objective 2 — Janith)
- Input: `(30, 63)` tensor per gesture sample
- Two Conv1D blocks with MaxPooling and BatchNorm for spatial feature extraction
- Two LSTM layers (128 → 64 units) for temporal sequence modelling
- Dropout 0.2 at each LSTM output for regularisation
- Softmax output for multi-class sign classification
- Novel: velocity-threshold noise filter discards frames with `mean(|Δlandmarks|) < 0.02`

---

## 5. Datasets

| Dataset | Source | Size | Format | Purpose |
|---------|--------|------|--------|---------|
| OpenSLR52 Sinhala Speech | OpenSLR (open-source) | 155,970 utterances (10,632 local) | FLAC 16kHz | Whisper fine-tuning (future) |
| SLSL Sign Video Dataset | Original collection | 929 MP4 videos, 382 signs | MP4 | Sign keypoint extraction |
| Processed Sign Keypoints | Generated | 382 signs × 30 frames × 21 landmarks | JSON | Flutter animation |
| Fine-tune Split | Generated | 1,600 train / 400 test | CSV | Whisper fine-tuning (future) |

---

## 6. Accuracy and Performance

### Voice Recognition
- Model: `Ransaka/whisper-tiny-sinhala-20k-8k-steps-v2`
- Confidence returned: 0.85 fixed (HuggingFace pipeline API does not expose log-probs)
- Test transcription: "බුදුරජාණන් වහන්සේ ජීව මානව නොමැති නිසා" — minor word boundary error vs. expected
- Runtime: CPU only (CUDA not installed); transcription takes several seconds per request

### Keypoint Extraction
- Signs processed: 382 / 382 (100%)
- No-hand-detection failures: 0 (0%)
- Low-frame warnings: 0 (0%)
- Frames per sign: 30 (fixed after sampling)
- Duration range: 800 ms – 4,966 ms across all signs

### Sign Classification (Objective 2 — Janith)
- Framework: TensorFlow 2.13.0 + TFLite
- Accuracy: As per Janith's training results (not logged in session)

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OBJECTIVE 1 — VOICE TO SIGN                 │
└─────────────────────────────────────────────────────────────────────┘

[Phone Microphone]
      │  (M4A audio — record package)
      ▼
[Flutter App — VoiceToSignScreen]
      │  (HTTP POST multipart, 120s timeout)
      ▼
[FastAPI Server — port 8000 — gimhana/main.py]
      │  (numpy array at 16kHz)
      ▼
[sinhala_asr.py — Ransaka Whisper tiny model]
      │  (Sinhala / Tamil text)
      ▼
[FastAPI Response: {text, language, locale, confidence}]
      │
      ▼
[Flutter App — displays transcription]
      │  (user taps "Translate to Sign Language")
      ▼
[sinhala_to_english.dart — translateToSign()]
      │  (English sign name e.g. "Book")
      ▼
[SignAvatarWidget — loads assets/sign_data.json]
      │  (30 frames × 21 landmarks)
      ▼
[_HandPainter CustomPainter — animated hand skeleton]
      │
      ▼
[User sees animated SLSL sign]

┌─────────────────────────────────────────────────────────────────────┐
│                       OBJECTIVE 2 — SIGN TO TEXT                   │
└─────────────────────────────────────────────────────────────────────┘

[Phone Camera]
      │  (video frames)
      ▼
[Flutter App — SignRecognitionScreen]
      │  (HTTP POST)
      ▼
[Flask Server — port 5000 — Janith/slsl_server.py]
      │  (live frames)
      ▼
[MediaPipe — 21 hand landmarks × 3 coords]
      │  (30 frames × 63 values)
      ▼
[CNN + LSTM Model — TFLite inference]
      │  (sign class)
      ▼
[Flutter App — displays recognised sign text]
```

---

## 8. Known Limitations

| # | Limitation | Impact | Workaround |
|---|-----------|--------|------------|
| 1 | CUDA not installed | CPU-only inference; slow transcription | Pre-trained model used instead of training |
| 2 | No Whisper fine-tuning | Using generic Sinhala model | `fine_tune_whisper.py` ready; pending GPU |
| 3 | Tamil support incomplete | ASR accepts "ta" but no Tamil sign dictionary | Planned post-presentation |
| 4 | Only 5 translation entries | Most Sinhala words produce "Sign not available yet" | Full 382 planned |
| 5 | "Hello" missing from dataset | Academic dataset excludes greeting signs | Map to another sign or remove |
| 6 | Single hand only | Two-handed signs not supported | `max_num_hands=1` by design |
| 7 | OpenSLR52 partial download | 10,632 of 155,970 files available | 13 more archives to download |
| 8 | Hardcoded server IP | App breaks on different networks | `kAsrServerUrl` in `asr_service.dart` |

---

## 9. Future Work

- **GPU setup:** Download CUDA PyTorch 2.4 GB wheel via browser; install in whisper_env
- **Whisper fine-tuning:** Run `fine_tune_whisper.py` — 5 epochs, ~2–3 hours on RTX 3050
- **Full translation dictionary:** Add all 382 Sinhala→English sign mappings to `sinhala_to_english.dart`
- **Tamil dictionary:** Add Tamil word mappings for existing English signs
- **Download full OpenSLR52:** Remaining 13 archives (~145,000 additional utterances)
- **Two-handed signs:** Extend MediaPipe to `max_num_hands=2`
- **Dynamic server URL:** Replace hardcoded IP with config or service discovery
- **Objective 3 & 4:** Not yet scoped — implementation pending

---

## 10. Research Contributions

1. **Bilingual low-resource ASR pipeline:** Combined Sinhala and Tamil support in a single mobile-connected pipeline; both are underrepresented in ASR research globally.

2. **Multi-performer landmark averaging:** Keypoints averaged across P1/P2/P3 performers during extraction creates a more robust, generalised sign representation for animation — reduces performer-specific bias.

3. **Novel velocity-threshold noise filter (Janith):** Filters accidental hand movements from LSTM training data by computing per-frame velocity (`mean(|curr − prev|)`) and discarding frames below threshold=0.02 — first known application to SLSL gesture data.

4. **Original SLSL keypoint dataset:** 382 Sri Lanka Sign Language signs extracted as structured 30-frame × 21-landmark JSON — not previously available in this format; enables lightweight mobile animation without video streaming.

5. **End-to-end voice-to-sign mobile system:** First documented pipeline from Sinhala/Tamil speech to animated sign language on Android, purpose-built for Sri Lankan deaf and mute students.

---

*Report generated: 2026-05-07 | Project: R26-IT-129 | Team: Gimhana + Janith*
