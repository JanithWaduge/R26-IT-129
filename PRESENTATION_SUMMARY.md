# SLSL App — Presentation Summary
## R26-IT-129 | Sri Lanka Sign Language Recognition App

---

## What We Built

A mobile app that converts **Sinhala/Tamil speech → Sri Lanka Sign Language animation** for deaf and mute students.

---

## Live Demo Flow

```
1. User speaks in Sinhala
        ↓
2. App transcribes speech using Whisper AI
        ↓
3. User taps "Translate to Sign Language"
        ↓
4. App shows animated hand skeleton performing the SLSL sign
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Signs in dataset | **382** (Nouns + Verbs) |
| Animation frames per sign | **30 frames** |
| Landmarks per frame | **21 hand landmarks** |
| Videos processed | **929 MP4 videos** |
| Detection failures | **0** (100% success) |
| Sinhala speech samples | **155,970** (OpenSLR52) |
| ASR timeout | **120 seconds** |
| Performers averaged per sign | Up to **3** (P1/P2/P3) |

---

## Technologies

| What | How |
|------|-----|
| Mobile App | Flutter (Android) |
| Speech Recognition | OpenAI Whisper — Sinhala fine-tuned |
| Hand Tracking | MediaPipe (21 landmarks) |
| Sign Animation | Flutter CustomPainter |
| ASR Backend | FastAPI (Python, port 8000) |
| Sign Backend | Flask (Python, port 5000) |
| ML Model (Obj 2) | CNN + LSTM (TensorFlow/TFLite) |

---

## System Architecture (Quick View)

```
[Microphone] → [Flutter App] → [FastAPI / Whisper]
                                        ↓
                              Sinhala/Tamil Text
                                        ↓
                           [Translation Dictionary]
                                        ↓
                         [Animated Hand Skeleton — 30fps]
```

```
[Camera] → [Flutter App] → [Flask / MediaPipe]
                                    ↓
                            [CNN + LSTM Model]
                                    ↓
                           Recognised Sign Text
```

---

## Our Novel Contributions

### 1. Multi-Performer Landmark Averaging
Hand keypoints extracted from up to 3 different performers per sign, then averaged — produces a more natural, generalised animation rather than any single person's style.

### 2. Velocity-Threshold Noise Filter (Janith)
During LSTM training, accidental hand movements are automatically filtered out by computing frame-to-frame velocity. Frames below threshold (0.02) are discarded — original approach applied to SLSL data.

### 3. Original SLSL Keypoint Dataset
382 Sri Lanka Sign Language signs converted from raw video into structured 30-frame × 21-landmark JSON — enables lightweight mobile animation with no video streaming required.

### 4. Bilingual Voice Pipeline
Single pipeline supports both Sinhala and Tamil voice input — both are low-resource languages underrepresented in global ASR research.

---

## What's Working Right Now

- ✅ Sinhala voice recording on phone
- ✅ Whisper ASR transcription (Sinhala)
- ✅ Animated hand skeleton for 382 signs
- ✅ Play/Pause and speed control (0.25× – 1×)
- ✅ Real-time sign recognition (Objective 2 — Janith)
- ✅ Full end-to-end voice → sign demo

---

## What's Coming Next

- Full Sinhala → Sign translation dictionary (382 words)
- Tamil translation support
- GPU-accelerated Whisper fine-tuning on Sinhala data
- Objectives 3 & 4

---

## Demo Signs (Available Now)

| Sinhala | Sign |
|---------|------|
| පොත | Book |
| පුටුව | Chair |
| පරිගණකය | Computer |
| පිළිතුර | Answer |

---

*R26-IT-129 | Gimhana (Objective 1) + Janith (Objective 2)*
