"""
extract_keypoints.py
Extracts hand landmarks from sign language videos using MediaPipe 0.10.5.
Processes gimhana/sign_dataset/Data/{Nouns,Verbs}/P1,P2,P3/
Outputs: gimhana/assets/sign_data.json

NEW file — does NOT modify any Objective 2 files.
"""

import os
import sys
import io
import json
import cv2
import numpy as np
import mediapipe as mp

# Force UTF-8 output so emoji prints correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "sign_dataset", "Data")
OUTPUT_DIR   = os.path.join(BASE_DIR, "assets")
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, "sign_data.json")

CATEGORIES   = ["Nouns", "Verbs"]
PERFORMERS   = ["P1", "P2", "P3"]
MAX_FRAMES   = 30   # fixed sequence length for LSTM + Flutter animation

# ── MediaPipe setup (same version/config as Janith's code) ───────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_frames(video_path: str) -> list:
    """
    Read every frame of a video and extract 21 hand landmarks.
    Returns list of frames; each frame is a list of 21 landmark dicts.
    If no hand detected in a frame, that frame is skipped.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            landmarks = [
                {"id": i, "x": round(p.x, 6),
                           "y": round(p.y, 6),
                           "z": round(p.z, 6)}
                for i, p in enumerate(lm.landmark)
            ]
            frames.append(landmarks)

    cap.release()
    return frames


def sample_frames(frames: list, n: int = MAX_FRAMES) -> list:
    """
    Evenly sample exactly n frames from the extracted frames list.
    If fewer than n frames available, pad with the last frame.
    """
    total = len(frames)
    if total == 0:
        return []
    if total >= n:
        indices = np.linspace(0, total - 1, n, dtype=int)
        return [frames[i] for i in indices]
    else:
        padding = [frames[-1]] * (n - total)
        return frames + padding


def get_video_duration_ms(video_path: str) -> int:
    """Return video duration in milliseconds using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int((frame_count / fps) * 1000)


# ── Main extraction loop ──────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sign_data       = {}
    no_hand_signs   = []
    low_frame_signs = []

    print("=" * 60)
    print("  SIGN KEYPOINT EXTRACTION")
    print("=" * 60)

    # Collect all unique sign names across categories and performers
    all_signs = set()
    for category in CATEGORIES:
        for performer in PERFORMERS:
            folder = os.path.join(DATASET_DIR, category, performer)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if f.lower().endswith(".mp4"):
                    all_signs.add(os.path.splitext(f)[0])

    total_signs = len(all_signs)
    print(f"Signs found in dataset : {total_signs}")
    print(f"Categories             : {', '.join(CATEGORIES)}")
    print(f"Performers             : {', '.join(PERFORMERS)}")
    print(f"Max frames per sign    : {MAX_FRAMES}")
    print()

    processed = 0

    for sign_name in sorted(all_signs):
        all_performer_frames = []
        duration_ms = 0

        for category in CATEGORIES:
            for performer in PERFORMERS:
                video_path = os.path.join(
                    DATASET_DIR, category, performer, f"{sign_name}.mp4"
                )
                if not os.path.exists(video_path):
                    continue

                raw_frames = extract_frames(video_path)
                if raw_frames:
                    sampled = sample_frames(raw_frames)
                    all_performer_frames.append(sampled)
                    if duration_ms == 0:
                        duration_ms = get_video_duration_ms(video_path)

        # No hands detected in any video for this sign
        if not all_performer_frames:
            no_hand_signs.append(sign_name)
            print(f"  ⚠️  {sign_name:<35} — no hands detected")
            continue

        # Warn if very few frames detected
        avg_frames_count = sum(
            len(pf) for pf in all_performer_frames
        ) // len(all_performer_frames)
        if avg_frames_count < 5:
            low_frame_signs.append(sign_name)
            print(f"  ⚠️  {sign_name:<35} — only {avg_frames_count} frames (< 5)")

        # Average landmarks across performers frame by frame
        num_performers = len(all_performer_frames)
        merged = []
        for frame_idx in range(MAX_FRAMES):
            avg_landmarks = []
            for lm_idx in range(21):
                avg_x = sum(
                    pf[frame_idx][lm_idx]["x"]
                    for pf in all_performer_frames
                ) / num_performers
                avg_y = sum(
                    pf[frame_idx][lm_idx]["y"]
                    for pf in all_performer_frames
                ) / num_performers
                avg_z = sum(
                    pf[frame_idx][lm_idx]["z"]
                    for pf in all_performer_frames
                ) / num_performers
                avg_landmarks.append({
                    "id": lm_idx,
                    "x":  round(avg_x, 6),
                    "y":  round(avg_y, 6),
                    "z":  round(avg_z, 6),
                })
            merged.append({"landmarks": avg_landmarks})

        sign_data[sign_name] = {
            "frames":      merged,
            "duration_ms": duration_ms,
            "frame_count": MAX_FRAMES,
        }

        processed += 1
        print(f"  ✅ {sign_name:<35} — {num_performers} performer(s), "
              f"{duration_ms} ms")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {"signs": sign_data}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Total signs processed  : {processed}")
    print(f"  Frames per sign        : {MAX_FRAMES}")
    print(f"  No hands detected      : {len(no_hand_signs)}")
    if no_hand_signs:
        for s in no_hand_signs:
            print(f"    - {s}")
    print(f"  Low frame warnings     : {len(low_frame_signs)}")
    if low_frame_signs:
        for s in low_frame_signs:
            print(f"    - {s}")
    print(f"  Output saved to        : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
