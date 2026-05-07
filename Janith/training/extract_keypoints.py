import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd

# ================================================
# PATHS
# ================================================
DATASET_PATH = r'D:\R26-IT-129\Janith\dataset'
OUTPUT_CSV   = r'D:\R26-IT-129\Janith\keypoints_data.csv'

# ================================================
# CONFIG — App එකත් match වෙන values
# ================================================
SEQUENCE_LENGTH   = 30    # model input (pad target)
CAPTURE_FRAMES    = 15    # app captures 15 frames
NOISE_THRESHOLD   = 0.02  # research contribution value

# ================================================
# MEDIAPIPE SETUP
# ✅ FIX 1: static_image_mode=True
#    Server එකත් static_image_mode=True use කරනවා.
#    Training + inference ගෙදී same mode ➜ consistent keypoints.
# ================================================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,          # ← FIX: was False, server uses True
    max_num_hands=1,
    min_detection_confidence=0.5,    # ← match server config
    min_tracking_confidence=0.5
)

# ================================================
# NOISE FILTER — Novel Research Contribution ✅
# Same logic as slsl_server.py apply_noise_filter()
# ================================================
def apply_noise_filter(sequence, threshold=NOISE_THRESHOLD):
    """
    Remove accidental low-velocity frames.
    Returns (filtered_sequence, removed_count)
    """
    if len(sequence) < 2:
        return sequence, 0

    filtered = [sequence[0]]
    removed  = 0

    for i in range(1, len(sequence)):
        prev     = np.array(sequence[i - 1])
        curr     = np.array(sequence[i])
        velocity = np.mean(np.abs(curr - prev))

        if velocity > threshold:
            filtered.append(sequence[i])   # intentional movement ✅
        else:
            removed += 1                   # accidental movement ❌

    return filtered, removed


# ================================================
# NORMALIZE TO FIXED LENGTH
# ✅ FIX 2: Evenly sample CAPTURE_FRAMES(15) from valid
#    frames, then pad zeros to SEQUENCE_LENGTH(30).
#    This EXACTLY matches what the app sends to server:
#      - App captures 15 frames
#      - Pads to 30 for model input
# ================================================
def normalize_sequence(sequence, target_frames=CAPTURE_FRAMES, total_length=SEQUENCE_LENGTH):
    """
    Step 1: Remove zero frames (no hand detected)
    Step 2: Evenly sample 15 frames from valid frames
    Step 3: Pad with zeros to reach 30 frames total
    """
    # Step 1 — Remove zero/empty frames
    valid = [f for f in sequence if np.sum(np.abs(f)) > 0.01]

    if len(valid) == 0:
        # No hand detected at all — return zeros
        return [[0.0] * 63] * total_length

    # Step 2 — Evenly sample target_frames (15) from valid
    if len(valid) >= target_frames:
        indices = np.linspace(0, len(valid) - 1, target_frames, dtype=int)
        sampled = [valid[i] for i in indices]
    else:
        # Less than 15 valid frames — interpolate
        indices = np.linspace(0, len(valid) - 1, target_frames)
        sampled = [valid[int(round(i))] for i in indices]

    # Step 3 — Pad zeros to reach total_length (30)
    padding = [[0.0] * 63] * (total_length - target_frames)
    return sampled + padding


# ================================================
# KEYPOINT EXTRACTION FROM VIDEO
# ================================================
def extract_keypoints_from_video(video_path):
    """
    Extract MediaPipe hand keypoints from every frame of video.
    Returns list of keypoint vectors (63 floats each).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  ⚠️  Cannot open: {video_path}")
        return []

    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            kp = []
            for point in lm.landmark:
                kp.extend([point.x, point.y, point.z])
            # 21 landmarks × 3 = 63 values
        else:
            kp = [0.0] * 63  # No hand detected

        sequence.append(kp)

    cap.release()
    return sequence


# ================================================
# MAIN — PROCESS ENTIRE DATASET
# ================================================
all_data      = []
all_labels    = []
total_removed = 0
errors        = []
skipped       = []

print("🚀 Keypoint extraction started...")
print("=" * 60)
print(f"  Dataset    : {DATASET_PATH}")
print(f"  Seq length : {SEQUENCE_LENGTH} frames (model input)")
print(f"  Capture    : {CAPTURE_FRAMES} active frames + padding")
print(f"  Threshold  : {NOISE_THRESHOLD} (noise filter)")
print(f"  MediaPipe  : static_image_mode=True (matches server)")
print("=" * 60)

for category in ['Nouns', 'Verbs']:
    cat_path = os.path.join(DATASET_PATH, category)

    if not os.path.exists(cat_path):
        print(f"\n⚠️  {category} folder not found — skipping")
        continue

    print(f"\n📁 Category: {category}")

    for person in ['person1', 'person2', 'person3']:
        person_path = os.path.join(cat_path, person)

        if not os.path.exists(person_path):
            continue

        print(f"  👤 {person}")

        for video_file in sorted(os.listdir(person_path)):
            if not video_file.lower().endswith('.mp4'):
                continue

            sign_name  = os.path.splitext(video_file)[0]
            video_path = os.path.join(person_path, video_file)

            try:
                # Step 1 — Extract raw keypoints from video
                raw = extract_keypoints_from_video(video_path)

                if len(raw) == 0:
                    skipped.append(video_path)
                    print(f"    ⚠️  Skipped (empty): {video_file}")
                    continue

                # Step 2 — Apply noise filter
                filtered, removed = apply_noise_filter(raw)
                total_removed += removed

                # Step 3 — Normalize: sample 15, pad to 30
                normalized = normalize_sequence(filtered)

                # Step 4 — Flatten (30 × 63 = 1890 numbers)
                flat = [val for frame in normalized for val in frame]

                # Validate length
                if len(flat) != 1890:
                    errors.append(f"Length error ({len(flat)}): {video_path}")
                    continue

                all_data.append(flat)
                all_labels.append(sign_name)

                print(f"    ✅ {sign_name} "
                      f"({len(raw)} raw → {len(filtered)} filtered → 15+15pad)")

            except Exception as e:
                errors.append(f"{video_path}: {e}")
                print(f"    ❌ Error: {video_file} → {e}")

# ================================================
# SUMMARY
# ================================================
print("\n" + "=" * 60)
print(f"✅ Total samples    : {len(all_data)}")
print(f"✅ Unique signs     : {len(set(all_labels))}")
print(f"🔬 Frames removed  : {total_removed} (noise filter)")
print(f"⚠️  Skipped         : {len(skipped)}")
print(f"❌ Errors          : {len(errors)}")

if len(all_data) == 0:
    print("\n❌ No data extracted! Check dataset path and video files.")
    exit(1)

# ================================================
# SAVE TO CSV
# ================================================
print("\n💾 Saving CSV...")

columns = [f'f{i}' for i in range(1890)]
df      = pd.DataFrame(all_data, columns=columns)
df['label'] = all_labels

df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved: {OUTPUT_CSV}")
print(f"📊 Shape: {df.shape}")

print(f"\n📋 Samples per sign:")
counts = df['label'].value_counts()
print(counts.to_string())

low = counts[counts < 5]
if len(low) > 0:
    print(f"\n⚠️  WARNING: {len(low)} signs have < 5 samples.")
    print("   Consider collecting more videos for these signs:")
    print(low.to_string())
    print("\n   💡 Tip: Run collect_more.py to record extra samples.")
else:
    print(f"\n✅ All signs have ≥ 5 samples — good for training!")

print("\n🎉 Extraction complete!")
print("   Next: python training/clean_data.py")
print("         python training/train_model.py")