"""
extract_keypoints.py
--------------------
Dataset1 + Dataset2 දෙකම process කරනවා.

Dataset1 structure:
  dataset/Nouns/person1/Teacher.mp4
  dataset/Verbs/person1/Answer.mp4
  dataset/Nouns/person1/Teacher_0.mp4   ← collect_more.py records

Dataset2 structure (3 signs only):
  dataset2/Verbs/Copying/Copying_001.mp4
  dataset2/Verbs/Study/Study_001.mov
  dataset2/Verbs/Teacher/Teacher_001.mp4
"""

import mediapipe as mp
import cv2
import numpy as np
import os
import zipfile
import pandas as pd

# ================================================
# PATHS
# ================================================
DATASET1_PATH = r'D:\R26-IT-129\Janith\dataset'
DATASET2_ZIP  = r'D:\R26-IT-129\Janith\dataset2.zip'   # zip directly use කරනවා
DATASET2_PATH = r'D:\R26-IT-129\Janith\dataset2'        # extracted folder (if exists)
OUTPUT_CSV    = r'D:\R26-IT-129\Janith\keypoints_data.csv'

# ================================================
# CONFIG
# ================================================
SEQUENCE_LENGTH = 30
CAPTURE_FRAMES  = 15
NOISE_THRESHOLD = 0.02

# ================================================
# TARGET SIGNS — 30 classroom signs
# ================================================
TARGET_SIGNS = [
    'Allocate', 'Answer', 'Answer Properly', 'Answer Sheet', 'Ask Question',
    'Attend', 'Attending', 'Calculate', 'Cancel', 'Collaborating',
    'Collect', 'Comparing', 'Concentrate', 'Continuing', 'Coordinate',
    'Copying', 'Correct Mistake', 'Describe', 'Discuss', 'Discuss Topic',
    'Distribute', 'Documenting', 'Grade', 'Practice', 'Research',
    'Review', 'Study', 'Support', 'Teacher', 'Whiteboard Marker',
]

# Dataset2 folders → your sign labels (exact match දැන්)
DATASET2_SIGNS = {
    'Copying' : 'Copying',   # Verbs/Copying/
    'Study'   : 'Study',     # Verbs/Study/
    'Teacher' : 'Teacher',   # Verbs/Teacher/
}

# ================================================
# MEDIAPIPE — static_image_mode=True (matches server)
# ================================================
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ================================================
# NOISE FILTER — Research Contribution
# ================================================
def apply_noise_filter(sequence, threshold=NOISE_THRESHOLD):
    if len(sequence) < 2:
        return sequence, 0
    filtered = [sequence[0]]
    removed  = 0
    for i in range(1, len(sequence)):
        velocity = np.mean(np.abs(
            np.array(sequence[i]) - np.array(sequence[i - 1])
        ))
        if velocity > threshold:
            filtered.append(sequence[i])
        else:
            removed += 1
    return filtered, removed

# ================================================
# NORMALIZE — 15 frames + 15 zeros = 30
# Exactly matches app capture
# ================================================
def normalize_sequence(sequence):
    valid = [f for f in sequence if np.sum(np.abs(f)) > 0.01]
    if len(valid) == 0:
        return [[0.0] * 63] * SEQUENCE_LENGTH
    if len(valid) >= CAPTURE_FRAMES:
        indices = np.linspace(0, len(valid) - 1, CAPTURE_FRAMES, dtype=int)
        sampled = [valid[i] for i in indices]
    else:
        indices = np.linspace(0, len(valid) - 1, CAPTURE_FRAMES)
        sampled = [valid[int(round(i))] for i in indices]
    padding = [[0.0] * 63] * (SEQUENCE_LENGTH - CAPTURE_FRAMES)
    return sampled + padding

# ================================================
# EXTRACT KEYPOINTS FROM VIDEO FILE
# ================================================
def extract_from_file(video_path, sign_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
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
        else:
            kp = [0.0] * 63
        sequence.append(kp)
    cap.release()
    if len(sequence) == 0:
        return None
    filtered, _ = apply_noise_filter(sequence)
    normalized  = normalize_sequence(filtered)
    flat        = [val for frame in normalized for val in frame]
    return flat if len(flat) == 1890 else None

# ================================================
# EXTRACT KEYPOINTS FROM VIDEO BYTES (zip)
# ================================================
def extract_from_bytes(video_bytes, sign_name):
    import tempfile
    # Write to temp file (cv2 cannot read bytes directly)
    ext = '.mp4'
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    result = extract_from_file(tmp_path, sign_name)
    os.unlink(tmp_path)
    return result

# ================================================
# GET SIGN NAME FROM FILENAME (Dataset1)
# Teacher.mp4             → "Teacher"
# Teacher_0.mp4           → "Teacher"
# Whiteboard_Marker_3.mp4 → "Whiteboard Marker"
# ================================================
def get_sign_name(filename):
    name  = os.path.splitext(filename)[0]
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    return name.replace('_', ' ').strip()

# ================================================
# MAIN
# ================================================
all_data   = []
all_labels = []

print("🚀 Keypoint extraction started...")
print("=" * 60)
print(f"  Dataset1 : {DATASET1_PATH}")
print(f"  Dataset2 : {DATASET2_ZIP}")
print(f"  Signs    : {len(TARGET_SIGNS)}")
print(f"  Frames   : {CAPTURE_FRAMES} active + {SEQUENCE_LENGTH-CAPTURE_FRAMES} padding = {SEQUENCE_LENGTH}")
print(f"  MediaPipe: static_image_mode=True (matches server)")
print("=" * 60)

# ════════════════════════════════════════════
# DATASET 1 — person folder structure
# ════════════════════════════════════════════
print("\n📁 DATASET 1")
d1_count = 0

for category in ['Nouns', 'Verbs']:
    cat_path = os.path.join(DATASET1_PATH, category)
    if not os.path.exists(cat_path):
        print(f"  ⚠️  {category} not found")
        continue
    print(f"  [{category}]")

    for person in ['person1', 'person2', 'person3']:
        person_path = os.path.join(cat_path, person)
        if not os.path.exists(person_path):
            continue

        for video_file in sorted(os.listdir(person_path)):
            ext = os.path.splitext(video_file)[1].lower()
            if ext not in ['.mp4', '.mov', '.avi']:
                continue

            sign_name  = get_sign_name(video_file)
            video_path = os.path.join(person_path, video_file)

            if sign_name not in TARGET_SIGNS:
                continue

            flat = extract_from_file(video_path, sign_name)
            if flat:
                all_data.append(flat)
                all_labels.append(sign_name)
                d1_count += 1
                print(f"    ✅ {sign_name:25s} ← {person}/{video_file}")

print(f"\n  📊 Dataset1: {d1_count} samples")

# ════════════════════════════════════════════
# DATASET 2 — from zip directly
# Copying, Study, Teacher only
# ════════════════════════════════════════════
print("\n📁 DATASET 2 (from zip — Copying, Study, Teacher)")
d2_count = 0

# First try extracted folder, then zip
use_zip = not os.path.exists(DATASET2_PATH)

if use_zip:
    if not os.path.exists(DATASET2_ZIP):
        print(f"  ⚠️  dataset2.zip not found at {DATASET2_ZIP}")
        print(f"  ⚠️  Skipping dataset2")
    else:
        print(f"  Reading from zip...")
        import tempfile

        with zipfile.ZipFile(DATASET2_ZIP, 'r') as zf:
            for zip_path in sorted(zf.namelist()):
                # Only mp4/mov files
                ext = os.path.splitext(zip_path)[1].lower()
                if ext not in ['.mp4', '.mov', '.avi']:
                    continue

                # Check if this is one of our 3 sign folders
                # Path: dataset2/Verbs/Copying/Copying_001.mp4
                parts = zip_path.split('/')
                if len(parts) < 4:
                    continue

                folder_name = parts[2]   # Copying / Study / Teacher
                if folder_name not in DATASET2_SIGNS:
                    continue

                sign_name = DATASET2_SIGNS[folder_name]

                # Extract to temp and process
                video_bytes = zf.read(zip_path)
                ext_use = ext if ext != '.mov' else '.mov'
                with tempfile.NamedTemporaryFile(suffix=ext_use, delete=False) as tmp:
                    tmp.write(video_bytes)
                    tmp_path = tmp.name

                flat = extract_from_file(tmp_path, sign_name)
                os.unlink(tmp_path)

                if flat:
                    all_data.append(flat)
                    all_labels.append(sign_name)
                    d2_count += 1
                    print(f"    ✅ {sign_name:25s} ← {parts[-1]}")
else:
    print(f"  Reading from extracted folder...")
    for sign_folder, sign_name in DATASET2_SIGNS.items():
        sign_path = os.path.join(DATASET2_PATH, 'Verbs', sign_folder)
        if not os.path.exists(sign_path):
            print(f"  ⚠️  Not found: Verbs/{sign_folder}")
            continue

        print(f"  [Verbs/{sign_folder}] → '{sign_name}'")
        for video_file in sorted(os.listdir(sign_path)):
            ext = os.path.splitext(video_file)[1].lower()
            if ext not in ['.mp4', '.mov', '.avi']:
                continue
            video_path = os.path.join(sign_path, video_file)
            flat = extract_from_file(video_path, sign_name)
            if flat:
                all_data.append(flat)
                all_labels.append(sign_name)
                d2_count += 1
                print(f"    ✅ {sign_name:25s} ← {video_file}")

print(f"\n  📊 Dataset2: {d2_count} samples")

# ════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"✅ Dataset1 samples  : {d1_count}")
print(f"✅ Dataset2 samples  : {d2_count}")
print(f"✅ Total samples     : {len(all_data)}")
print(f"✅ Unique signs      : {len(set(all_labels))}")

if len(all_data) == 0:
    print("\n❌ No data extracted!")
    exit(1)

# Save CSV
print("\n💾 Saving CSV...")
columns = [f'f{i}' for i in range(1890)]
df = pd.DataFrame(all_data, columns=columns)
df['label'] = all_labels
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved : {OUTPUT_CSV}")
print(f"📊 Shape : {df.shape}")

# Samples per sign report
print(f"\n📋 Samples per sign:")
counts = df['label'].value_counts()
print(counts.to_string())

# Missing signs
missing = [s for s in TARGET_SIGNS if s not in counts.index]
if missing:
    print(f"\n❌ Missing signs ({len(missing)}) — record with collect_more.py:")
    for s in missing:
        print(f"   - {s}")

# Low samples
low = counts[counts < 10]
if len(low) > 0:
    print(f"\n⚠️  Signs with < 10 samples (need more recording):")
    for sign, cnt in low.items():
        needed = 10 - cnt
        print(f"   - {sign}: {cnt} samples (record {needed} more)")
    print(f"\n   💡 python training/collect_more.py")
else:
    print(f"\n✅ All signs 10+ samples — ready to train!")

print("\n🎉 Extraction complete!")
print("   python training/clean_data.py")
print("   python training/train_model.py")