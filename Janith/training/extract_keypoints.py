import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd

# ================================================
# PATHS — ඔයාගේ path ගෙදී replace කරන්න
# ================================================
DATASET_PATH = r'C:\Users\Janith\Desktop\R26-IT-129\Janith\dataset'
OUTPUT_CSV   =  r'C:\Users\Janith\Desktop\R26-IT-129\Janith\keypoints_data.csv'

SEQUENCE_LENGTH = 30  # frames per sign
NOISE_THRESHOLD = 0.02  # research experiment value

# ================================================
# MEDIAPIPE SETUP
# ================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================================================
# NOISE FILTER — Novel Research Part ✅
# ================================================
def apply_noise_filter(sequence, threshold=NOISE_THRESHOLD):
    if len(sequence) < 2:
        return sequence
    
    filtered = [sequence[0]]
    removed = 0
    
    for i in range(1, len(sequence)):
        prev = np.array(sequence[i-1])
        curr = np.array(sequence[i])
        velocity = np.mean(np.abs(curr - prev))
        
        if velocity > threshold:
            filtered.append(sequence[i])  # Intentional sign ✅
        else:
            removed += 1  # Accidental movement ❌ ignore
    
    return filtered, removed

# ================================================
# KEYPOINT EXTRACTION
# ================================================
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            kp = []
            for point in lm.landmark:
                kp.extend([point.x, point.y, point.z])
            # 21 points × 3 = 63 numbers
        else:
            kp = [0.0] * 63
        
        sequence.append(kp)
    
    cap.release()
    return sequence

# ================================================
# NORMALIZE TO FIXED LENGTH
# ================================================
def normalize_sequence(sequence, length=SEQUENCE_LENGTH):
    if len(sequence) >= length:
        return sequence[:length]
    else:
        padding = [[0.0] * 63] * (length - len(sequence))
        return sequence + padding

# ================================================
# MAIN — PROCESS ENTIRE DATASET
# ================================================
all_data   = []
all_labels = []
total_removed = 0
errors = []

print("🚀 Processing started...")
print("=" * 60)

for category in ['Nouns', 'Verbs']:
    cat_path = os.path.join(DATASET_PATH, category)
    
    if not os.path.exists(cat_path):
        print(f"⚠️  {category} folder not found!")
        continue
    
    print(f"\n📁 Category: {category}")
    
    for person in ['person1', 'person2', 'person3']:
        person_path = os.path.join(cat_path, person)
        
        if not os.path.exists(person_path):
            continue
        
        for video_file in os.listdir(person_path):
            if not video_file.lower().endswith('.mp4'):
                continue
            
            # Sign name = file name (without .mp4)
            sign_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(person_path, video_file)
            
            try:
                # Step 1: Extract
                raw = extract_keypoints(video_path)
                
                # Step 2: Noise Filter (NOVEL ✅)
                filtered, removed = apply_noise_filter(raw)
                total_removed += removed
                
                # Step 3: Normalize
                normalized = normalize_sequence(filtered)
                
                # Step 4: Flatten (30 × 63 = 1890 numbers)
                flat = [val for frame in normalized for val in frame]
                
                all_data.append(flat)
                all_labels.append(sign_name)
                
            except Exception as e:
                errors.append(f"{video_path}: {e}")

print("\n" + "=" * 60)
print(f"✅ Total samples    : {len(all_data)}")
print(f"✅ Unique signs     : {len(set(all_labels))}")
print(f"🔬 Frames removed  : {total_removed} (noise filter)")
print(f"❌ Errors          : {len(errors)}")

# ================================================
# SAVE TO CSV
# ================================================
print("\n💾 Saving CSV...")

columns = [f'f{i}' for i in range(1890)]
df = pd.DataFrame(all_data, columns=columns)
df['label'] = all_labels

df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved: {OUTPUT_CSV}")
print(f"📊 Shape: {df.shape}")
print(f"\n📋 Samples per sign:")
print(df['label'].value_counts().to_string())