"""
collect_more.py
---------------
Sign එකකට videos 10ක් quickly record කරන්න.
Run කරලා screen instructions follow කරන්න.
"""

import cv2
import os
import time

# ================================================
# CONFIG — මේ values change කරන්න
# ================================================
DATASET_PATH = r'D:\R26-IT-129\Janith\dataset'

# Sign කියන්නේ record කරන sign name
# File name = sign name (exactly SIGN_LABELS list එකේ tiyena vidihata)
SIGN_NAME = "Teacher"       # ← Change this for each sign
CATEGORY  = "Nouns"         # ← "Nouns" or "Verbs"
PERSON    = "person1"       # ← "person1", "person2", "person3"

VIDEOS_PER_SIGN = 10        # ← videos count
DURATION_SEC    = 3         # ← seconds per video
FPS             = 30

# ================================================
# SIGNS LIST — ඔක්කොම 30 signs
# ================================================
ALL_SIGNS = [
    # Nouns
    'Allocate', 'Answer Sheet', 'Attend', 'Attending',
    'Collaborating', 'Teacher', 'Whiteboard Marker',
    # Verbs
    'Answer', 'Answer Properly', 'Ask Question',
    'Calculate', 'Cancel', 'Collect', 'Comparing',
    'Concentrate', 'Continuing', 'Coordinate', 'Copying',
    'Correct Mistake', 'Describe', 'Discuss', 'Discuss Topic',
    'Distribute', 'Documenting', 'Grade', 'Practice',
    'Research', 'Review', 'Study', 'Support',
]

# ================================================
# SETUP
# ================================================
save_path = os.path.join(DATASET_PATH, CATEGORY, PERSON)
os.makedirs(save_path, exist_ok=True)

# Count existing videos for this sign
existing = [f for f in os.listdir(save_path)
            if f.lower().startswith(SIGN_NAME.lower().replace(' ', '_'))
            and f.endswith('.mp4')]
start_idx = len(existing)

print("=" * 55)
print(f"  SLSL Data Collection")
print("=" * 55)
print(f"  Sign     : {SIGN_NAME}")
print(f"  Category : {CATEGORY}")
print(f"  Person   : {PERSON}")
print(f"  Save to  : {save_path}")
print(f"  Existing : {start_idx} videos")
print(f"  To record: {VIDEOS_PER_SIGN} new videos")
print(f"  Duration : {DURATION_SEC}s each")
print("=" * 55)
print("\n  Instructions:")
print("  1. Camera window opens")
print("  2. Press SPACE to start recording")
print("  3. Do the sign and hold for 3 seconds")
print("  4. Recording stops automatically")
print("  5. Press Q to quit early")
print("=" * 55)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

total_frames = DURATION_SEC * FPS
recorded     = 0

for i in range(VIDEOS_PER_SIGN):
    idx       = start_idx + i
    filename  = f"{SIGN_NAME.replace(' ', '_')}_{idx}.mp4"
    filepath  = os.path.join(save_path, filename)
    recording = False
    out       = None
    frame_count = 0

    print(f"\n📹 Video {i+1}/{VIDEOS_PER_SIGN} — {filename}")
    print(f"   Press SPACE to start | Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if not recording:
            # Waiting screen
            cv2.rectangle(display, (0, 0), (640, 480),
                          (50, 50, 50), -1)
            ret2, frame2 = cap.read()
            if ret2:
                display = frame2.copy()

            cv2.putText(display,
                        f"Sign: {SIGN_NAME}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2)
            cv2.putText(display,
                        f"Video {i+1}/{VIDEOS_PER_SIGN}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(display,
                        "SPACE = Start  |  Q = Quit",
                        (20, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)

        else:
            # Recording screen
            progress = frame_count / total_frames
            bar_w    = int(600 * progress)
            remaining = DURATION_SEC - (frame_count / FPS)

            # Progress bar
            cv2.rectangle(display, (20, 450), (620, 470),
                          (50, 50, 50), -1)
            cv2.rectangle(display, (20, 450), (20 + bar_w, 470),
                          (0, 200, 0), -1)

            # Recording indicator
            cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(display,
                        f"RECORDING  {remaining:.1f}s",
                        (55, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            cv2.putText(display,
                        f"Sign: {SIGN_NAME}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.putText(display,
                        f"Frame {frame_count}/{total_frames}",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

        cv2.imshow('SLSL Data Collection', display)
        key = cv2.waitKey(1) & 0xFF

        if not recording:
            if key == ord(' '):
                # Start recording
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out    = cv2.VideoWriter(filepath, fourcc, FPS, (640, 480))
                recording   = True
                frame_count = 0
                print(f"   🔴 Recording...")

            elif key == ord('q') or key == 27:
                print("\n⏸  Quit by user.")
                cap.release()
                cv2.destroyAllWindows()
                print(f"\n✅ Recorded {recorded} videos for '{SIGN_NAME}'")
                exit(0)

        else:
            # Write frame
            out.write(frame)
            frame_count += 1

            if frame_count >= total_frames:
                # Done
                out.release()
                recording = False
                recorded += 1
                print(f"   ✅ Saved: {filename}")
                time.sleep(0.5)
                break

            elif key == ord('q') or key == 27:
                out.release()
                os.remove(filepath)
                print(f"   ⚠️  Cancelled video {i+1}")
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

# ================================================
# DONE
# ================================================
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 55)
print(f"✅ Done! {recorded} new videos recorded for '{SIGN_NAME}'")
print(f"   Saved to: {save_path}")
print(f"   Total videos now: {start_idx + recorded}")
print("=" * 55)
print("\n📌 Next sign? Edit SIGN_NAME in this script and run again.")
print("📌 All signs done? Run:")
print("   python training/extract_keypoints.py")
print("   python training/clean_data.py")
print("   python training/train_model.py")