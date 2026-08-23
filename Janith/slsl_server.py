"""
SLSL Flask Server — Final Version
----------------------------------
Matches camera_screen_janith.dart exactly:
  - /health              GET
  - /predict_frame       POST  { image, frame_id }
  - /predict_sequence    POST  { frames } ?filter=true|false|both

Threading lock for MediaPipe (parallel batch requests safe).
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import os
import sys          # ADDED — needed for path setup below
import threading

app = Flask(__name__)

# ================================================
# ADDED — Register Teacher Dashboard Blueprint (Hansika's module)
# ================================================
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Hansika'))
from teacher_dashboard_server import teacher_bp
app.register_blueprint(teacher_bp)

# ================================================
# ADDED — Register Adaptive Lesson System Blueprint (Kisal's module)
# ================================================


# ================================================
# PATHS
# ================================================
MODEL_CANDIDATES = [
    r'D:\R26-IT-129\Janith\models\slsl_model.tflite',
    os.path.join(os.path.dirname(__file__), 'models', 'slsl_model.tflite'),
    os.path.join(os.getcwd(), 'models', 'slsl_model.tflite'),
]

MODEL_PATH = None
for _p in MODEL_CANDIDATES:
    if os.path.exists(_p):
        MODEL_PATH = _p
        break

if MODEL_PATH is None:
    raise FileNotFoundError(f"TFLite model not found. Checked: {MODEL_CANDIDATES}")

# ================================================
# SIGN LABELS — loaded from the trained model itself
# CHANGED: previously hardcoded. Now loaded from classes.npy (saved by
# train_model.py's LabelEncoder), so if retraining ever changes which
# 30 signs make the cut, or their order, the server automatically
# stays in sync instead of silently mislabeling predictions.
# ================================================
_CLASSES_PATH = os.path.join(os.path.dirname(MODEL_PATH), 'classes.npy')
if os.path.exists(_CLASSES_PATH):
    SIGN_LABELS = np.load(_CLASSES_PATH, allow_pickle=True).tolist()
else:
    # Fallback if classes.npy hasn't been generated yet — matches the
    # 30 classroom signs as of the last known training run.
    SIGN_LABELS = [
        'Allocate', 'Answer', 'Answer Properly', 'Answer Sheet', 'Ask Question',
        'Attend', 'Attending', 'Calculate', 'Cancel', 'Collaborating',
        'Collect', 'Comparing', 'Concentrate', 'Continuing', 'Coordinate',
        'Copying', 'Correct Mistake', 'Describe', 'Discuss', 'Discuss Topic',
        'Distribute', 'Documenting', 'Grade', 'Practice', 'Research',
        'Review', 'Study', 'Support', 'Teacher', 'Whiteboard Marker',
    ]
    print(f"⚠️  classes.npy not found at {_CLASSES_PATH} — using fallback SIGN_LABELS")

# ================================================
# SINHALA TRANSLATIONS
# ================================================
SINHALA_TRANSLATIONS = {
    'Allocate'         : 'බෙදා හැරීම',
    'Answer'           : 'පිළිතුර',
    'Answer Properly'  : 'නිසි ලෙස පිළිතුරු දෙන්න',
    'Answer Sheet'     : 'පිළිතුරු පත්‍රය',
    'Ask Question'     : 'ප්‍රශ්නය අසන්න',
    'Attend'           : 'සහභාගී වෙන්න',
    'Attending'        : 'සහභාගී වෙමින්',
    'Calculate'        : 'ගණනය කරන්න',
    'Cancel'           : 'අවලංගු කරන්න',
    'Collaborating'    : 'සහයෝගයෙන් කටයුතු කිරීම',
    'Collect'          : 'එකතු කරන්න',
    'Comparing'        : 'සංසන්දනය කිරීම',
    'Concentrate'      : 'අවධානය යොමු කරන්න',
    'Continuing'       : 'දිගටම කරගෙන යන්න',
    'Coordinate'       : 'සම්බන්ධීකරණය කරන්න',
    'Copying'          : 'පිටපත් කිරීම',
    'Correct Mistake'  : 'වැරදි නිවැරදි කරන්න',
    'Describe'         : 'විස්තර කරන්න',
    'Discuss'          : 'සාකච්ඡා කරන්න',
    'Discuss Topic'    : 'මාතෘකාව සාකච්ඡා කරන්න',
    'Distribute'       : 'බෙදා දෙන්න',
    'Documenting'      : 'ලේඛනගත කිරීම',
    'Grade'            : 'ශ්‍රේණිය',
    'Practice'         : 'පුහුණු වෙන්න',
    'Research'         : 'පර්යේෂණය',
    'Review'           : 'සමාලෝචනය',
    'Study'            : 'අධ්‍යයනය කරන්න',
    'Support'          : 'සහාය දෙන්න',
    'Teacher'          : 'ගුරුවරයා',
    'Whiteboard Marker': 'වයිට්බෝඩ් මාකර්',
}

SEQUENCE_LENGTH = 30
NOISE_THRESHOLD = 0.02

# ================================================
# LOAD TFLITE MODEL
# ================================================
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"✅ Model loaded: {MODEL_PATH}")
print(f"   Input : {input_details[0]['shape']}")
print(f"   Output: {output_details[0]['shape']}")

# ================================================
# MEDIAPIPE — Thread-safe with Lock
# ================================================
mp_hands       = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mediapipe_lock = threading.Lock()   # ← thread-safe for parallel requests
inference_lock = threading.Lock()   # ← thread-safe TFLite
print("✅ MediaPipe ready (thread-safe)")

# ================================================
# NOISE FILTER — Research Contribution
# FIXED: now matches extract_keypoints.py exactly (the function that
# produced the training data), so inference-time input has the SAME
# shape the model was trained on: CAPTURE_FRAMES (15) "active" frames
# followed by zero-padding up to SEQUENCE_LENGTH (30).
# Previous version interpolated valid frames across all 30 slots,
# which never matched the 15-real+15-zero training format — that
# mismatch is fixed here.
# ================================================
CAPTURE_FRAMES = 15  # must match extract_keypoints.py / training data shape

def apply_noise_filter(sequence, threshold=NOISE_THRESHOLD):
    """
    Step 1: Velocity-filter the raw sequence (removes low-motion /
             accidental-movement frames), same pairwise comparison
             used when the training CSV was built.
    Step 2: Remove near-zero (no hand detected) frames from what's left.
    Step 3: Sample down to CAPTURE_FRAMES valid frames (or upsample if
             fewer than that), then zero-pad to SEQUENCE_LENGTH.
    Returns (result_sequence, frames_removed) — frames_removed is how
    many of the original frames were dropped as noise (low-velocity or
    no-hand), used to show a live "filter cleaned N frames" diagnostic
    in the app. This is a per-capture diagnostic, NOT the research
    false-positive rate (that requires ground truth and is measured
    offline in noise_filter_experiment.py).
    """
    # Step 1: velocity filter on the raw sequence
    if len(sequence) < 2:
        velocity_filtered = list(sequence)
    else:
        velocity_filtered = [sequence[0]]
        for i in range(1, len(sequence)):
            velocity = np.mean(np.abs(
                np.array(sequence[i]) - np.array(sequence[i - 1])
            ))
            if velocity > threshold:
                velocity_filtered.append(sequence[i])

    # Step 2: remove near-zero (no-hand) frames
    valid = [f for f in velocity_filtered if np.sum(np.abs(f)) > 0.01]
    frames_removed = len(sequence) - len(valid)

    if len(valid) == 0:
        return [[0.0] * 63] * SEQUENCE_LENGTH, frames_removed

    # Step 3: sample to CAPTURE_FRAMES, then zero-pad to SEQUENCE_LENGTH
    if len(valid) >= CAPTURE_FRAMES:
        indices = np.linspace(0, len(valid) - 1, CAPTURE_FRAMES, dtype=int)
        sampled = [valid[i] for i in indices]
    else:
        indices = np.linspace(0, len(valid) - 1, CAPTURE_FRAMES)
        sampled = [valid[int(round(i))] for i in indices]

    padding = [[0.0] * 63] * (SEQUENCE_LENGTH - CAPTURE_FRAMES)
    return sampled + padding, frames_removed

# ================================================
# NORMALIZATION — Wrist-centered, scale-invariant
# ADDED: must match extract_keypoints.py exactly, since it's applied
# to the training data. If this function and the one in
# extract_keypoints.py ever drift apart, training and serving will
# see different coordinate systems again (same class of bug as the
# noise-filter shape mismatch fixed earlier).
# ================================================
def normalize_keypoints(kp):
    """
    kp: flat list of 63 floats (21 MediaPipe hand landmarks x,y,z).
    Step 1: translate so the wrist (landmark 0) becomes the origin.
    Step 2: scale by the wrist→middle-finger-MCP (landmark 9) distance,
             so hand size in frame no longer matters.
    """
    arr = np.array(kp, dtype=np.float32).reshape(21, 3)
    if np.sum(np.abs(arr)) < 0.01:
        return kp  # no-hand / zero frame — leave as-is

    wrist = arr[0].copy()
    arr = arr - wrist  # translate: wrist -> origin

    scale = np.linalg.norm(arr[9])  # distance to middle finger MCP
    if scale < 1e-6:
        scale = 1.0
    arr = arr / scale

    return arr.flatten().tolist()

# ================================================
# EXTRACT KEYPOINTS
# ================================================
def extract_keypoints(image_bgr):
    """Thread-safe MediaPipe keypoint extraction."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mediapipe_lock:
        results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    kp = []
    for lm in results.multi_hand_landmarks[0].landmark:
        kp.extend([lm.x, lm.y, lm.z])
    return normalize_keypoints(kp)  # 63 floats, wrist-centered + scaled

# ================================================
# RUN INFERENCE
# ================================================
def run_inference(sequence):
    """Thread-safe TFLite inference."""
    input_data = np.array([sequence], dtype=np.float32)

    with inference_lock:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

    top_idx   = int(np.argmax(output))
    top_conf  = float(output[top_idx])
    top_label = SIGN_LABELS[top_idx] if top_idx < len(SIGN_LABELS) else 'Unknown'

    sorted_idx = np.argsort(output)[::-1][:3]
    top3 = [{
        'label'     : SIGN_LABELS[i] if i < len(SIGN_LABELS) else 'Unknown',
        'sinhala'   : SINHALA_TRANSLATIONS.get(
            SIGN_LABELS[i] if i < len(SIGN_LABELS) else '', ''),
        'confidence': float(output[i]),
    } for i in sorted_idx]

    return top_label, top_conf, top3

# ================================================
# ROUTES
# ================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model' : 'loaded',
        'signs' : len(SIGN_LABELS),
    })


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Flutter sends single JPEG frame.
    Request : { "image": "<base64>", "frame_id": 0 }
    Response: { "keypoints": [63 floats], "hand_detected": bool }
    """
    try:
        data     = request.get_json()
        img_b64  = data.get('image', '')
        frame_id = data.get('frame_id', 0)

        img_bytes = base64.b64decode(img_b64)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        image     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image'}), 400

        keypoints = extract_keypoints(image)

        if keypoints is None:
            return jsonify({
                'hand_detected': False,
                'keypoints'    : [0.0] * 63,
                'frame_id'     : frame_id,
            })

        return jsonify({
            'hand_detected': True,
            'keypoints'    : keypoints,
            'frame_id'     : frame_id,
        })

    except Exception as e:
        print(f"❌ predict_frame error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    """
    Flutter sends 30 keypoint frames.
    Request : { "frames": [[63 floats] x 30] }
    Query   : ?filter=true|false|both
    Response: { label, sinhala, confidence, top3, filtered, valid_frames }
              ?filter=both → { model_a: {...}, model_b: {...}, valid_frames }
    """
    try:
        data        = request.get_json()
        frames      = data.get('frames', [])
        filter_mode = request.args.get('filter', 'true')  # true|false|both

        if len(frames) == 0:
            return jsonify({'error': 'No frames received'}), 400

        valid_count = sum(1 for f in frames if np.sum(np.abs(f)) > 0.01)
        print(f"📊 frames={len(frames)} | valid={valid_count} | filter={filter_mode}")

        # No hand detected
        if valid_count < 5:
            no_hand = {
                'label'       : 'No hand detected',
                'sinhala'     : 'අත හොයාගත නොහැක — නැවත try කරන්න',
                'confidence'  : 0.0,
                'top3'        : [],
                'filtered'    : False,
                'valid_frames': valid_count,
            }
            if filter_mode == 'both':
                return jsonify({'model_a': no_hand, 'model_b': no_hand,
                                'valid_frames': valid_count})
            return jsonify(no_hand)

        # ── Model A (baseline — no filter) ──────────────
        def predict_no_filter():
            label, conf, top3 = run_inference(frames)
            sinhala = SINHALA_TRANSLATIONS.get(label, label)
            return {
                'label'       : label,
                'sinhala'     : sinhala,
                'confidence'  : conf,
                'top3'        : top3,
                'filtered'    : False,
                'valid_frames': valid_count,
            }

        # ── Model B (proposed — with filter) ────────────
        def predict_with_filter():
            filtered_frames, frames_removed = apply_noise_filter(frames)
            label, conf, top3 = run_inference(filtered_frames)
            sinhala = SINHALA_TRANSLATIONS.get(label, label)
            return {
                'label'          : label,
                'sinhala'        : sinhala,
                'confidence'     : conf,
                'top3'           : top3,
                'filtered'       : True,
                'valid_frames'   : valid_count,
                'frames_removed' : frames_removed,  # live diagnostic: how many
                                                     # frames the filter dropped
                                                     # as noise on THIS capture
            }

        # ── Return based on filter mode ──────────────────
        if filter_mode == 'false':
            result = predict_no_filter()
            print(f"✅ [A-No Filter] {result['label']} ({result['confidence']*100:.1f}%)")
            return jsonify(result)

        elif filter_mode == 'both':
            result_a = predict_no_filter()
            result_b = predict_with_filter()
            print(f"⚖️  A: {result_a['label']} ({result_a['confidence']*100:.1f}%) "
                  f"| B: {result_b['label']} ({result_b['confidence']*100:.1f}%)")
            return jsonify({
                'model_a'     : result_a,
                'model_b'     : result_b,
                'valid_frames': valid_count,
                'agreement'   : result_a['label'] == result_b['label'],
            })

        else:  # filter='true' (default — Model B)
            result = predict_with_filter()
            print(f"✅ [B-Filtered] {result['label']} ({result['confidence']*100:.1f}%)")
            return jsonify(result)

    except Exception as e:
        print(f"❌ predict_sequence error: {e}")
        return jsonify({'error': str(e)}), 500


# ================================================
# MAIN
# ================================================
if __name__ == '__main__':
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "=" * 50)
    print("  SLSL Flask Server — Final")
    print("=" * 50)
    print(f"  Signs    : {len(SIGN_LABELS)}")
    print(f"  Sequence : {SEQUENCE_LENGTH} frames")
    print(f"  Filter   : velocity threshold {NOISE_THRESHOLD}")
    print(f"  Threads  : MediaPipe + TFLite locks")
    print("=" * 50)
    print(f"\n  Flutter kServerUrl:")
    print(f"  http://{local_ip}:5000")
    print(f"\n  Health: http://{local_ip}:5000/health")
    print(f"  Adaptive quiz: http://{local_ip}:5000/adaptive/quiz/next")
    print(f"  Adaptive dashboard: http://{local_ip}:5000/adaptive/dashboard/analytics")
    print("=" * 50 + "\n")

    app.run(
        host    ='0.0.0.0',
        port    =5000,
        debug   =False,
        threaded=True,
    )