"""
SLSL Flask Server
-----------------
PC එකේ run කරන්න — MediaPipe + TFLite inference
Phone එකෙන් HTTP request එනවා → keypoints extract → sign predict → JSON response
Research: filter=true/false/both parameter වලින් Model A vs Model B compare කරන්න පුළුවන්
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import os
import socket
import threading

app = Flask(__name__)

# ================================================
# PATHS
# ================================================
MODEL_CANDIDATES = [
    r'C:\Users\Janith\Desktop\R26-IT-129\Janith\models\slsl_model.tflite',
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
# SIGN LABELS + SINHALA TRANSLATIONS
# ================================================
SIGN_LABELS = [
    'Allocate', 'Answer', 'Answer Properly', 'Answer Sheet', 'Ask Question',
    'Attend', 'Attending', 'Calculate', 'Cancel', 'Collaborating',
    'Collect', 'Comparing', 'Concentrate', 'Continuing', 'Coordinate',
    'Copying', 'Correct Mistake', 'Describe', 'Discuss', 'Discuss Topic',
    'Distribute', 'Documenting', 'Grade', 'Practice', 'Research',
    'Review', 'Study', 'Support', 'Teacher', 'Whiteboard Marker',
]

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
print(f"✅ Model loaded")
print(f"   Input shape : {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# ================================================
# MEDIAPIPE SETUP
# ================================================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
print("✅ MediaPipe ready")

# ── THREAD-SAFE LOCKS ────────────────────────────
# MediaPipe Hands සහ TFLite Interpreter දෙකම thread-safe නෑ.
# Flask threaded=True නිසා concurrent requests වලට ඒවා එකවර access වෙනවා.
# මේ locks දෙකෙන් එකකට කෙනෙක් ගන්න ඉඩ දෙනවා.
mediapipe_lock   = threading.Lock()
interpreter_lock = threading.Lock()

# ================================================
# NOISE FILTER — Research Contribution
# Novel velocity-threshold-based noise filtering
# ================================================
def apply_noise_filter(sequence, threshold=NOISE_THRESHOLD):
    """
    Model B (Proposed) — WITH noise filter
    Step 1: Zero frames ඉවත් කරනවා
    Step 2: Low-velocity frames ඉවත් කරනවා
    Step 3: Valid frames 30 frames වලට interpolate කරනවා
    """
    valid_frames = [f for f in sequence if np.sum(np.abs(f)) > 0.01]

    if len(valid_frames) == 0:
        return [[0.0] * 63] * SEQUENCE_LENGTH

    if len(valid_frames) >= 2:
        filtered = [valid_frames[0]]
        for i in range(1, len(valid_frames)):
            velocity = np.mean(np.abs(
                np.array(valid_frames[i]) - np.array(valid_frames[i - 1])
            ))
            if velocity > threshold:
                filtered.append(valid_frames[i])
        valid_frames = filtered if len(filtered) > 3 else valid_frames

    if len(valid_frames) >= SEQUENCE_LENGTH:
        return valid_frames[:SEQUENCE_LENGTH]
    else:
        indices = np.linspace(0, len(valid_frames) - 1, SEQUENCE_LENGTH)
        return [valid_frames[int(round(i))] for i in indices]


def no_filter(sequence):
    """
    Model A (Baseline) — WITHOUT noise filter
    Raw sequence pad/trim only
    """
    if len(sequence) < SEQUENCE_LENGTH:
        padding = [[0.0] * 63] * (SEQUENCE_LENGTH - len(sequence))
        return sequence + padding
    return sequence[:SEQUENCE_LENGTH]


# ================================================
# EXTRACT KEYPOINTS FROM IMAGE  (THREAD-SAFE)
# ================================================
def extract_keypoints(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Lock — only one thread can call MediaPipe at a time
    with mediapipe_lock:
        results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
    return keypoints


# ================================================
# RUN TFLITE INFERENCE  (THREAD-SAFE)
# ================================================
def run_inference(sequence):
    input_data = np.array([sequence], dtype=np.float32)

    # Lock — TFLite interpreter is not thread-safe
    with interpreter_lock:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0].copy()

    top_idx   = int(np.argmax(output))
    top_conf  = float(output[top_idx])
    top_label = SIGN_LABELS[top_idx] if top_idx < len(SIGN_LABELS) else 'Unknown'

    sorted_idx = np.argsort(output)[::-1][:3]
    top3 = [
        {
            'label'     : SIGN_LABELS[i] if i < len(SIGN_LABELS) else 'Unknown',
            'sinhala'   : SINHALA_TRANSLATIONS.get(
                SIGN_LABELS[i] if i < len(SIGN_LABELS) else '', ''),
            'confidence': float(output[i]),
        }
        for i in sorted_idx
    ]
    return top_label, top_conf, top3


# ================================================
# ROUTES
# ================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded', 'signs': len(SIGN_LABELS)})


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
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
            return jsonify({'hand_detected': False, 'keypoints': [0.0] * 63, 'frame_id': frame_id})

        return jsonify({'hand_detected': True, 'keypoints': keypoints, 'frame_id': frame_id})

    except Exception as e:
        print(f"❌ predict_frame error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    """
    ?filter=true  → Model B (WITH noise filter)    — Proposed
    ?filter=false → Model A (WITHOUT noise filter)  — Baseline
    ?filter=both  → දෙකම — Examiner comparison mode
    """
    try:
        data       = request.get_json()
        frames     = data.get('frames', [])
        use_filter = request.args.get('filter', 'true').lower()

        if len(frames) == 0:
            return jsonify({'error': 'No frames received'}), 400

        valid_count = sum(1 for f in frames if np.sum(np.abs(f)) > 0.01)
        print(f"📊 frames={len(frames)} | valid={valid_count} | filter={use_filter}")

        if valid_count < 5:
            no_hand = {
                'label': 'No hand detected',
                'sinhala': 'අත හොයාගත නොහැක — නැවත try කරන්න',
                'confidence': 0.0,
                'top3': [],
                'filtered': False,
                'valid_frames': valid_count,
            }
            if use_filter == 'both':
                return jsonify({'mode': 'comparison', 'valid_frames': valid_count,
                                'model_a': no_hand, 'model_b': no_hand})
            return jsonify(no_hand)

        # ── COMPARISON MODE (examiner demo) ──────────
        if use_filter == 'both':
            label_a, conf_a, top3_a = run_inference(no_filter(frames))
            label_b, conf_b, top3_b = run_inference(apply_noise_filter(frames))

            print(f"  🔴 Model A (baseline) : {label_a} ({conf_a*100:.1f}%)")
            print(f"  🟢 Model B (proposed) : {label_b} ({conf_b*100:.1f}%)")

            return jsonify({
                'mode'        : 'comparison',
                'valid_frames': valid_count,
                'model_a': {
                    'label'     : label_a,
                    'sinhala'   : SINHALA_TRANSLATIONS.get(label_a, label_a),
                    'confidence': conf_a,
                    'top3'      : top3_a,
                    'filtered'  : False,
                },
                'model_b': {
                    'label'     : label_b,
                    'sinhala'   : SINHALA_TRANSLATIONS.get(label_b, label_b),
                    'confidence': conf_b,
                    'top3'      : top3_b,
                    'filtered'  : True,
                },
            })

        # ── SINGLE MODE ───────────────────────────────
        if use_filter == 'false':
            sequence      = no_filter(frames)
            filtered_flag = False
        else:
            sequence      = apply_noise_filter(frames)
            filtered_flag = True

        label, confidence, top3 = run_inference(sequence)
        sinhala = SINHALA_TRANSLATIONS.get(label, label)
        print(f"✅ {'[B-Filtered]' if filtered_flag else '[A-NoFilter]'} {label} ({confidence*100:.1f}%)")

        return jsonify({
            'label'       : label,
            'sinhala'     : sinhala,
            'confidence'  : confidence,
            'top3'        : top3,
            'filtered'    : filtered_flag,
            'valid_frames': valid_count,
        })

    except Exception as e:
        print(f"❌ predict_sequence error: {e}")
        return jsonify({'error': str(e)}), 500


# ================================================
# MAIN
# ================================================
if __name__ == '__main__':
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "=" * 52)
    print("  SLSL Flask Server — Research Demo")
    print("=" * 52)
    print(f"  Signs       : {len(SIGN_LABELS)}")
    print(f"  Sequence    : {SEQUENCE_LENGTH} frames")
    print(f"  Filter      : velocity threshold {NOISE_THRESHOLD}")
    print(f"  Thread-safe : MediaPipe + TFLite locked")
    print("─" * 52)
    print(f"  Flutter     : http://{local_ip}:5000")
    print(f"  Health      : http://{local_ip}:5000/health")
    print("─" * 52)
    print("  Endpoints:")
    print("  ?filter=true  → Model B (proposed)")
    print("  ?filter=false → Model A (baseline)")
    print("  ?filter=both  → Comparison mode 🎓")
    print("=" * 52 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)