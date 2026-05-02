"""
SLSL Flask Server
-----------------
PC එකේ run කරන්න — MediaPipe + TFLite inference
Phone එකෙන් HTTP request එනවා → keypoints extract → sign predict → JSON response
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import os

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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
print("✅ MediaPipe ready")

# ================================================
# NOISE FILTER (IMPROVED)
# ================================================
def apply_noise_filter(sequence, threshold=NOISE_THRESHOLD):
    """
    Step 1: Zero frames (hand detect නොවුණ) ඉවත් කරනවා
    Step 2: Low-velocity frames (accidental movements) ඉවත් කරනවා
    Step 3: Valid frames 30 frames වලට interpolate කරනවා
    """
    # Step 1: Zero frames ඉවත් කරන්න
    valid_frames = [
        f for f in sequence
        if np.sum(np.abs(f)) > 0.01
    ]

    if len(valid_frames) == 0:
        return [[0.0] * 63] * SEQUENCE_LENGTH

    # Step 2: Velocity filter (low-velocity frames ඉවත් කරන්න)
    if len(valid_frames) >= 2:
        filtered = [valid_frames[0]]
        for i in range(1, len(valid_frames)):
            velocity = np.mean(np.abs(
                np.array(valid_frames[i]) - np.array(valid_frames[i - 1])
            ))
            if velocity > threshold:
                filtered.append(valid_frames[i])
        # Filter කළාට පස්සේ ඉතිරිය 3 frames වලට වඩා අඩු නම් original valid frames use කරන්න
        valid_frames = filtered if len(filtered) > 3 else valid_frames

    # Step 3: 30 frames වලට interpolate කරන්න
    if len(valid_frames) >= SEQUENCE_LENGTH:
        return valid_frames[:SEQUENCE_LENGTH]
    else:
        indices = np.linspace(0, len(valid_frames) - 1, SEQUENCE_LENGTH)
        return [valid_frames[int(round(i))] for i in indices]

# ================================================
# EXTRACT KEYPOINTS FROM IMAGE
# ================================================
def extract_keypoints(image_bgr):
    """
    MediaPipe hand keypoints 63ක් extract කරනවා.
    Returns list of 63 floats or None if no hand detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results   = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])

    return keypoints  # 21 landmarks × 3 = 63 values

# ================================================
# RUN TFLITE INFERENCE
# ================================================
def run_inference(sequence):
    """
    TFLite model එකෙන් prediction ගන්නවා.
    sequence: list of 30 lists, each with 63 floats
    Returns (label, confidence, top3)
    """
    input_data = np.array([sequence], dtype=np.float32)  # shape: (1, 30, 63)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (num_classes,)

    # Top result
    top_idx   = int(np.argmax(output))
    top_conf  = float(output[top_idx])
    top_label = SIGN_LABELS[top_idx] if top_idx < len(SIGN_LABELS) else 'Unknown'

    # Top 3
    sorted_idx = np.argsort(output)[::-1][:3]
    top3 = [
        {
            'label'      : SIGN_LABELS[i] if i < len(SIGN_LABELS) else 'Unknown',
            'sinhala'    : SINHALA_TRANSLATIONS.get(
                SIGN_LABELS[i] if i < len(SIGN_LABELS) else '', ''),
            'confidence' : float(output[i]),
        }
        for i in sorted_idx
    ]

    return top_label, top_conf, top3

# ================================================
# ROUTES
# ================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check — Flutter app connect වුණාම test කරනවා"""
    return jsonify({
        'status': 'ok',
        'model' : 'loaded',
        'signs' : len(SIGN_LABELS),
    })


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Flutter app single frame (base64 image) send කරනවා.
    Request : { "image": "<base64>", "frame_id": 0-29 }
    Response: { "keypoints": [63 floats], "hand_detected": bool }
    """
    try:
        data     = request.get_json()
        img_b64  = data.get('image', '')
        frame_id = data.get('frame_id', 0)

        # Decode base64 image
        img_bytes = base64.b64decode(img_b64)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        image     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Extract keypoints
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
    Flutter app 30 frames එකවර send කරනවා.
    Request : { "frames": [ [63 floats], ... ] }  (30 frames)
    Response: { "label", "sinhala", "confidence", "top3", "valid_frames" }
    """
    try:
        data   = request.get_json()
        frames = data.get('frames', [])

        if len(frames) == 0:
            return jsonify({'error': 'No frames received'}), 400

        # Valid (non-zero) frames count කරන්න
        valid_count = sum(
            1 for f in frames if np.sum(np.abs(f)) > 0.01
        )

        print(f"📊 Received {len(frames)} frames, {valid_count} valid (hand detected)")

        # Valid frames ඉතාම අඩු නම් reject කරන්න
        if valid_count < 5:
            return jsonify({
                'label'       : 'No hand detected',
                'sinhala'     : 'අත හොයාගත නොහැක — නැවත try කරන්න',
                'confidence'  : 0.0,
                'top3'        : [],
                'filtered'    : False,
                'valid_frames': valid_count,
            })

        # Improved noise filter apply කරන්න
        filtered = apply_noise_filter(frames)

        # Run inference
        label, confidence, top3 = run_inference(filtered)
        sinhala = SINHALA_TRANSLATIONS.get(label, label)

        print(f"✅ Prediction: {label} ({confidence*100:.1f}%) | Valid frames: {valid_count}/30")

        return jsonify({
            'label'       : label,
            'sinhala'     : sinhala,
            'confidence'  : confidence,
            'top3'        : top3,
            'filtered'    : True,
            'valid_frames': valid_count,
        })

    except Exception as e:
        print(f"❌ predict_sequence error: {e}")
        return jsonify({'error': str(e)}), 500


# ================================================
# MAIN
# ================================================
if __name__ == '__main__':
    # Get local IP for display
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "=" * 50)
    print("  SLSL Flask Server")
    print("=" * 50)
    print(f"  Signs    : {len(SIGN_LABELS)}")
    print(f"  Sequence : {SEQUENCE_LENGTH} frames")
    print(f"  Filter   : velocity threshold {NOISE_THRESHOLD}")
    print("=" * 50)
    print(f"\n  Flutter app එකේ SERVER_IP මේකට set කරන්න:")
    print(f"  http://{local_ip}:5000")
    print(f"\n  Health check:")
    print(f"  http://{local_ip}:5000/health")
    print("=" * 50 + "\n")

    app.run(
        host    ='0.0.0.0',
        port    =5000,
        debug   =False,
        threaded=True,
    )