# Hansika/teacher_dashboard_server.py
"""
Teacher Dashboard & Content Authoring — Backend
------------------------------------------------
Endpoints:
  GET  /api/teacher/health
  GET  /api/vocabulary
  POST /api/teacher/validate-frame        { image (base64) }
  POST /api/teacher/submit-sign           { teacher_id, english_word, sinhala_word, category, frames }
  GET  /api/teacher/my-submissions/<teacher_id>
  POST /api/authority/approve/<submission_id>
  POST /api/authority/reject/<submission_id>   { reason }

Does NOT modify Janith's dataset, model, or server logic.
Reads his keypoints_clean.csv read-only to check for duplicate signs.
"""

from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
import base64
import os
import threading
import smtplib
from email.mime.text import MIMEText

teacher_bp = Blueprint("teacher_dashboard", __name__)

# ================================================
# DATABASE
# ================================================
client = MongoClient("mongodb://localhost:27017")
db = client["slsl_app"]
submissions = db["sign_submissions"]
vocabulary = db["sign_vocabulary"]

# ================================================
# JANITH'S DATASET — READ-ONLY, for duplicate checking
# ================================================
JANITH_CLEAN_CSV = os.path.join(
    os.path.dirname(__file__), '..', 'Janith', 'keypoints_clean.csv'
)

def get_existing_labels():
    """Reads Janith's cleaned dataset to get current sign labels. Read-only, never modifies it."""
    if not os.path.exists(JANITH_CLEAN_CSV):
        print(f"⚠️  Janith's clean CSV not found at {JANITH_CLEAN_CSV} — duplicate check skipped")
        return set()
    import pandas as pd
    df = pd.read_csv(JANITH_CLEAN_CSV, usecols=['label'])
    return set(df['label'].str.strip().str.lower())

# ================================================
# MEDIAPIPE — separate instances from Janith's, thread-safe
# ================================================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

teacher_hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)
teacher_pose_detector = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
)
teacher_mediapipe_lock = threading.Lock()

# ================================================
# VALIDATION THRESHOLDS — tune these after testing with real footage
# ================================================
MIN_BRIGHTNESS = 60           # grayscale mean, 0-255
MAX_BRIGHTNESS = 220
MAX_EDGE_DENSITY = 0.12       # background clutter — fraction of pixels that are edges
KNEE_VISIBILITY_LIMIT = 0.5   # if knees this visible → too much lower body shown
SHOULDER_VISIBILITY_MIN = 0.3 # shoulders must be at least this visible

BATCH_SIZE_FOR_AUTHORITY_EMAIL = 20

# ================================================
# VALIDATION CHECKS
# ================================================
def check_lighting(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < MIN_BRIGHTNESS:
        return False, "lighting", "Too dark. Please move to a brighter area."
    if brightness > MAX_BRIGHTNESS:
        return False, "lighting", "Too bright/overexposed. Reduce direct light on the camera."
    return True, None, None


def check_background(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size
    if edge_density > MAX_EDGE_DENSITY:
        return False, "background", "Background too cluttered. Please use a plain, clear background."
    return True, None, None


def check_body_framing(pose_results):
    if not pose_results.pose_landmarks:
        return False, "framing", "No person detected. Please position yourself in front of the camera."

    lm = pose_results.pose_landmarks.landmark
    # MediaPipe Pose indices: 11/12 = shoulders, 25/26 = knees
    knee_visibility = max(lm[25].visibility, lm[26].visibility)
    shoulder_visibility = max(lm[11].visibility, lm[12].visibility)

    if knee_visibility > KNEE_VISIBILITY_LIMIT:
        return False, "framing", "Too much of your body is visible. Move closer so only your upper body (head, shoulders, hands) shows."
    if shoulder_visibility < SHOULDER_VISIBILITY_MIN:
        return False, "framing", "Upper body not clearly visible. Please adjust your position."
    return True, None, None


def check_hand_visibility(hands_results):
    if not hands_results.multi_hand_landmarks:
        return False, "hand", "No hand detected. Make sure your hand is visible while signing."
    return True, None, None


def run_validation(image_bgr):
    """Runs all checks in order, returns first failure or success."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with teacher_mediapipe_lock:
        hands_results = teacher_hands_detector.process(rgb)
        pose_results = teacher_pose_detector.process(rgb)

    for check in (
        lambda: check_lighting(image_bgr),
        lambda: check_background(image_bgr),
        lambda: check_body_framing(pose_results),
        lambda: check_hand_visibility(hands_results),
    ):
        ok, reason, message = check()
        if not ok:
            return {"valid": False, "reason": reason, "message": message}

    return {"valid": True, "message": "Frame passed all validation checks."}

# ================================================
# EMAIL NOTIFICATION — batch alert to Authority
# ================================================
def send_batch_email_to_authority():
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    authority_email = os.environ.get("AUTHORITY_EMAIL")

    if not all([smtp_user, smtp_pass, authority_email]):
        print("⚠️  Email not configured — set SMTP_USER, SMTP_PASS, AUTHORITY_EMAIL env vars. Skipping email.")
        return False

    pending_docs = list(
        submissions.find({"status": "pending", "notified_authority": False})
        .limit(BATCH_SIZE_FOR_AUTHORITY_EMAIL)
    )
    if not pending_docs:
        return False

    lines = [
        f"- {d['english_word']} ({d['category']}) — submitted by {d['teacher_id']} — id: {d['_id']}"
        for d in pending_docs
    ]
    body = "New SLSL sign submissions awaiting review:\n\n" + "\n".join(lines)

    msg = MIMEText(body)
    msg["Subject"] = f"SLSL App: {len(pending_docs)} New Signs Awaiting Approval"
    msg["From"] = smtp_user
    msg["To"] = authority_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    except Exception as e:
        print(f"❌ Failed to send authority email: {e}")
        return False

    ids = [d["_id"] for d in pending_docs]
    submissions.update_many({"_id": {"$in": ids}}, {"$set": {"notified_authority": True}})
    print(f"✅ Authority notified about {len(pending_docs)} pending signs.")
    return True


def check_and_notify_authority():
    pending_count = submissions.count_documents({"status": "pending", "notified_authority": False})
    if pending_count >= BATCH_SIZE_FOR_AUTHORITY_EMAIL:
        send_batch_email_to_authority()

# ================================================
# ROUTES
# ================================================

@teacher_bp.route("/api/teacher/health", methods=["GET"])
def teacher_health():
    return jsonify({"status": "ok"})


@teacher_bp.route("/api/teacher/validate-frame", methods=["POST"])
def validate_frame():
    """
    Called by Flutter during capture to give real-time feedback.
    Request : { "image": "<base64 jpeg>" }
    Response: { "valid": bool, "reason": str|null, "message": str }
    """
    try:
        data = request.get_json()
        img_b64 = data.get("image", "")
        img_bytes = base64.b64decode(img_b64)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"valid": False, "reason": "invalid_image", "message": "Could not read image."}), 400

        result = run_validation(image)
        return jsonify(result)

    except Exception as e:
        print(f"❌ validate_frame error: {e}")
        return jsonify({"valid": False, "reason": "error", "message": str(e)}), 500


@teacher_bp.route("/api/teacher/submit-sign", methods=["POST"])
def submit_sign():
    """
    Request : { teacher_id, english_word, sinhala_word, category, frames: [[63 floats] x 30] }
    """
    try:
        data = request.json
        required_fields = ["teacher_id", "english_word", "sinhala_word", "category", "frames"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        if data["category"] not in ("noun", "verb"):
            return jsonify({"error": "category must be 'noun' or 'verb'"}), 400

        label = data["english_word"].strip()

        # ── Duplicate check against Janith's actual dataset ──
        existing_labels = get_existing_labels()
        if label.lower() in existing_labels:
            return jsonify({
                "status": "rejected",
                "reason": "duplicate",
                "message": f"'{label}' already exists in the sign dataset."
            }), 409

        # ── Duplicate check against already-approved teacher submissions ──
        if vocabulary.find_one({"english_word": {"$regex": f"^{label}$", "$options": "i"}}):
            return jsonify({
                "status": "rejected",
                "reason": "duplicate",
                "message": f"'{label}' has already been approved and added previously."
            }), 409

        doc = {
            "teacher_id": data["teacher_id"],
            "english_word": label,
            "sinhala_word": data["sinhala_word"],
            "category": data["category"],
            "keypoint_sequence": data["frames"],  # 30 x 63, matches CNN+LSTM input shape
            "status": "pending",
            "created_at": datetime.utcnow(),
            "notified_authority": False,
        }
        result = submissions.insert_one(doc)
        check_and_notify_authority()

        return jsonify({"submission_id": str(result.inserted_id), "status": "pending"}), 201

    except Exception as e:
        print(f"❌ submit_sign error: {e}")
        return jsonify({"error": str(e)}), 500


@teacher_bp.route("/api/teacher/my-submissions/<teacher_id>", methods=["GET"])
def my_submissions(teacher_id):
    docs = list(submissions.find({"teacher_id": teacher_id}, {"keypoint_sequence": 0}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return jsonify(docs)


@teacher_bp.route("/api/vocabulary", methods=["GET"])
def get_vocabulary():
    docs = list(vocabulary.find({}, {"_id": 0}))
    return jsonify(docs)


@teacher_bp.route("/api/authority/pending", methods=["GET"])
def list_pending():
    docs = list(submissions.find({"status": "pending"}, {"keypoint_sequence": 0}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return jsonify(docs)


@teacher_bp.route("/api/authority/approve/<submission_id>", methods=["POST"])
def approve_submission(submission_id):
    try:
        sub = submissions.find_one({"_id": ObjectId(submission_id)})
        if not sub:
            return jsonify({"error": "Submission not found"}), 404

        vocabulary.insert_one({
            "english_word": sub["english_word"],
            "sinhala_word": sub["sinhala_word"],
            "category": sub["category"],
            "keypoint_sequence": sub["keypoint_sequence"],
            "source": "teacher_submission",
            "approved_at": datetime.utcnow(),
        })
        submissions.update_one({"_id": sub["_id"]}, {"$set": {"status": "approved"}})
        return jsonify({"status": "approved"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@teacher_bp.route("/api/authority/reject/<submission_id>", methods=["POST"])
def reject_submission(submission_id):
    try:
        data = request.json or {}
        reason = data.get("reason", "Not specified")
        result = submissions.update_one(
            {"_id": ObjectId(submission_id)},
            {"$set": {"status": "rejected", "rejection_reason": reason}}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Submission not found"}), 404
        return jsonify({"status": "rejected"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500