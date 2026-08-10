# Hansika/teacher_dashboard_server.py
"""
Teacher Dashboard & Content Authoring — Backend
------------------------------------------------
Endpoints:
  GET    /api/teacher/health
  GET    /api/vocabulary
  POST   /api/teacher/validate-frame        { image (base64) }
  POST   /api/teacher/submit-sign           { teacher_id, teacher_email, english_word, sinhala_word, category, frames }
  GET    /api/teacher/my-submissions/<teacher_id>
  GET    /api/teacher/submission/<submission_id>          -- NEW: full detail incl. keypoints, for playback
  DELETE /api/teacher/delete-submission/<submission_id>    -- NEW
  GET    /api/teacher/pending-batch                        -- now also returns total_awaiting_decision
  POST   /api/teacher/send-to-authority     { authority_email }
  GET    /api/authority/pending
  POST   /api/authority/approve/<submission_id>   -- emails teacher back
  POST   /api/authority/reject/<submission_id>    { reason }   -- emails teacher back
  GET    /authority/review                        -- simple web page, Approve/Reject buttons

Does NOT modify Janith's dataset, model, or server logic.
Reads his keypoints_clean.csv read-only to check for duplicate signs.
"""

from flask import Blueprint, request, jsonify, render_template_string
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
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 220
MAX_EDGE_DENSITY = 0.12
KNEE_VISIBILITY_LIMIT = 0.5
SHOULDER_VISIBILITY_MIN = 0.3

BATCH_SIZE_FOR_AUTHORITY_EMAIL = 5

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
# EMAIL — generic sender, used both directions
# ================================================
def send_email(to_address, subject, body):
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")

    if not all([smtp_user, smtp_pass, to_address]):
        print(f"⚠️  Email not sent (missing SMTP_USER/SMTP_PASS or recipient). "
              f"Would have sent to: {to_address} | Subject: {subject}")
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_address

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"✅ Email sent to {to_address}: {subject}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {to_address}: {e}")
        return False

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
    Request : { teacher_id, teacher_email, english_word, sinhala_word, category, frames: [[63 floats] x 30] }
    """
    try:
        data = request.json
        required_fields = ["teacher_id", "teacher_email", "english_word", "sinhala_word", "category", "frames"]
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
            "teacher_email": data["teacher_email"],
            "english_word": label,
            "sinhala_word": data["sinhala_word"],
            "category": data["category"],
            "keypoint_sequence": data["frames"],
            "status": "pending",
            "created_at": datetime.utcnow(),
            "notified_authority": False,
        }
        result = submissions.insert_one(doc)

        pending_count = submissions.count_documents({"status": "pending", "notified_authority": False})
        batch_ready = pending_count >= BATCH_SIZE_FOR_AUTHORITY_EMAIL

        return jsonify({
            "submission_id": str(result.inserted_id),
            "status": "pending",
            "pending_batch_count": pending_count,
            "batch_ready": batch_ready,
        }), 201

    except Exception as e:
        print(f"❌ submit_sign error: {e}")
        return jsonify({"error": str(e)}), 500


@teacher_bp.route("/api/teacher/my-submissions/<teacher_id>", methods=["GET"])
def my_submissions(teacher_id):
    docs = list(submissions.find({"teacher_id": teacher_id}, {"keypoint_sequence": 0}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return jsonify(docs)


# ================================================
# NEW — Full submission detail (includes keypoint_sequence) for playback
# ================================================
@teacher_bp.route("/api/teacher/submission/<submission_id>", methods=["GET"])
def get_submission_detail(submission_id):
    try:
        sub = submissions.find_one({"_id": ObjectId(submission_id)})
        if not sub:
            return jsonify({"error": "Submission not found"}), 404
        sub["_id"] = str(sub["_id"])
        return jsonify(sub)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================
# NEW — Delete a submission
# ================================================
@teacher_bp.route("/api/teacher/delete-submission/<submission_id>", methods=["DELETE"])
def delete_submission(submission_id):
    try:
        result = submissions.delete_one({"_id": ObjectId(submission_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Submission not found"}), 404
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@teacher_bp.route("/api/vocabulary", methods=["GET"])
def get_vocabulary():
    docs = list(vocabulary.find({}, {"_id": 0}))
    return jsonify(docs)


@teacher_bp.route("/api/teacher/pending-batch", methods=["GET"])
def pending_batch():
    """
    Response: { count, ready, signs, total_awaiting_decision }
    - count: signs not yet emailed to authority
    - total_awaiting_decision: ALL pending signs regardless of email status
      (i.e. includes ones already emailed but not yet approved/rejected)
    """
    pending_docs = list(
        submissions.find(
            {"status": "pending", "notified_authority": False},
            {"english_word": 1, "category": 1, "teacher_id": 1}
        ).limit(BATCH_SIZE_FOR_AUTHORITY_EMAIL)
    )
    count = submissions.count_documents({"status": "pending", "notified_authority": False})
    total_awaiting_decision = submissions.count_documents({"status": "pending"})
    for d in pending_docs:
        d["_id"] = str(d["_id"])
    return jsonify({
        "count": count,
        "ready": count >= BATCH_SIZE_FOR_AUTHORITY_EMAIL,
        "signs": pending_docs,
        "total_awaiting_decision": total_awaiting_decision,
    })


@teacher_bp.route("/api/teacher/send-to-authority", methods=["POST"])
def send_to_authority():
    """
    Request : { "authority_email": "someone@example.com" }
    """
    try:
        data = request.json or {}
        authority_email = data.get("authority_email", "").strip()
        if not authority_email or "@" not in authority_email:
            return jsonify({"error": "Please provide a valid email address."}), 400

        pending_docs = list(
            submissions.find({"status": "pending", "notified_authority": False})
            .limit(BATCH_SIZE_FOR_AUTHORITY_EMAIL)
        )

        if not pending_docs:
            return jsonify({"error": "No pending signs to send."}), 400

        review_link = f"{request.host_url}authority/review"

        lines = [
            f"- {d['english_word']} ({d['category']}) — submitted by {d['teacher_id']}"
            for d in pending_docs
        ]
        body = (
            f"New SLSL sign submissions awaiting review ({len(pending_docs)} signs):\n\n"
            + "\n".join(lines)
            + f"\n\nReview and approve/reject them here:\n{review_link}"
        )
        subject = f"SLSL App: {len(pending_docs)} New Signs Awaiting Approval"

        sent = send_email(authority_email, subject, body)

        if sent:
            ids = [d["_id"] for d in pending_docs]
            submissions.update_many(
                {"_id": {"$in": ids}},
                {"$set": {"notified_authority": True, "sent_to_authority_email": authority_email,
                          "notified_at": datetime.utcnow()}}
            )

        return jsonify({
            "sent": sent,
            "count": len(pending_docs),
            "message": f"Sent {len(pending_docs)} signs to {authority_email}."
                       if sent else "Email delivery failed — check SMTP configuration. Your signs are still pending and ready to resend."
        })

    except Exception as e:
        print(f"❌ send_to_authority error: {e}")
        return jsonify({"error": str(e)}), 500


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

        teacher_email = sub.get("teacher_email")
        if teacher_email:
            send_email(
                teacher_email,
                f"Your sign '{sub['english_word']}' was approved ✅",
                f"Good news! Your submitted sign '{sub['english_word']}' "
                f"has been reviewed and approved. It has been added to the SLSL vocabulary."
            )

        return jsonify({"status": "approved"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@teacher_bp.route("/api/authority/reject/<submission_id>", methods=["POST"])
def reject_submission(submission_id):
    try:
        data = request.json or {}
        reason = data.get("reason", "Not specified")

        sub = submissions.find_one({"_id": ObjectId(submission_id)})
        if not sub:
            return jsonify({"error": "Submission not found"}), 404

        submissions.update_one(
            {"_id": sub["_id"]},
            {"$set": {"status": "rejected", "rejection_reason": reason}}
        )

        teacher_email = sub.get("teacher_email")
        if teacher_email:
            send_email(
                teacher_email,
                f"Your sign '{sub['english_word']}' was not approved",
                f"Your submitted sign '{sub['english_word']}' was reviewed and not approved.\n\n"
                f"Reason: {reason}\n\n"
                f"You're welcome to record and resubmit this sign."
            )

        return jsonify({"status": "rejected"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================
# Simple web page for the Authority to review signs
# No app install needed — just open this link in any browser.
# ================================================
AUTHORITY_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>SLSL — Authority Review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; background:#020818; color:#fff; margin:0; padding:24px; }
  h1 { font-size: 20px; margin-bottom: 20px; }
  .card { background:#023E8A33; border:1px solid #00B4D855; border-radius:14px; padding:18px; margin-bottom:14px; }
  .word { font-size:18px; font-weight:700; }
  .cat { font-weight:400; color:#ffffff77; font-size:13px; }
  .sinhala { color:#90E0EF; font-size:15px; margin:6px 0; }
  .meta { color:#ffffff66; font-size:12px; margin-bottom:14px; }
  button { padding:10px 18px; border:none; border-radius:8px; font-weight:700; margin-right:10px; cursor:pointer; font-size:14px; }
  .approve { background:#06D6A0; color:#000; }
  .reject { background:#EF233C; color:#fff; }
  .empty { color:#ffffff66; text-align:center; padding:60px 0; }
  .refresh { background:#00B4D8; color:#fff; margin-bottom:20px; }
</style>
</head>
<body>
<h1>Pending Sign Submissions ({{ count }})</h1>
<button class="refresh" onclick="location.reload()">Refresh</button>

{% if signs|length == 0 %}
  <div class="empty">No pending signs right now.</div>
{% endif %}

{% for s in signs %}
<div class="card" id="card-{{ s._id }}">
  <div class="word">{{ s.english_word }} <span class="cat">({{ s.category }})</span></div>
  <div class="sinhala">{{ s.sinhala_word }}</div>
  <div class="meta">Submitted by: {{ s.teacher_id }}</div>
  <button class="approve" onclick="act('{{ s._id }}','approve')">Approve</button>
  <button class="reject" onclick="act('{{ s._id }}','reject')">Reject</button>
</div>
{% endfor %}

<script>
async function act(id, action) {
  let reason = '';
  if (action === 'reject') {
    reason = prompt('Reason for rejection (optional):') || 'Not specified';
  }
  const url = action === 'approve'
    ? '/api/authority/approve/' + id
    : '/api/authority/reject/' + id;
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: action === 'reject' ? JSON.stringify({reason: reason}) : null,
  });
  if (res.ok) {
    document.getElementById('card-' + id).remove();
  } else {
    alert('Something went wrong. Please try again.');
  }
}
</script>
</body>
</html>
"""

@teacher_bp.route("/authority/review", methods=["GET"])
def authority_review_page():
    docs = list(submissions.find({"status": "pending"}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return render_template_string(AUTHORITY_PAGE_TEMPLATE, signs=docs, count=len(docs))