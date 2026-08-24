# Hansika/teacher_dashboard_server.py
"""
Teacher Dashboard & Content Authoring — Backend
------------------------------------------------
NEW: DTW-based motion-duplicate detection — compares a new sign's actual
hand-motion sequence against every approved sign's motion, catching
duplicates even when the submitted word/label is completely different.
This is separate from the simple label-matching duplicate check.
"""

from flask import Blueprint, request, jsonify, render_template_string
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from fastdtw import fastdtw
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
# JANITH'S DATASET — READ-ONLY
# ================================================
JANITH_CLEAN_CSV = os.path.join(
    os.path.dirname(__file__), '..', 'Janith', 'keypoints_clean.csv'
)
JANITH_CLASSES_NPY = os.path.join(
    os.path.dirname(__file__), '..', 'Janith', 'models', 'classes.npy'
)

FALLBACK_CLASSROOM_SIGNS = [
    'Allocate', 'Answer', 'Answer Properly', 'Answer Sheet', 'Ask Question',
    'Attend', 'Attending', 'Calculate', 'Cancel', 'Collaborating',
    'Collect', 'Comparing', 'Concentrate', 'Continuing', 'Coordinate',
    'Copying', 'Correct Mistake', 'Describe', 'Discuss', 'Discuss Topic',
    'Distribute', 'Documenting', 'Grade', 'Practice', 'Research',
    'Review', 'Study', 'Support', 'Teacher', 'Whiteboard Marker',
]


def get_existing_labels():
    if not os.path.exists(JANITH_CLEAN_CSV):
        print(f"⚠️  Janith's clean CSV not found — duplicate check skipped")
        return set()
    import pandas as pd
    df = pd.read_csv(JANITH_CLEAN_CSV, usecols=['label'])
    return set(df['label'].str.strip().str.lower())

# ================================================
# MEDIAPIPE
# ================================================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

teacher_hands_detector = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5,
)
teacher_pose_detector = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5,
)
teacher_mediapipe_lock = threading.Lock()

# ================================================
# VALIDATION THRESHOLDS
# ================================================
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 220
MAX_EDGE_DENSITY = 0.12
KNEE_VISIBILITY_LIMIT = 0.5
SHOULDER_VISIBILITY_MIN = 0.3

# ================================================
# NEW — DTW MOTION-DUPLICATE DETECTION
# ================================================
# This distance threshold needs calibration (run calibrate_dtw_threshold.py
# and look at real distances between known-same vs known-different signs
# in your own dataset before trusting this default value).
DTW_DUPLICATE_THRESHOLD = 0.25


def compute_dtw_distance(seq_a, seq_b):
    """
    Compares two hand-motion sequences (each a list of 30 frames x 63 keypoints)
    using Dynamic Time Warping. Returns a distance normalized by sequence length
    so results are comparable regardless of how fast/slow each was recorded.
    Lower distance = more similar motion.
    """
    a = np.array(seq_a, dtype=float)
    b = np.array(seq_b, dtype=float)
    distance, _ = fastdtw(a, b, dist=lambda x, y: float(np.linalg.norm(x - y)))
    return distance / max(len(a), len(b))


def check_motion_duplicate(new_sequence, exclude_label=None):
    """
    Compares a new submission's motion against every already-approved sign's
    stored motion. Returns a list of {label, distance} for any match under
    the threshold, sorted by closest match first. Catches duplicates even
    when the submitted word is completely different from the existing sign.
    """
    matches = []
    approved_signs = vocabulary.find({}, {"english_word": 1, "keypoint_sequence": 1})

    for doc in approved_signs:
        if exclude_label and doc.get("english_word", "").lower() == exclude_label.lower():
            continue
        existing_seq = doc.get("keypoint_sequence")
        if not existing_seq:
            continue
        try:
            dist = compute_dtw_distance(new_sequence, existing_seq)
        except Exception as e:
            print(f"⚠️  DTW compare failed for '{doc.get('english_word')}': {e}")
            continue
        if dist <= DTW_DUPLICATE_THRESHOLD:
            matches.append({"label": doc["english_word"], "distance": round(dist, 4)})

    matches.sort(key=lambda m: m["distance"])
    return matches

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
# EMAIL
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
    Request : { teacher_id, teacher_email, english_word, sinhala_word, category, frames }

    NEW: after label-based duplicate checks pass, the submission's motion is
    compared against every approved sign using DTW. If a close motion match
    is found, the submission is still saved (not blocked), but flagged with
    motion_duplicate_warning=True and similar_signs=[...] so the Authority
    can inspect it closely before approving.
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
        frames = data["frames"]

        # ── Label-based duplicate check against Janith's dataset ──
        existing_labels = get_existing_labels()
        if label.lower() in existing_labels:
            return jsonify({
                "status": "rejected",
                "reason": "duplicate",
                "message": f"'{label}' already exists in the sign dataset."
            }), 409

        # ── Label-based duplicate check against approved teacher signs ──
        if vocabulary.find_one({"english_word": {"$regex": f"^{label}$", "$options": "i"}}):
            return jsonify({
                "status": "rejected",
                "reason": "duplicate",
                "message": f"'{label}' has already been approved and added previously."
            }), 409

        # ── NEW: motion-based duplicate check (DTW) ──
        motion_matches = check_motion_duplicate(frames, exclude_label=label)
        has_motion_warning = len(motion_matches) > 0

        doc = {
            "teacher_id": data["teacher_id"],
            "teacher_email": data["teacher_email"],
            "english_word": label,
            "sinhala_word": data["sinhala_word"],
            "category": data["category"],
            "keypoint_sequence": frames,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "notified_authority": False,
            "motion_duplicate_warning": has_motion_warning,   # NEW
            "similar_signs": motion_matches,                   # NEW
        }
        result = submissions.insert_one(doc)
        pending_count = submissions.count_documents({"status": "pending", "notified_authority": False})

        response = {
            "submission_id": str(result.inserted_id),
            "status": "pending",
            "pending_batch_count": pending_count,
            "motion_duplicate_warning": has_motion_warning,   # NEW
            "similar_signs": motion_matches,                   # NEW
        }
        return jsonify(response), 201

    except Exception as e:
        print(f"❌ submit_sign error: {e}")
        return jsonify({"error": str(e)}), 500


@teacher_bp.route("/api/teacher/my-submissions/<teacher_id>", methods=["GET"])
def my_submissions(teacher_id):
    docs = list(
        submissions.find({"teacher_id": teacher_id}, {"keypoint_sequence": 0})
        .sort("created_at", -1)
    )
    for d in docs:
        d["_id"] = str(d["_id"])
        d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else None
    return jsonify(docs)


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


@teacher_bp.route("/api/teacher/classroom-signs", methods=["GET"])
def get_classroom_signs():
    if os.path.exists(JANITH_CLASSES_NPY):
        try:
            classes = np.load(JANITH_CLASSES_NPY, allow_pickle=True)
            signs = sorted([str(c) for c in classes.tolist()])
            return jsonify({"signs": signs, "source": "classes.npy"})
        except Exception as e:
            print(f"⚠️  Could not load classes.npy: {e} — using fallback list")

    return jsonify({"signs": sorted(FALLBACK_CLASSROOM_SIGNS), "source": "fallback"})


@teacher_bp.route("/api/teacher/pending-batch", methods=["GET"])
def pending_batch():
    pending_docs = list(
        submissions.find(
            {"status": "pending", "notified_authority": False},
            {"english_word": 1, "category": 1, "teacher_id": 1, "sinhala_word": 1}
        )
    )
    count = len(pending_docs)
    total_awaiting_decision = submissions.count_documents({"status": "pending"})
    for d in pending_docs:
        d["_id"] = str(d["_id"])
    return jsonify({
        "count": count,
        "signs": pending_docs,
        "total_awaiting_decision": total_awaiting_decision,
    })


@teacher_bp.route("/api/teacher/send-to-authority", methods=["POST"])
def send_to_authority():
    try:
        data = request.json or {}
        authority_email = data.get("authority_email", "").strip()
        submission_ids = data.get("submission_ids", [])

        if not authority_email or "@" not in authority_email:
            return jsonify({"error": "Please provide a valid email address."}), 400
        if not submission_ids:
            return jsonify({"error": "Please select at least one sign to send."}), 400

        try:
            object_ids = [ObjectId(sid) for sid in submission_ids]
        except Exception:
            return jsonify({"error": "Invalid submission id(s)."}), 400

        pending_docs = list(submissions.find({
            "_id": {"$in": object_ids},
            "status": "pending",
            "notified_authority": False,
        }))

        if not pending_docs:
            return jsonify({"error": "Selected signs are no longer available to send."}), 400

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
        subject = f"SLSL App: {len(pending_docs)} New Sign(s) Awaiting Approval"

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
            "message": f"Sent {len(pending_docs)} sign(s) to {authority_email}."
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
# NEW: shows a motion-duplicate warning banner when DTW flagged similarity
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
  .skeleton-box { background:#000000aa; border-radius:12px; margin-bottom:14px; overflow:hidden; }
  canvas { display:block; width:100%; height:220px; }
  .play-controls { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
  .play-btn { background:#00B4D8; color:#fff; padding:8px 14px; font-size:13px; margin:0; }
  .frame-label { color:#ffffff77; font-size:12px; }
  .warning-banner { background:#FFB70322; border:1px solid #FFB703; border-radius:10px; padding:12px 14px; margin-bottom:14px; }
  .warning-title { color:#FFB703; font-weight:700; font-size:13px; margin-bottom:6px; }
  .warning-item { color:#ffffffcc; font-size:12px; }
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

  {% if s.motion_duplicate_warning %}
  <div class="warning-banner">
    <div class="warning-title">⚠️ Possible motion duplicate detected</div>
    {% for m in s.similar_signs %}
    <div class="warning-item">Similar to "{{ m.label }}" (DTW distance: {{ m.distance }})</div>
    {% endfor %}
  </div>
  {% endif %}

  <div class="skeleton-box">
    <canvas id="canvas-{{ s._id }}" width="400" height="220"></canvas>
  </div>
  <div class="play-controls">
    <button class="play-btn" onclick="togglePlay('{{ s._id }}')" id="playbtn-{{ s._id }}">⏸ Pause</button>
    <span class="frame-label" id="framelabel-{{ s._id }}">Frame 1 / 30</span>
  </div>

  <button class="approve" onclick="act('{{ s._id }}','approve')">Approve</button>
  <button class="reject" onclick="act('{{ s._id }}','reject')">Reject</button>
</div>
{% endfor %}

<script>
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const players = {};

function initPlayer(id, keypointData) {
  const canvas = document.getElementById('canvas-' + id);
  const ctx = canvas.getContext('2d');
  players[id] = { keypoints: keypointData, frame: 0, playing: true, canvas, ctx };
}

function drawFrame(id) {
  const p = players[id];
  if (!p) return;
  const kp = p.keypoints[p.frame];
  if (!kp || kp.length < 63) return;

  const { ctx, canvas } = p;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const points = [];
  for (let i = 0; i < 21; i++) {
    const x = kp[i*3] * canvas.width;
    const y = kp[i*3+1] * canvas.height;
    points.push([x, y]);
  }

  ctx.strokeStyle = '#00B4D8';
  ctx.lineWidth = 2.5;
  HAND_CONNECTIONS.forEach(([a,b]) => {
    ctx.beginPath();
    ctx.moveTo(points[a][0], points[a][1]);
    ctx.lineTo(points[b][0], points[b][1]);
    ctx.stroke();
  });

  ctx.fillStyle = '#06D6A0';
  points.forEach(([x,y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  const label = document.getElementById('framelabel-' + id);
  if (label) label.textContent = 'Frame ' + (p.frame + 1) + ' / ' + p.keypoints.length;
}

function togglePlay(id) {
  const p = players[id];
  if (!p) return;
  p.playing = !p.playing;
  const btn = document.getElementById('playbtn-' + id);
  if (btn) btn.textContent = p.playing ? '⏸ Pause' : '▶ Play';
}

setInterval(() => {
  Object.keys(players).forEach(id => {
    const p = players[id];
    if (p.playing && p.keypoints && p.keypoints.length > 0) {
      drawFrame(id);
      p.frame = (p.frame + 1) % p.keypoints.length;
    }
  });
}, 130);

{% for s in signs %}
initPlayer('{{ s._id }}', {{ s.keypoint_sequence | tojson }});
{% endfor %}

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
    delete players[id];
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