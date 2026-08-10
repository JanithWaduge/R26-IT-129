# adaptive_lesson_routes_kisal.py
#
# Part 03 - Adaptive Lesson System (Kisal)
#
# WHAT THIS IS
# Kisal's original module was a *separate* FastAPI + Motor (async MongoDB)
# service (slsl_backend/app/...), meant to run on its own with its own
# `uvicorn app.main:app` process and its own database ("slsl_learning_db").
#
# This file ports that same logic (routes, SM-2 algorithm, dashboard
# analytics) into a single Flask Blueprint using PyMongo (sync), following
# the exact same self-contained pattern as Hansika's teacher_dashboard_server.py
# (which slsl_server.py already imports as `from teacher_dashboard_server
# import teacher_bp`). This file connects to Mongo itself and exposes a
# ready-to-register `adaptive_bp` - no wiring/init function needed, so the
# import line in slsl_server.py is a one-liner, matching Hansika's module.
#
# ROUTES (all under the /adaptive prefix so they can never collide with
# Janith's or Hansika's routes):
#   GET  /adaptive/quiz/next?user_id=...&category=...&limit=5&include_practice=false
#   POST /adaptive/quiz/submit               (JSON body: user_id, sign_id, user_answer, response_time_seconds)
#   GET  /adaptive/dashboard/analytics?user_id=...
#
# COLLECTIONS USED (in your shared MongoDB database):
#   curriculum        - static vocabulary items (sign_id, word_english, word_sinhala, word_tamil, category, avatar_asset_path)
#   student_tracker   - per-student SM-2 state (user_id, sign_id, ease_factor, repetitions, interval_days, next_review_date)
#
# HOW TO WIRE THIS INTO slsl_server.py
# -------------------------------------
#   sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Kisal'))
#   from adaptive_lesson_routes_kisal import adaptive_bp
#   app.register_blueprint(adaptive_bp)
#
# IMPORTANT: set MONGO_URI / DB_NAME below to match EXACTLY what
# teacher_dashboard_server.py uses, so this module reads/writes the same
# shared database instead of a separate one. Open that file and copy the
# same two values across.

from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# TODO: replace these two values with whatever teacher_dashboard_server.py
# already uses (same MongoDB server, same database name = shared database).
# ---------------------------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "slsl_app"  # matches Hansika's teacher_dashboard_server.py: db = client["slsl_app"]

_client = MongoClient(MONGO_URI)
db = _client[DB_NAME]

adaptive_bp = Blueprint("adaptive_lesson_kisal", __name__, url_prefix="/adaptive")


# ---------------------------------------------------------------------------
# SM-2 engine (pure functions - unchanged from Kisal's original sm2_engine.py,
# these have no async/Mongo dependency so they port over verbatim)
# ---------------------------------------------------------------------------

def calculate_sm2(current_ef, current_repetitions, current_interval, quality_score):
    """Executes pure SuperMemo-2 (SM-2) mathematical calculations."""
    if not (0 <= quality_score <= 5):
        raise ValueError("Quality score must be an integer between 0 and 5.")

    if quality_score >= 3:
        if current_repetitions == 0:
            next_interval = 1
        elif current_repetitions == 1:
            next_interval = 6
        else:
            next_interval = round(current_interval * current_ef)
        next_repetitions = current_repetitions + 1
    else:
        # Failure path: reset repetitions, push to immediate next-day review
        next_interval = 1
        next_repetitions = 0

    next_ef = current_ef + (0.1 - (5 - quality_score) * (0.08 + (5 - quality_score) * 0.02))
    if next_ef < 1.3:
        next_ef = 1.3

    next_review_date = datetime.utcnow() + timedelta(days=next_interval)
    return next_ef, next_repetitions, next_interval, next_review_date


def calculate_receptive_score(user_answer, correct_answer, response_time_seconds):
    """Heuristically translates a text answer + response time into a 0-5 SM-2 quality score."""
    cleaned_user = (user_answer or "").strip().lower()
    cleaned_correct = (correct_answer or "").strip().lower()

    if cleaned_user != cleaned_correct:
        return 0
    if response_time_seconds < 5.0:
        return 5
    elif response_time_seconds <= 15.0:
        return 4
    else:
        return 3


# ---------------------------------------------------------------------------
# Quiz endpoints (ported from routes/quiz.py)
# ---------------------------------------------------------------------------

@adaptive_bp.route("/quiz/next", methods=["GET"])
def get_next_quiz_items():
    """
    Query Core Priority Loop:
    1. Look for signs in the category that are due (next_review_date <= now).
    2. If limit isn't reached, fill remaining slots with new/unstudied signs.
    3. Optionally (include_practice=true), fill any leftover slots with extra
       practice signs the scheduler hasn't queued yet.
    """
    user_id = request.args.get("user_id")
    category = request.args.get("category")
    limit = int(request.args.get("limit", 5))
    include_practice = request.args.get("include_practice", "false").lower() == "true"

    if not user_id or not category:
        return jsonify({"detail": "user_id and category are required"}), 400

    current_time = datetime.utcnow()

    # Bucket 1: items explicitly due for review
    due_trackers = list(db["student_tracker"].find({
        "user_id": user_id,
        "next_review_date": {"$lte": current_time}
    }).limit(limit))
    due_sign_ids = [t["sign_id"] for t in due_trackers]

    quiz_items = []
    if due_sign_ids:
        quiz_items = list(db["curriculum"].find({
            "sign_id": {"$in": due_sign_ids},
            "category": category
        }))

    # Bucket 2: fallback to unstudied signs in this category
    if len(quiz_items) < limit:
        remaining_slots = limit - len(quiz_items)
        all_tracked = list(db["student_tracker"].find({"user_id": user_id}).limit(500))
        tracked_sign_ids = [t["sign_id"] for t in all_tracked]
        new_items = list(db["curriculum"].find({
            "category": category,
            "sign_id": {"$nin": tracked_sign_ids}
        }).limit(remaining_slots))
        quiz_items.extend(new_items)

    # Bucket 3: optional manual practice mode
    if include_practice and len(quiz_items) < limit:
        remaining_slots = limit - len(quiz_items)
        selected_sign_ids = [item["sign_id"] for item in quiz_items]
        practice_items = list(db["curriculum"].find({
            "category": category,
            "sign_id": {"$nin": selected_sign_ids}
        }).limit(remaining_slots))
        quiz_items.extend(practice_items)

    if not quiz_items:
        return jsonify({"detail": f"No due or new quiz items for category: '{category}'"}), 404

    # Strip Mongo's ObjectId - it isn't JSON serializable and Flutter's
    # SignItem.fromJson doesn't expect it
    for item in quiz_items:
        item.pop("_id", None)

    return jsonify(quiz_items), 200


@adaptive_bp.route("/quiz/submit", methods=["POST"])
def submit_quiz_answer():
    """
    Grades the student's text submission, updates the SM-2 algorithm
    variables, and reschedules the item in MongoDB.
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id")
    sign_id = data.get("sign_id")
    user_answer = data.get("user_answer")
    response_time_seconds = data.get("response_time_seconds")

    if not user_id or not sign_id or response_time_seconds is None:
        return jsonify({"detail": "user_id, sign_id and response_time_seconds are required"}), 400

    target_item = db["curriculum"].find_one({"sign_id": sign_id})
    if not target_item:
        return jsonify({"detail": "Target vocabulary sign item not found."}), 404

    quality_score = calculate_receptive_score(
        user_answer=user_answer,
        correct_answer=target_item["word_english"],
        response_time_seconds=response_time_seconds
    )

    tracker = db["student_tracker"].find_one({"user_id": user_id, "sign_id": sign_id})
    if tracker:
        current_ef = tracker["ease_factor"]
        current_rep = tracker["repetitions"]
        current_interval = tracker["interval_days"]
    else:
        current_ef, current_rep, current_interval = 2.5, 0, 0

    next_ef, next_rep, next_interval, next_date = calculate_sm2(
        current_ef=current_ef,
        current_repetitions=current_rep,
        current_interval=current_interval,
        quality_score=quality_score
    )

    db["student_tracker"].update_one(
        {"user_id": user_id, "sign_id": sign_id},
        {"$set": {
            "ease_factor": next_ef,
            "repetitions": next_rep,
            "interval_days": next_interval,
            "next_review_date": next_date
        }},
        upsert=True
    )

    return jsonify({
        "is_correct": quality_score >= 3,
        "calculated_quality_score": quality_score,
        "next_scheduled_interval_days": next_interval,
        "next_review_date": next_date.isoformat()
    }), 200


# ---------------------------------------------------------------------------
# Dashboard analytics endpoint (ported from routes/dashboard.py)
# ---------------------------------------------------------------------------

@adaptive_bp.route("/dashboard/analytics", methods=["GET"])
def get_student_analytics():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"detail": "user_id is required"}), 400

    now = datetime.utcnow()
    trackers = list(db["student_tracker"].find({"user_id": user_id}).limit(1000))

    mastered_count = sum(
        1 for t in trackers
        if t.get("ease_factor", 0) >= 2.5 and t.get("repetitions", 0) >= 3
    )

    categories = ["school", "daily_life", "numbers", "emotions", "greetings"]
    category_mastery = {}
    for cat in categories:
        total_cat_signs = db["curriculum"].count_documents({"category": cat})
        if total_cat_signs == 0:
            category_mastery[cat] = 0.0
            continue

        cat_sign_ids = [
            d["sign_id"] for d in
            db["curriculum"].find({"category": cat}, {"sign_id": 1}).limit(500)
        ]
        cat_mastered = sum(
            1 for t in trackers
            if t["sign_id"] in cat_sign_ids and t.get("ease_factor", 0) >= 2.5 and t.get("repetitions", 0) >= 3
        )
        category_mastery[cat] = round((cat_mastered / total_cat_signs) * 100, 1)

    tracker_sign_ids = [t["sign_id"] for t in trackers]
    curriculum_lookup = {}
    if tracker_sign_ids:
        curriculum_items = list(db["curriculum"].find(
            {"sign_id": {"$in": tracker_sign_ids}}, {"_id": 0}
        ))
        curriculum_lookup = {item["sign_id"]: item for item in curriculum_items}

    review_queue = []
    total_ease_factor = 0.0
    total_interval_days = 0
    due_now_count = 0

    for t in trackers:
        sign_id = t["sign_id"]
        curriculum = curriculum_lookup.get(sign_id, {})
        next_review_date = t.get("next_review_date")
        is_due = bool(next_review_date and next_review_date <= now)

        total_ease_factor += t.get("ease_factor", 0)
        total_interval_days += t.get("interval_days", 0)
        if is_due:
            due_now_count += 1

        review_queue.append({
            "sign_id": sign_id,
            "word_english": curriculum.get("word_english", sign_id),
            "category": curriculum.get("category", "unknown"),
            "ease_factor": round(t.get("ease_factor", 0), 2),
            "repetitions": t.get("repetitions", 0),
            "interval_days": t.get("interval_days", 0),
            "next_review_date": next_review_date.isoformat() if next_review_date else None,
            "is_due": is_due
        })

    review_queue.sort(key=lambda item: item["next_review_date"] or "")

    tracked_count = len(trackers)
    sm2_summary = {
        "average_ease_factor": round(total_ease_factor / tracked_count, 2) if tracked_count else 0,
        "average_interval_days": round(total_interval_days / tracked_count, 1) if tracked_count else 0,
        "due_now_count": due_now_count,
        "scheduled_count": tracked_count - due_now_count,
        "algorithm_steps": [
            "Student watches the sign video and enters an English answer.",
            "The engine grades recall quality from 0 to 5 using correctness and response time.",
            "SM-2 updates ease factor, repetition streak, interval days, and next review date.",
            "Dashboard and quiz queue use the new schedule for adaptive practice."
        ],
        "quality_scale": [
            {"score": 5, "meaning": "Correct and fast recall"},
            {"score": 4, "meaning": "Correct with normal hesitation"},
            {"score": 3, "meaning": "Correct but slow"},
            {"score": 0, "meaning": "Incorrect answer"}
        ]
    }

    collected_data = {
        "student_id": user_id,
        "tracked_sign_records": tracked_count,
        "curriculum_categories": categories,
        "stored_tracker_fields": [
            "user_id", "sign_id", "ease_factor", "repetitions", "interval_days", "next_review_date"
        ],
        "latest_review_items": review_queue[:8]
    }

    # Mock/stub streak logic (matches Kisal's original placeholder note:
    # in full production this maps to explicit login timestamp differentials)
    active_streak = 3 if len(trackers) > 0 else 0

    return jsonify({
        "total_signs_encountered": len(trackers),
        "total_signs_mastered": mastered_count,
        "active_login_streak": active_streak,
        "category_mastery_percentages": category_mastery,
        "sm2_summary": sm2_summary,
        "collected_data": collected_data,
        "review_queue": review_queue
    }), 200