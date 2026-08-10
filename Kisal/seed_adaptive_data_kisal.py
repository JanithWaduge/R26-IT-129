# seed_adaptive_data_kisal.py
#
# Sync (PyMongo) version of Kisal's original seed_data.py.
# Run this to populate the `curriculum` collection in your SHARED
# database with vocabulary so /adaptive/quiz/next has data to serve.
#
# DB_NAME confirmed to match Hansika's teacher_dashboard_server.py
# (db = client["slsl_app"]) - do not change unless that file changes too.
#
# Usage:
#   python -m pip install pymongo --break-system-packages   (if not already installed)
#   python seed_adaptive_data_kisal.py

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "slsl_app"  # matches Hansika's teacher_dashboard_server.py: db = client["slsl_app"]

# NOTE on avatar_asset_path:
# quiz_screen_kisal.dart does NOT use this field at inference time - it
# derives the video filename itself as `lib/video_ass/{word_english}.mp4`.
# The field is kept here for completeness / future use, but what actually
# matters for the quiz to work is that a matching .mp4 file exists in
# lib/video_ass/ named exactly after word_english.
#
# NOTE on translations:
# word_sinhala values below are copied directly from slsl_server.py's own
# SINHALA_TRANSLATIONS dictionary (your team's existing, already-used data
# for these 30 classroom signs) - not invented here.
# word_tamil is left as "" (empty) for every entry: there is no existing
# Tamil translation source in your codebase for this vocabulary, and a
# fabricated translation would be worse than none. Fill these in with a
# Tamil speaker/reviewer before relying on them, or drop the field's
# display in the UI until they're filled in.
sample_signs = [
    {"sign_id": "SLSL_SCH_01", "word_english": "Allocate",          "word_sinhala": "බෙදා හැරීම",              "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Allocate.mp4"},
    {"sign_id": "SLSL_SCH_02", "word_english": "Answer",            "word_sinhala": "පිළිතුර",                  "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Answer.mp4"},
    {"sign_id": "SLSL_SCH_03", "word_english": "Answer Properly",   "word_sinhala": "නිසි ලෙස පිළිතුරු දෙන්න",  "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Answer Properly.mp4"},
    {"sign_id": "SLSL_SCH_04", "word_english": "Answer Sheet",      "word_sinhala": "පිළිතුරු පත්‍රය",          "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Answer Sheet.mp4"},
    {"sign_id": "SLSL_SCH_05", "word_english": "Ask Question",      "word_sinhala": "ප්‍රශ්නය අසන්න",           "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Ask Question.mp4"},
    {"sign_id": "SLSL_SCH_06", "word_english": "Attend",            "word_sinhala": "සහභාගී වෙන්න",             "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Attend.mp4"},
    {"sign_id": "SLSL_SCH_07", "word_english": "Attending",         "word_sinhala": "සහභාගී වෙමින්",            "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Attending.mp4"},
    {"sign_id": "SLSL_SCH_08", "word_english": "Calculate",         "word_sinhala": "ගණනය කරන්න",               "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Calculate.mp4"},
    {"sign_id": "SLSL_SCH_09", "word_english": "Cancel",            "word_sinhala": "අවලංගු කරන්න",             "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Cancel.mp4"},
    {"sign_id": "SLSL_SCH_10", "word_english": "Collaborating",     "word_sinhala": "සහයෝගයෙන් කටයුතු කිරීම",   "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Collaborating.mp4"},
    {"sign_id": "SLSL_SCH_11", "word_english": "Collect",           "word_sinhala": "එකතු කරන්න",               "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Collect.mp4"},
    {"sign_id": "SLSL_SCH_12", "word_english": "Comparing",         "word_sinhala": "සංසන්දනය කිරීම",           "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Comparing.mp4"},
    {"sign_id": "SLSL_SCH_13", "word_english": "Concentrate",       "word_sinhala": "අවධානය යොමු කරන්න",        "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Concentrate.mp4"},
    {"sign_id": "SLSL_SCH_14", "word_english": "Continuing",        "word_sinhala": "දිගටම කරගෙන යන්න",         "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Continuing.mp4"},
    {"sign_id": "SLSL_SCH_15", "word_english": "Coordinate",        "word_sinhala": "සම්බන්ධීකරණය කරන්න",       "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Coordinate.mp4"},
    {"sign_id": "SLSL_SCH_16", "word_english": "Correct Mistake",   "word_sinhala": "වැරදි නිවැරදි කරන්න",      "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Correct Mistake.mp4"},
    {"sign_id": "SLSL_SCH_17", "word_english": "Describe",          "word_sinhala": "විස්තර කරන්න",             "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Describe.mp4"},
    {"sign_id": "SLSL_SCH_18", "word_english": "Discuss",           "word_sinhala": "සාකච්ඡා කරන්න",            "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Discuss.mp4"},
    {"sign_id": "SLSL_SCH_19", "word_english": "Discuss Topic",     "word_sinhala": "මාතෘකාව සාකච්ඡා කරන්න",    "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Discuss Topic.mp4"},
    {"sign_id": "SLSL_SCH_20", "word_english": "Distribute",        "word_sinhala": "බෙදා දෙන්න",               "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Distribute.mp4"},
    {"sign_id": "SLSL_SCH_21", "word_english": "Documenting",       "word_sinhala": "ලේඛනගත කිරීම",             "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Documenting.mp4"},
    {"sign_id": "SLSL_SCH_22", "word_english": "Grade",             "word_sinhala": "ශ්‍රේණිය",                 "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Grade.mp4"},
    {"sign_id": "SLSL_SCH_23", "word_english": "Practice",          "word_sinhala": "පුහුණු වෙන්න",             "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Practice.mp4"},
    {"sign_id": "SLSL_SCH_24", "word_english": "Research",          "word_sinhala": "පර්යේෂණය",                 "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Research.mp4"},
    {"sign_id": "SLSL_SCH_25", "word_english": "Review",            "word_sinhala": "සමාලෝචනය",                 "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Review.mp4"},
    {"sign_id": "SLSL_SCH_26", "word_english": "Support",           "word_sinhala": "සහාය දෙන්න",               "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Support.mp4"},
    {"sign_id": "SLSL_SCH_27", "word_english": "Teacher",           "word_sinhala": "ගුරුවරයා",                 "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Teacher.mp4"},
    {"sign_id": "SLSL_SCH_28", "word_english": "Whiteboard Marker",  "word_sinhala": "වයිට්බෝඩ් මාකර්",          "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Whiteboard Marker.mp4"},

    # Not one of Janith's 30 classroom signs - kept from your original demo
    # set since Student.mp4 already exists in video_ass. Fill in real
    # word_sinhala / word_tamil here once you have verified translations -
    # left empty rather than guessed.
    {"sign_id": "SLSL_SCH_29", "word_english": "Student",           "word_sinhala": "",                          "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Student.mp4"},

    # Not yet added to video_ass/ - uncomment once you get these two videos
    # from Janith's dataset (Copying, Study):
    # {"sign_id": "SLSL_SCH_30", "word_english": "Copying", "word_sinhala": "පිටපත් කිරීම",   "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Copying.mp4"},
    # {"sign_id": "SLSL_SCH_31", "word_english": "Study",   "word_sinhala": "අධ්‍යයනය කරන්න", "word_tamil": "", "category": "school", "avatar_asset_path": "lib/video_ass/Study.mp4"},
]


def seed():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    for sign in sample_signs:
        db["curriculum"].update_one(
            {"sign_id": sign["sign_id"]},
            {"$set": sign},
            upsert=True,
        )

    print(f"Database '{DB_NAME}' successfully seeded with {len(sample_signs)} SLSL curriculum entries!")
    client.close()


if __name__ == "__main__":
    seed()