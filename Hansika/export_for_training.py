# Hansika/export_for_training.py
"""
Exports approved teacher-submitted signs into Janith's keypoints_data.csv format.
Run manually after the Authority approves new signs. Then Janith re-runs his own
clean_data.py + train_model.py to retrain — his scripts are untouched.

Safe to run multiple times: already-exported signs are skipped automatically.
"""
import pandas as pd
from pymongo import MongoClient
import os
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")
db = client["slsl_app"]
vocabulary = db["sign_vocabulary"]

JANITH_CSV = os.path.join(os.path.dirname(__file__), '..', 'Janith', 'keypoints_data.csv')


def export_approved_signs():
    # Only pull signs that haven't been exported yet
    approved = list(vocabulary.find({
        "source": "teacher_submission",
        "exported": {"$ne": True}
    }))

    if not approved:
        print("No new approved signs to export. (All caught up.)")
        return

    rows = []
    exported_ids = []

    for doc in approved:
        seq = doc.get("keypoint_sequence")
        if not seq:
            print(f"⚠️  Skipping '{doc.get('english_word')}' — no keypoint data stored")
            continue

        flat = [val for frame in seq for val in frame]
        if len(flat) != 1890:  # 30 frames x 63 keypoints
            print(f"⚠️  Skipping '{doc.get('english_word')}' — bad shape ({len(flat)} != 1890)")
            continue

        rows.append(flat + [doc["english_word"]])
        exported_ids.append(doc["_id"])

    if not rows:
        print("Nothing valid to export.")
        return

    columns = [f'f{i}' for i in range(1890)] + ['label']
    new_df = pd.DataFrame(rows, columns=columns)

    if os.path.exists(JANITH_CSV):
        existing_df = pd.read_csv(JANITH_CSV)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(JANITH_CSV, index=False)

    # Mark these as exported so they're never duplicated on the next run
    vocabulary.update_many(
        {"_id": {"$in": exported_ids}},
        {"$set": {"exported": True, "exported_at": datetime.utcnow()}}
    )

    print(f"✅ Exported {len(new_df)} new sign(s) into {JANITH_CSV}")
    print(f"   Total dataset size now: {len(combined)} rows")
    print("\n📌 Next steps for Janith:")
    print("   python training/clean_data.py")
    print("   python training/train_model.py")


if __name__ == "__main__":
    export_approved_signs()