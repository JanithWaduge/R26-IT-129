# Hansika/export_for_training.py
"""
Exports approved teacher-submitted signs into Janith's keypoints_data.csv format.
Run manually after authority approves new signs. Then Janith re-runs his own
clean_data.py + train_model.py to retrain — his scripts are untouched.
"""
import pandas as pd
from pymongo import MongoClient
import os

client = MongoClient("mongodb://localhost:27017")
db = client["slsl_app"]
vocabulary = db["sign_vocabulary"]

JANITH_CSV = os.path.join(os.path.dirname(__file__), '..', 'Janith', 'keypoints_data.csv')

def export_approved_signs():
    approved = list(vocabulary.find({"source": "teacher_submission"}))
    if not approved:
        print("No approved teacher-submitted signs to export.")
        return

    rows = []
    for doc in approved:
        seq = doc["keypoint_sequence"]  # 30 x 63
        flat = [val for frame in seq for val in frame]
        if len(flat) != 1890:
            print(f"⚠️  Skipping '{doc['english_word']}' — bad shape ({len(flat)} != 1890)")
            continue
        rows.append(flat + [doc["english_word"]])

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
    print(f"✅ Exported {len(new_df)} new signs into {JANITH_CSV}")
    print("   Janith should now run: clean_data.py → train_model.py")

if __name__ == "__main__":
    export_approved_signs()