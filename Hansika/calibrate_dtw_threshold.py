# Hansika/calibrate_dtw_threshold.py
"""
Run this after you have a few approved signs in vocabulary, to see actual
DTW distances between different signs — this tells you what threshold value
makes sense for YOUR data, instead of guessing.

Look at the printed distances:
- Compare pairs you KNOW are different signs → note the typical distance range
- If you have any accidental near-duplicates recorded → note how close they are
- Pick a threshold between those two ranges
"""
from pymongo import MongoClient
from fastdtw import fastdtw
import numpy as np
import itertools

client = MongoClient("mongodb://localhost:27017")
db = client["slsl_app"]
vocabulary = db["sign_vocabulary"]


def compute_dtw_distance(seq_a, seq_b):
    a = np.array(seq_a, dtype=float)
    b = np.array(seq_b, dtype=float)
    distance, _ = fastdtw(a, b, dist=lambda x, y: float(np.linalg.norm(x - y)))
    return distance / max(len(a), len(b))


def main():
    docs = list(vocabulary.find({}, {"english_word": 1, "keypoint_sequence": 1}))
    if len(docs) < 2:
        print("Need at least 2 approved signs in vocabulary to compare. Add more signs first.")
        return

    print(f"Comparing {len(docs)} approved signs pairwise...\n")
    results = []
    for a, b in itertools.combinations(docs, 2):
        if not a.get("keypoint_sequence") or not b.get("keypoint_sequence"):
            continue
        dist = compute_dtw_distance(a["keypoint_sequence"], b["keypoint_sequence"])
        results.append((a["english_word"], b["english_word"], dist))

    results.sort(key=lambda r: r[2])

    print(f"{'Sign A':<20} {'Sign B':<20} {'DTW Distance'}")
    print("-" * 55)
    for a, b, dist in results:
        print(f"{a:<20} {b:<20} {dist:.4f}")

    print("\n📌 Look at the numbers above:")
    print("   - Pick a threshold BELOW the lowest distance you see between")
    print("     signs you know are genuinely different.")
    print("   - If any pair looks suspiciously close, investigate — it might")
    print("     be an actual accidental duplicate in your existing dataset.")


if __name__ == "__main__":
    main()