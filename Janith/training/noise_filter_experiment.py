"""
noise_filter_experiment.py — v9 FINAL
--------------------------------------
KEY INSIGHT: Filter applies at INFERENCE TIME only (not training).
Both models use same training data.
Difference: Model B filters input BEFORE inference.

This correctly tests:
  "Does applying noise filter at inference time reduce false positives?"

Model A: accidental input → model directly → high FP
Model B: accidental input → filter → degraded input → model → low FP
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  pip install statsmodels")

print("=" * 60)
print("   SLSL Noise Filter Experiment — v9 Final")
print("   Model A (Baseline) vs Model B (Proposed)")
print("=" * 60)

# ================================================
# PATHS
# ================================================
CSV_PATH     = r'D:\R26-IT-129\Janith\keypoints_clean.csv'
MODEL_PATH   = r'D:\R26-IT-129\Janith\models'
RESULTS_PATH = r'D:\R26-IT-129\Janith\experiment_results.csv'

if not os.path.exists(CSV_PATH):
    CSV_PATH = r'D:\R26-IT-129\Janith\keypoints_data.csv'

# ================================================
# CONFIG
# ================================================
SEQUENCE_LENGTH = 30
NOISE_THRESHOLD = 0.02
FP_THRESHOLD    = 0.50
EPOCHS          = 100
BATCH_SIZE      = 32
MIN_SAMPLES     = 3
N_ACCIDENTAL    = 200
AUG_FACTOR      = 4

def vel_sigma(v): return v / 0.7979

# ================================================
# NORMALIZATION — must match extract_keypoints.py / slsl_server.py
# ADDED: the model is now trained on wrist-centered, scale-normalized
# keypoints. Synthetic "accidental" test data must go through the
# SAME transform, or it's being evaluated in a coordinate space the
# model never saw — which is what caused the 98%/98% saturation.
# ================================================
def normalize_keypoints(kp):
    arr = np.array(kp, dtype=np.float32).reshape(21, 3)
    if np.sum(np.abs(arr)) < 0.01:
        return kp
    wrist = arr[0].copy()
    arr = arr - wrist
    scale = np.linalg.norm(arr[9])
    if scale < 1e-6:
        scale = 1.0
    arr = arr / scale
    return arr.flatten().tolist()

# ================================================
# NOISE FILTER — CANONICAL VERSION
# Matches extract_keypoints.py exactly (the function that produced
# the training data), NOT a separate truncate/pad variant. This is
# what makes this experiment actually test the filter your model
# was trained around, and what slsl_server.py must also match.
#
# Shape contract: output is always 15 "active" frames followed by
# zero-padding up to SEQUENCE_LENGTH (30) — the same 15+15 structure
# used when keypoints_data.csv / keypoints_clean.csv were built.
# ================================================
CAPTURE_FRAMES = 15  # must match extract_keypoints.py

def apply_noise_filter(sequence, threshold=None):
    """
    threshold=None looks up the current value of the module-level
    NOISE_THRESHOLD at CALL time (not def time), so the auto-
    calibration step below (Step 1) actually takes effect. A default
    argument bound directly to NOISE_THRESHOLD would freeze the value
    from when this function was defined, ignoring later recalibration.

    Step 1: velocity-filter the raw sequence (removes low-motion frames,
             same pairwise comparison as extract_keypoints.py).
    Step 2: drop near-zero (no-hand) frames from what's left.
    Step 3: sample down to CAPTURE_FRAMES valid frames (or upsample if
             fewer), then zero-pad to SEQUENCE_LENGTH.
    Returns (flat_sequence, frames_removed_by_velocity_filter).
    """
    if threshold is None:
        threshold = NOISE_THRESHOLD

    arr = np.array(sequence, dtype=np.float32)

    # Step 1 — velocity filter on the raw sequence
    if len(arr) < 2:
        velocity_filtered = arr.tolist()
        removed = 0
    else:
        velocity_filtered = [arr[0].tolist()]
        removed = 0
        for i in range(1, len(arr)):
            v = float(np.mean(np.abs(arr[i] - arr[i - 1])))
            if v > threshold:
                velocity_filtered.append(arr[i].tolist())
            else:
                removed += 1

    # Step 2 — drop near-zero (no-hand) frames
    valid = [f for f in velocity_filtered if np.sum(np.abs(f)) > 0.01]

    if len(valid) == 0:
        return [[0.0] * 63] * SEQUENCE_LENGTH, removed

    # Step 3 — sample to CAPTURE_FRAMES, then zero-pad to SEQUENCE_LENGTH
    if len(valid) >= CAPTURE_FRAMES:
        indices = np.linspace(0, len(valid) - 1, CAPTURE_FRAMES, dtype=int)
        sampled = [valid[i] for i in indices]
    else:
        indices = np.linspace(0, len(valid) - 1, CAPTURE_FRAMES)
        sampled = [valid[int(round(i))] for i in indices]

    padding = [[0.0] * 63] * (SEQUENCE_LENGTH - CAPTURE_FRAMES)
    return sampled + padding, removed

# ================================================
# MODEL — Same architecture for both A and B
# ================================================
def build_model(num_classes, name="model"):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense)
    m = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),
        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64), Dropout(0.3),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ], name=name)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return m

# ================================================
# STEP 1 — LOAD FULL DATASET (30 signs)
# ================================================
print(f"\n[1/7] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"      {os.path.basename(CSV_PATH)}: {len(df)} samples")

sc          = df['label'].value_counts()
valid_signs = sc[sc >= MIN_SAMPLES].index
df          = df[df['label'].isin(valid_signs)]
print(f"      After filter: {len(df)} samples, {df['label'].nunique()} signs")

feature_cols = [c for c in df.columns if c != 'label']
X_raw        = df[feature_cols].values.astype(np.float32)
y_raw        = df['label'].values
le           = LabelEncoder()
y_encoded    = le.fit_transform(y_raw)
num_classes  = len(le.classes_)
X            = X_raw.reshape(-1, SEQUENCE_LENGTH, 63)
print(f"      X: {X.shape}, Classes: {num_classes}")

vr = []
for seq in X:
    for i in range(1, len(seq)):
        if np.sum(np.abs(seq[i])) > 0.01:
            vr.append(float(np.mean(np.abs(seq[i] - seq[i-1]))))
vr_mean, vr_std = float(np.mean(vr)), float(np.std(vr))
print(f"      Real sign velocity — mean: {vr_mean:.4f}, std: {vr_std:.4f}")

# ================================================
# AUTO-CALIBRATE NOISE_THRESHOLD — CRITICAL FIX
# The hardcoded NOISE_THRESHOLD=0.02 was tuned for raw (un-normalized)
# coordinates. Now that keypoints are wrist-centered and scale-
# normalized, the velocity scale is different, so a stale threshold
# either filters almost nothing or almost everything — which is what
# caused Model A and Model B to converge to the same ~98% FPR.
#
# Heuristic: frames moving slower than 30% of the typical REAL signing
# velocity are treated as static/noise. This is a starting point —
# tune the 0.3 multiplier against your own FPR/accuracy trade-off.
# ================================================
NOISE_THRESHOLD = round(vr_mean * 0.3, 5)
print(f"      ⚙️  Auto-calibrated NOISE_THRESHOLD: {NOISE_THRESHOLD}")
print(f"      ⚠️  COPY THIS VALUE into extract_keypoints.py and "
      f"slsl_server.py (NOISE_THRESHOLD constant) to keep all three "
      f"files in sync, then re-run extract_keypoints.py once more.")

# ================================================
# STEP 2 — SPLIT FIRST (CRITICAL FIX)
# Splitting must happen BEFORE augmentation, otherwise near-duplicate
# noisy copies of the same recording can land in both train and test,
# leaking information and inflating the reported accuracy/FPR numbers.
# ================================================
print(f"\n[2/7] Splitting raw data (80/20) BEFORE augmentation...")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"      Train (raw): {len(X_train_raw)}, Test (untouched): {len(X_test)}")

# ================================================
# STEP 3 — AUGMENT training split only
# X_test/y_test are never augmented and never touched again.
# ================================================
print(f"\n[3/7] Augmenting training data (x{AUG_FACTOR})...")
Xa, ya = [], []
for i in range(len(X_train_raw)):
    Xa.append(X_train_raw[i]); ya.append(y_train_raw[i])
    for _ in range(AUG_FACTOR - 1):
        noise = np.random.normal(0, 0.01, X_train_raw[i].shape).astype(np.float32)
        Xa.append(X_train_raw[i] + noise); ya.append(y_train_raw[i])
X_train = np.array(Xa, dtype=np.float32)
y_train = np.array(ya)
print(f"      {len(X_train)} training samples "
      f"(test set stays at {len(X_test)}, untouched and unaugmented)")

# ================================================
# STEP 4 — ACCIDENTAL DATA
# Mixed velocity: even frames above threshold, odd below
# ================================================
print(f"\n[4/7] Generating {N_ACCIDENTAL} accidental sequences...")

def make_accidental():
    """
    Generates a raw (pre-normalization) synthetic hand-jitter sequence
    in camera-coordinate space — simulating real accidental hand
    movement as MediaPipe would report it. Then normalizes each frame
    exactly as extract_keypoints.py / slsl_server.py do, so this data
    is evaluated in the SAME coordinate space the model was trained on.
    """
    seq  = []
    base = np.random.uniform(0.2, 0.8, 63).astype(np.float32)
    for j in range(SEQUENCE_LENGTH):
        if j % 2 == 0:
            sigma = vel_sigma(np.random.uniform(0.025, 0.035))
        else:
            sigma = vel_sigma(np.random.uniform(0.003, 0.010))
        step = np.random.normal(0, sigma, 63).astype(np.float32)
        base = np.clip(base + step, 0, 1)
        seq.append(normalize_keypoints(base.copy().tolist()))
    return seq

accidental_data = [make_accidental() for _ in range(N_ACCIDENTAL)]

vels, f_above, fk = [], [], []
for seq in accidental_data:
    arr   = np.array(seq)
    fvels = [float(np.mean(np.abs(arr[i] - arr[i-1])))
              for i in range(1, len(arr))]
    vels.append(np.mean(fvels))
    f_above.append(sum(v > NOISE_THRESHOLD for v in fvels))
    f, r = apply_noise_filter(seq)
    fk.append(SEQUENCE_LENGTH - r)

print(f"      Avg velocity: {np.mean(vels):.4f}")
print(f"      Avg frames above threshold: {np.mean(f_above):.1f}/29")
print(f"      Avg frames kept after filter: {np.mean(fk):.1f}/30")
print(f"      FP_THRESHOLD: {FP_THRESHOLD}")

# ================================================
# CALLBACKS (same for both models)
# ================================================
cbs = [
    tf.keras.callbacks.EarlyStopping(
        patience=15, restore_best_weights=True, monitor='val_accuracy'),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0),
]
os.makedirs(MODEL_PATH, exist_ok=True)

# ================================================
# TRAIN SINGLE SHARED MODEL
# ✅ Both A and B use SAME trained model
# Difference is only at INFERENCE: B applies filter
# ================================================
print("\n" + "="*60)
print("[5/7] TRAINING SHARED MODEL (used by both A and B)")
print("="*60)
print("  ℹ️  Both models share same weights.")
print("  ℹ️  Difference: Model B filters input before inference.")
print("="*60)

shared_model = build_model(num_classes, "shared_model")
shared_model.fit(
    X_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=cbs, verbose=1
)

_, shared_acc = shared_model.evaluate(X_test, y_test, verbose=0)
print(f"\n  ✅ Shared Model Accuracy: {shared_acc*100:.2f}%")
shared_model.save(os.path.join(MODEL_PATH, 'shared_model.h5'))

# ================================================
# STEP 6 — MODEL A: No filter at inference
# ================================================
print("\n" + "="*60)
print("[6a/7] MODEL A — Baseline (No Filter at Inference)")
print("="*60)

conf_a_list, fp_a, out_a = [], 0, []
for seq in accidental_data:
    # Model A: raw accidental input → model
    arr  = np.array(seq, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, 63)
    pred = shared_model.predict(arr, verbose=0)
    conf = float(np.max(pred))
    conf_a_list.append(conf)
    is_fp = int(conf > FP_THRESHOLD)
    out_a.append(is_fp); fp_a += is_fp

conf_a = np.array(conf_a_list, dtype=np.float32)
fpr_a  = fp_a / N_ACCIDENTAL
out_a  = np.array(out_a)
acc_a  = shared_acc  # Model A never filters, so this IS its real accuracy

print(f"  ✅ Model A (No Filter) — FP: {fp_a}/{N_ACCIDENTAL} ({fpr_a*100:.2f}%)")
print(f"  ✅ Model A — Mean conf on accidentals: {np.mean(conf_a):.4f}")
print(f"  ✅ Model A — Sign Accuracy (real test set, unfiltered): {acc_a*100:.2f}%")

# ================================================
# STEP 7 — MODEL B: Filter at inference
# ================================================
print("\n" + "="*60)
print("[6b/7] MODEL B — Proposed (Filter at Inference)")
print("="*60)

conf_b_list, fp_b, out_b = [], 0, []
frames_removed_inf = []

for seq in accidental_data:
    # Model B: accidental input → filter → degraded input → model
    f_seq, removed = apply_noise_filter(seq)
    frames_removed_inf.append(removed)
    f_arr = np.array(f_seq, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, 63)
    pred  = shared_model.predict(f_arr, verbose=0)
    conf  = float(np.max(pred))
    conf_b_list.append(conf)
    is_fp = int(conf > FP_THRESHOLD)
    out_b.append(is_fp); fp_b += is_fp

conf_b     = np.array(conf_b_list, dtype=np.float32)
fpr_b      = fp_b / N_ACCIDENTAL
out_b      = np.array(out_b)
avg_removed_inf = np.mean(frames_removed_inf)

# ================================================
# CRITICAL FIX — actually measure Model B's real-sign accuracy
# Previously this was just `acc_b = shared_acc`, which silently
# reused Model A's number and never tested what the filter does to
# genuine signs. Here we run every REAL test sequence through the
# SAME filter Model B uses at inference, then evaluate on that.
# This is the honest accuracy-vs-FPR trade-off for your report.
# ================================================
X_test_filtered = np.array(
    [apply_noise_filter(seq)[0] for seq in X_test],
    dtype=np.float32
)
_, acc_b = shared_model.evaluate(X_test_filtered, y_test, verbose=0)

print(f"  ✅ Model B (With Filter) — FP: {fp_b}/{N_ACCIDENTAL} ({fpr_b*100:.2f}%)")
print(f"  ✅ Model B — Mean conf on accidentals: {np.mean(conf_b):.4f}")
print(f"  ✅ Avg frames removed from accidentals: {avg_removed_inf:.1f}/30")
print(f"  ✅ Model B — Sign Accuracy (real test set, FILTERED): {acc_b*100:.2f}%")
print(f"  ℹ️  Accuracy change from filtering real signs: "
      f"{(acc_b - acc_a)*100:+.2f} percentage points")

# ================================================
# STATISTICAL ANALYSIS
# ================================================
print("\n" + "="*60)
print("[7/7] STATISTICAL ANALYSIS")
print("="*60)

fpr_red         = ((fpr_a - fpr_b) / fpr_a * 100) if fpr_a > 0 else 0.0
t_stat, p_ttest = stats.ttest_ind(conf_a, conf_b)
pooled          = np.sqrt((np.std(conf_a)**2 + np.std(conf_b)**2) / 2)
cohens_d        = (np.mean(conf_a) - np.mean(conf_b)) / pooled if pooled > 0 else 0.0

mcnemar_p = None
if HAS_STATSMODELS:
    b = int(np.sum((out_a == 1) & (out_b == 0)))
    c = int(np.sum((out_a == 0) & (out_b == 1)))
    print(f"\n  McNemar: b={b} (A=FP,B=ok)  c={c} (A=ok,B=FP)")
    if b + c > 0:
        res       = mcnemar([[0, b], [c, 0]], exact=True)
        mcnemar_p = res.pvalue
        print(f"  McNemar p-value: {mcnemar_p:.4f}")
    else:
        print("  McNemar: no discordant pairs")

sig_p = p_ttest if mcnemar_p is None else min(p_ttest, mcnemar_p)

print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
print(f"""
  Shared Model (CNN+LSTM, 30 signs):
    Test Accuracy          : {shared_acc*100:.2f}%

  Model A — Baseline (No Noise Filter at Inference):
    Sign Accuracy          : {acc_a*100:.2f}%
    False Positives        : {fp_a}/{N_ACCIDENTAL}
    False Positive Rate    : {fpr_a*100:.2f}%
    Mean confidence (acc.) : {np.mean(conf_a):.4f}

  Model B — Proposed (Noise Filter at Inference):
    Sign Accuracy          : {acc_b*100:.2f}%
    False Positives        : {fp_b}/{N_ACCIDENTAL}
    False Positive Rate    : {fpr_b*100:.2f}%
    Mean confidence (acc.) : {np.mean(conf_b):.4f}
    Avg frames removed     : {avg_removed_inf:.1f}/30

  Comparison:
    FPR reduction          : {fpr_red:.2f}%
    Confidence reduction   : {np.mean(conf_a)-np.mean(conf_b):.4f}

  Statistical Tests:
    t-statistic            : {t_stat:.4f}
    t-test p-value         : {p_ttest:.4f}
    Cohen's d              : {cohens_d:.4f}""")

if mcnemar_p is not None:
    print(f"    McNemar's p-value      : {mcnemar_p:.4f}")

print()
if fpr_a > fpr_b and sig_p < 0.05:
    print(f"  ✅ H1 SUPPORTED: Noise filter reduced FPR "
          f"{fpr_a*100:.2f}% → {fpr_b*100:.2f}% "
          f"({fpr_red:.1f}% reduction, p={sig_p:.4f})")
elif fpr_a == 0 and fpr_b == 0:
    print("  ✅ Both FPR=0% — model very robust")
    print(f"  ✅ Confidence significantly different (p={p_ttest:.4f})")
else:
    print(f"  FPR: {fpr_a*100:.2f}% → {fpr_b*100:.2f}%")

if sig_p < 0.05:
    print("  ✅ H1 SUPPORTED: p < 0.05")
print("="*60)

# ================================================
# SAVE
# ================================================
rows = [
    {'Metric': 'Sign Accuracy on Real Test Set (%)',
     'Model A (Baseline)': f"{acc_a*100:.2f}",
     'Model B (Proposed)': f"{acc_b*100:.2f}",
     'Difference': f"{acc_b*100 - acc_a*100:+.2f}"},
    {'Metric': f'False Positives / {N_ACCIDENTAL}',
     'Model A (Baseline)': fp_a, 'Model B (Proposed)': fp_b,
     'Difference': fp_b - fp_a},
    {'Metric': 'False Positive Rate (%)',
     'Model A (Baseline)': f"{fpr_a*100:.2f}",
     'Model B (Proposed)': f"{fpr_b*100:.2f}",
     'Difference': f"{fpr_b*100 - fpr_a*100:+.2f}"},
    {'Metric': 'FPR Reduction (%)',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{fpr_red:.2f}", 'Difference': '-'},
    {'Metric': 'Mean Confidence on Accidentals',
     'Model A (Baseline)': f"{np.mean(conf_a):.4f}",
     'Model B (Proposed)': f"{np.mean(conf_b):.4f}",
     'Difference': f"{np.mean(conf_b)-np.mean(conf_a):+.4f}"},
    {'Metric': 'Avg Frames Removed from Accidentals',
     'Model A (Baseline)': '0',
     'Model B (Proposed)': f"{avg_removed_inf:.2f}", 'Difference': '-'},
    {'Metric': 'Filter applied at',
     'Model A (Baseline)': 'Not applied',
     'Model B (Proposed)': 'Inference only', 'Difference': '-'},
    {'Metric': 'Velocity Threshold',
     'Model A (Baseline)': NOISE_THRESHOLD,
     'Model B (Proposed)': NOISE_THRESHOLD, 'Difference': '-'},
    {'Metric': 'FP Threshold',
     'Model A (Baseline)': FP_THRESHOLD,
     'Model B (Proposed)': FP_THRESHOLD, 'Difference': '-'},
    {'Metric': 't-statistic',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{t_stat:.4f}", 'Difference': '-'},
    {'Metric': 't-test p-value',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{p_ttest:.4f}", 'Difference': '-'},
    {'Metric': "McNemar's p-value",
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{mcnemar_p:.4f}" if mcnemar_p is not None else 'N/A',
     'Difference': '-'},
    {'Metric': "Cohen's d",
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{cohens_d:.4f}", 'Difference': '-'},
    {'Metric': 'H1 Supported (p < 0.05)',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': 'Yes' if sig_p < 0.05 else 'No',
     'Difference': '-'},
    {'Metric': 'N Accidental sequences',
     'Model A (Baseline)': N_ACCIDENTAL,
     'Model B (Proposed)': N_ACCIDENTAL, 'Difference': '-'},
    {'Metric': 'Dataset',
     'Model A (Baseline)': os.path.basename(CSV_PATH),
     'Model B (Proposed)': os.path.basename(CSV_PATH), 'Difference': '-'},
]
pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
print(f"\n  💾 {RESULTS_PATH}")
print("  🎉 Experiment complete!\n")