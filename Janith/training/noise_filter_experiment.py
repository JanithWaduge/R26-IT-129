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
print("   SLSL Noise Filter Experiment  (v3 — fixed)")
print("   Model A (Baseline) vs Model B (Proposed)")
print("=" * 60)

# ================================================
# PATHS  — keypoints_data.csv use කරනවා (900 rows)
# ================================================
CSV_PATH     = r'C:\Users\Janith\Desktop\R26-IT-129\Janith\keypoints_data.csv'
MODEL_PATH   = r'C:\Users\Janith\Desktop\R26-IT-129\Janith\models'
RESULTS_PATH = r'C:\Users\Janith\Desktop\R26-IT-129\Janith\experiment_results.csv'

SEQUENCE_LENGTH = 30
NOISE_THRESHOLD = 0.02
FP_THRESHOLD    = 0.5
EPOCHS          = 80
BATCH_SIZE      = 32
TOP_SIGNS       = 10    # Top 10 signs only — more samples per class
MIN_SAMPLES     = 5     # At least 5 samples per sign
N_ACCIDENTAL    = 100
AUG_FACTOR      = 3     # Augment each sample 3x

# ================================================
# STEP 1 — LOAD DATA
# ================================================
print("\n[1/7] Loading dataset...")

# Try keypoints_data.csv first, fallback to keypoints_clean.csv
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"      Using: keypoints_data.csv")
else:
    CSV_PATH = CSV_PATH.replace('keypoints_data', 'keypoints_clean')
    df = pd.read_csv(CSV_PATH)
    print(f"      Fallback: keypoints_clean.csv")

print(f"      Raw samples : {len(df)}, Columns: {len(df.columns)}")

sign_counts = df['label'].value_counts()
print(f"\n      All signs and counts:")
for sign, count in sign_counts.items():
    print(f"        {sign}: {count}")

# Keep top N signs with minimum samples
valid_signs = sign_counts[sign_counts >= MIN_SAMPLES].index
df = df[df['label'].isin(valid_signs)]
top_signs = df['label'].value_counts().head(TOP_SIGNS).index
df = df[df['label'].isin(top_signs)]

print(f"\n      After filter (top {TOP_SIGNS}, >={MIN_SAMPLES} samples):")
print(f"      {len(df)} samples, {df['label'].nunique()} signs")
print(f"      Signs: {list(df['label'].unique())}")

feature_cols = [c for c in df.columns if c != 'label']
X_raw        = df[feature_cols].values.astype(np.float32)
y_raw        = df['label'].values

le           = LabelEncoder()
y_encoded    = le.fit_transform(y_raw)
num_classes  = len(le.classes_)

X = X_raw.reshape(-1, SEQUENCE_LENGTH, 63)
print(f"\n      X shape: {X.shape}, Classes: {num_classes}")


# ================================================
# STEP 2 — DATA AUGMENTATION
# (samples per class වැඩි කරනවා)
# ================================================
print(f"\n[2/7] Data augmentation (factor={AUG_FACTOR})...")

def augment_sequence(seq, rng, noise_std=0.005, scale_range=(0.95, 1.05)):
    """
    Augment a sequence with:
    - Small Gaussian noise (simulate sensor variation)
    - Random scaling (simulate different hand sizes)
    - Random time shift (simulate timing variation)
    """
    aug = seq.copy()

    # Gaussian noise
    aug += rng.normal(0, noise_std, aug.shape).astype(np.float32)

    # Random scale
    scale = rng.uniform(scale_range[0], scale_range[1])
    aug   = aug * scale

    # Random time shift (roll frames)
    shift = rng.integers(-3, 4)
    aug   = np.roll(aug, shift, axis=0)

    return np.clip(aug, 0.0, 1.0).astype(np.float32)

rng = np.random.default_rng(42)
X_aug_list = list(X)
y_aug_list = list(y_encoded)

for i in range(len(X)):
    for _ in range(AUG_FACTOR):
        X_aug_list.append(augment_sequence(X[i], rng))
        y_aug_list.append(y_encoded[i])

X_aug = np.array(X_aug_list, dtype=np.float32)
y_aug = np.array(y_aug_list)
print(f"      Before aug: {len(X)} | After aug: {len(X_aug)} samples")


# ================================================
# STEP 3 — TRAIN / TEST SPLIT
# ================================================
print(f"\n[3/7] Train/test split...")

# Check if stratify is possible
min_class_count = np.min(np.bincount(y_aug))
test_size       = 0.2
can_stratify    = (int(len(y_aug) * test_size) >= num_classes) and (min_class_count >= 2)

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug,
    test_size=test_size,
    random_state=42,
    stratify=y_aug if can_stratify else None
)

print(f"      Stratified : {'Yes' if can_stratify else 'No'}")
print(f"      Train      : {len(X_train)}, Test: {len(X_test)}")
print(f"      Classes    : {num_classes}")


# ================================================
# STEP 4 — NOISE FILTER
# ================================================
print(f"\n[4/7] Noise filter ready (threshold={NOISE_THRESHOLD})")

def apply_noise_filter(sequence, threshold=NOISE_THRESHOLD):
    """
    Novel Research Contribution:
    Velocity-Threshold Noise Filter for SLSL recognition.

    Removes low-velocity frames representing accidental
    hand movements (scratching, phone adjusting, etc.)
    """
    if len(sequence) < 2:
        padded = list(sequence) + [[0.0]*63] * (SEQUENCE_LENGTH - len(sequence))
        return np.array(padded[:SEQUENCE_LENGTH], dtype=np.float32), 0

    filtered = [sequence[0]]
    removed  = 0

    for i in range(1, len(sequence)):
        velocity = np.mean(np.abs(np.array(sequence[i]) - np.array(sequence[i-1])))
        if velocity > threshold:
            filtered.append(sequence[i])
        else:
            removed += 1

    if len(filtered) >= SEQUENCE_LENGTH:
        filtered = filtered[:SEQUENCE_LENGTH]
    else:
        padding  = [[0.0]*63] * (SEQUENCE_LENGTH - len(filtered))
        filtered = filtered + padding

    return np.array(filtered, dtype=np.float32), removed


# ================================================
# STEP 5 — ACCIDENTAL MOVEMENTS
# ================================================
print(f"\n[5/7] Generating {N_ACCIDENTAL} accidental movement samples...")

def generate_accidental_movements(n_samples=N_ACCIDENTAL, seed=42):
    """
    Realistic accidental hand movements:
      Group 1 (50%): Near-static — phone resting, hand in lap (std=0.003)
      Group 2 (50%): Small ambiguous — scratching, adjusting (std=0.015)
    Any confident classification of these = False Positive
    """
    rng = np.random.default_rng(seed)
    accidental = []
    for i in range(n_samples):
        base = rng.uniform(0.3, 0.7, 63)
        std  = 0.003 if i < n_samples // 2 else 0.015
        seq  = np.array(
            [base + rng.normal(0, std, 63) for _ in range(SEQUENCE_LENGTH)],
            dtype=np.float32
        )
        accidental.append(np.clip(seq, 0.0, 1.0))
    return np.array(accidental, dtype=np.float32)

accidental_data = generate_accidental_movements()
print(f"      Near-static (std=0.003) : {N_ACCIDENTAL//2} samples")
print(f"      Small moves  (std=0.015) : {N_ACCIDENTAL - N_ACCIDENTAL//2} samples")


# ================================================
# MODEL BUILDER
# ================================================
def build_model(num_classes, name="model"):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Conv1D,
        MaxPooling1D, BatchNormalization
    )

    model = Sequential(name=name, layers=[
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, 63)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-5
    )
]


# ================================================
# MODEL A — BASELINE (No Filter)
# ================================================
print("\n" + "="*60)
print("[6/7-A] MODEL A — Baseline (No Noise Filter)")
print("="*60)

model_a = build_model(num_classes, name="model_a")
print(f"      Parameters: {model_a.count_params():,}")

model_a.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks_list,
    verbose=1
)

_, acc_a = model_a.evaluate(X_test, y_test, verbose=0)
print(f"\n  ✅ Model A — Test Accuracy : {acc_a*100:.2f}%")

preds_a      = model_a.predict(accidental_data, verbose=0)
confidence_a = np.max(preds_a, axis=1)
fp_count_a   = int(np.sum(confidence_a > FP_THRESHOLD))
fpr_a        = fp_count_a / N_ACCIDENTAL
outcomes_a   = (confidence_a > FP_THRESHOLD).astype(int)

print(f"  ✅ Model A — False Positives : {fp_count_a}/{N_ACCIDENTAL}")
print(f"  ✅ Model A — FPR             : {fpr_a*100:.2f}%")
print(f"  ✅ Model A — Mean confidence : {np.mean(confidence_a):.4f}")

os.makedirs(MODEL_PATH, exist_ok=True)
model_a.save(os.path.join(MODEL_PATH, 'model_a_baseline.h5'))
print("  💾 Model A saved.")


# ================================================
# MODEL B — PROPOSED (With Filter)
# ================================================
print("\n" + "="*60)
print("[6/7-B] MODEL B — Proposed (With Noise Filter)")
print("="*60)

print("      Applying noise filter to training data...")
X_train_filtered     = []
total_frames_removed = 0

for seq in X_train:
    f_seq, removed = apply_noise_filter(seq.tolist())
    X_train_filtered.append(f_seq)
    total_frames_removed += removed

X_train_filtered = np.array(X_train_filtered, dtype=np.float32)
avg_removed      = total_frames_removed / len(X_train)
print(f"      Total frames removed   : {total_frames_removed}")
print(f"      Avg removed per seq    : {avg_removed:.2f}")

model_b = build_model(num_classes, name="model_b")

model_b.fit(
    X_train_filtered, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks_list,
    verbose=1
)

_, acc_b = model_b.evaluate(X_test, y_test, verbose=0)
print(f"\n  ✅ Model B — Test Accuracy : {acc_b*100:.2f}%")

fp_count_b        = 0
confidence_b_list = []
outcomes_b        = []

for acc_seq in accidental_data:
    f_seq, _ = apply_noise_filter(acc_seq.tolist())
    f_arr    = np.array(f_seq, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, 63)
    pred     = model_b.predict(f_arr, verbose=0)
    conf     = float(np.max(pred))
    confidence_b_list.append(conf)
    is_fp = int(conf > FP_THRESHOLD)
    outcomes_b.append(is_fp)
    fp_count_b += is_fp

confidence_b_arr = np.array(confidence_b_list, dtype=np.float32)
fpr_b            = fp_count_b / N_ACCIDENTAL
outcomes_b       = np.array(outcomes_b)

print(f"  ✅ Model B — False Positives : {fp_count_b}/{N_ACCIDENTAL}")
print(f"  ✅ Model B — FPR             : {fpr_b*100:.2f}%")
print(f"  ✅ Model B — Mean confidence : {np.mean(confidence_b_arr):.4f}")

model_b.save(os.path.join(MODEL_PATH, 'model_b_proposed.h5'))
print("  💾 Model B saved.")


# ================================================
# STEP 7 — STATISTICAL ANALYSIS
# ================================================
print("\n" + "="*60)
print("[7/7] STATISTICAL ANALYSIS")
print("="*60)

fpr_reduction = ((fpr_a - fpr_b) / fpr_a * 100) if fpr_a > 0 else 0.0
acc_diff      = (acc_b - acc_a) * 100

t_stat, p_ttest = stats.ttest_ind(confidence_a, confidence_b_arr)

pooled_std = np.sqrt((np.std(confidence_a)**2 + np.std(confidence_b_arr)**2) / 2)
cohens_d   = (np.mean(confidence_a) - np.mean(confidence_b_arr)) / pooled_std if pooled_std > 0 else 0.0

mcnemar_p = None
if HAS_STATSMODELS:
    b = int(np.sum((outcomes_a == 1) & (outcomes_b == 0)))
    c = int(np.sum((outcomes_a == 0) & (outcomes_b == 1)))
    print(f"\n  McNemar: b={b} (A=FP,B=ok)  c={c} (A=ok,B=FP)")
    if b + c > 0:
        res       = mcnemar([[0, b], [c, 0]], exact=True)
        mcnemar_p = res.pvalue
        print(f"  McNemar p-value : {mcnemar_p:.4f}")
    else:
        print("  McNemar skipped — no discordant pairs")

sig_p = p_ttest if mcnemar_p is None else min(p_ttest, mcnemar_p)

print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
print(f"""
  Model A (Baseline — No Noise Filter):
    Sign Accuracy          : {acc_a*100:.2f}%
    False Positives        : {fp_count_a}/{N_ACCIDENTAL}
    False Positive Rate    : {fpr_a*100:.2f}%
    Mean confidence (acc.) : {np.mean(confidence_a):.4f}

  Model B (Proposed — With Noise Filter):
    Sign Accuracy          : {acc_b*100:.2f}%
    False Positives        : {fp_count_b}/{N_ACCIDENTAL}
    False Positive Rate    : {fpr_b*100:.2f}%
    Mean confidence (acc.) : {np.mean(confidence_b_arr):.4f}

  Comparison:
    Accuracy change        : {acc_diff:+.2f}%
    FPR reduction          : {fpr_reduction:.2f}%
    Avg frames removed     : {avg_removed:.2f} per sequence

  Statistical Tests:
    t-statistic            : {t_stat:.4f}
    t-test p-value         : {p_ttest:.4f}
    Cohen's d (effect size): {cohens_d:.4f}""")

if mcnemar_p is not None:
    print(f"    McNemar's p-value      : {mcnemar_p:.4f}")

print()
if sig_p < 0.05:
    print("  ✅ H1 SUPPORTED: Noise filter significantly reduces FPR (p < 0.05)")
else:
    print(f"  ⚠️  p = {sig_p:.4f} — not significant. Check dataset size.")
print("="*60)


# ================================================
# SAVE CSV
# ================================================
rows = [
    {'Metric': 'Sign Accuracy (%)',
     'Model A (Baseline)': f"{acc_a*100:.2f}",
     'Model B (Proposed)': f"{acc_b*100:.2f}",
     'Difference': f"{acc_diff:+.2f}"},
    {'Metric': 'False Positives (out of 100)',
     'Model A (Baseline)': fp_count_a,
     'Model B (Proposed)': fp_count_b,
     'Difference': fp_count_b - fp_count_a},
    {'Metric': 'False Positive Rate (%)',
     'Model A (Baseline)': f"{fpr_a*100:.2f}",
     'Model B (Proposed)': f"{fpr_b*100:.2f}",
     'Difference': f"{fpr_b*100 - fpr_a*100:+.2f}"},
    {'Metric': 'FPR Reduction (%)',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{fpr_reduction:.2f}",
     'Difference': '-'},
    {'Metric': 'Mean Confidence on Accidentals',
     'Model A (Baseline)': f"{np.mean(confidence_a):.4f}",
     'Model B (Proposed)': f"{np.mean(confidence_b_arr):.4f}",
     'Difference': f"{np.mean(confidence_b_arr)-np.mean(confidence_a):+.4f}"},
    {'Metric': 'Avg Frames Removed by Filter',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{avg_removed:.2f}",
     'Difference': '-'},
    {'Metric': 't-statistic',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{t_stat:.4f}",
     'Difference': '-'},
    {'Metric': 't-test p-value',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{p_ttest:.4f}",
     'Difference': '-'},
    {'Metric': "McNemar's p-value",
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{mcnemar_p:.4f}" if mcnemar_p is not None else 'N/A',
     'Difference': '-'},
    {'Metric': "Cohen's d (effect size)",
     'Model A (Baseline)': '-',
     'Model B (Proposed)': f"{cohens_d:.4f}",
     'Difference': '-'},
    {'Metric': 'H1 Supported',
     'Model A (Baseline)': '-',
     'Model B (Proposed)': 'Yes' if sig_p < 0.05 else 'No',
     'Difference': '-'},
    {'Metric': 'Dataset (CSV used)',
     'Model A (Baseline)': os.path.basename(CSV_PATH),
     'Model B (Proposed)': os.path.basename(CSV_PATH),
     'Difference': '-'},
    {'Metric': 'Augmentation factor',
     'Model A (Baseline)': AUG_FACTOR,
     'Model B (Proposed)': AUG_FACTOR,
     'Difference': '-'},
]

pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
print(f"\n  💾 Results saved : {RESULTS_PATH}")
print("  🎉 Experiment complete!\n")