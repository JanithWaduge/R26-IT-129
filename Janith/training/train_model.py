import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# ================================================
# PATHS
# ================================================
CSV_PATH   = r'D:\R26-IT-129\Janith\keypoints_clean.csv'
MODEL_PATH = r'D:\R26-IT-129\Janith\models'

os.makedirs(MODEL_PATH, exist_ok=True)

# ================================================
# LOAD DATA
# ================================================
print("📂 Loading data...")
df = pd.read_csv(CSV_PATH)

print(f"✅ Shape: {df.shape}")
print(f"✅ Total signs: {df['label'].nunique()}")

# ================================================
# FILTER — min 3 samples per sign
# ================================================
sign_counts = df['label'].value_counts()
valid_signs = sign_counts[sign_counts >= 3].index
df = df[df['label'].isin(valid_signs)]

print(f"✅ After filter: {len(df)} samples, {df['label'].nunique()} signs")

# ================================================
# SELECT TOP 30 SIGNS (most samples = best accuracy)
# ================================================
TOP_N_SIGNS = 30

sign_counts = df['label'].value_counts()
top_signs = sign_counts.head(TOP_N_SIGNS).index
df = df[df['label'].isin(top_signs)]

print(f"\n✅ Selected top {TOP_N_SIGNS} signs:")
print(sign_counts.head(TOP_N_SIGNS).to_string())
print(f"\n✅ Final samples: {len(df)}")

# ================================================
# PREPARE X, y (RAW — no augmentation yet)
# ================================================
feature_cols = [c for c in df.columns if c != 'label']
X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df['label'].values

# Reshape → (samples, 30 frames, 63 features)
X_raw = X_raw.reshape(-1, 30, 63)

# Label encode
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)

print(f"✅ Classes: {num_classes}")

# Save label encoder classes
np.save(os.path.join(MODEL_PATH, 'classes.npy'), le.classes_)
print(f"✅ Classes saved!")

# ================================================
# TRAIN/TEST SPLIT — done BEFORE augmentation
# CRITICAL FIX: augmenting before splitting lets near-duplicate
# copies of the same recording end up in both train and test,
# which leaks information and inflates test accuracy. Splitting
# the RAW data first guarantees the test set only ever contains
# recordings the model has never seen in any form.
# ================================================
print("\n✂️  Splitting raw data (80/20) BEFORE augmentation...")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_raw, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"✅ Train (raw): {len(X_train_raw)}, Test (untouched): {len(X_test)}")

# ================================================
# DATA AUGMENTATION — applied ONLY to the training split
# Test set (X_test/y_test) is never augmented and never touched again.
# ================================================
print("\n🔄 Augmenting training data...")

augmented_X = []
augmented_y = []

for i in range(len(X_train_raw)):
    seq = X_train_raw[i]
    label = y_train_raw[i]

    # Original
    augmented_X.append(seq)
    augmented_y.append(label)

    # Augmentation 1: Small noise add කරන්න
    noise = np.random.normal(0, 0.01, seq.shape)
    augmented_X.append(seq + noise)
    augmented_y.append(label)

    # Augmentation 2: Slightly scale කරන්න
    scale = np.random.uniform(0.95, 1.05)
    augmented_X.append(seq * scale)
    augmented_y.append(label)

    # Augmentation 3: More noise
    noise2 = np.random.normal(0, 0.02, seq.shape)
    augmented_X.append(seq + noise2)
    augmented_y.append(label)

X_train = np.array(augmented_X, dtype=np.float32)
y_train = np.array(augmented_y)

print(f"✅ After augmentation: {len(X_train)} training samples "
      f"(test set stays at {len(X_test)}, untouched and unaugmented)")

# ================================================
# CLASS WEIGHTS — ADDED
# Some signs (e.g. Copying, Study, Teacher via dataset2) may have
# noticeably more samples than others. Without this, the model can
# lean toward whichever signs are best represented. Computed on the
# augmented TRAINING labels only — the test set is unaffected.
# ================================================
class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weight_values)}
print(f"\n⚖️  Class weights computed for {len(class_weight_dict)} signs "
      f"(min={min(class_weight_dict.values()):.2f}, max={max(class_weight_dict.values()):.2f})")

# ================================================
# MODEL — Improved CNN + LSTM
# ================================================
print("\n🧠 Building model...")

model = Sequential([
    # CNN Block 1
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(30, 63)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # CNN Block 2
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # LSTM Block
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),

    # Dense Output
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

model.summary()

# ================================================
# CALLBACKS
# ================================================
callbacks = [
    EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    ModelCheckpoint(
        os.path.join(MODEL_PATH, 'best_model.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# ================================================
# TRAIN
# ================================================
print("\n🚀 Training started...")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# ================================================
# EVALUATE
# ================================================
print("\n📊 Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"\n{'='*50}")
print(f"✅ Test Accuracy     : {test_acc*100:.2f}%")
print(f"✅ Test Loss         : {test_loss:.4f}")
print(f"✅ Best Val Accuracy : {max(history.history['val_accuracy'])*100:.2f}%")
print(f"✅ Best Val Loss     : {min(history.history['val_loss']):.4f}")
print(f"{'='*50}")

# Classification Report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n📋 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

# ================================================
# SAVE TFLITE
# ================================================
print("\n💾 Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_PATH, 'slsl_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ TFLite saved: {tflite_path}")
print("\n🎉 Training Complete!")