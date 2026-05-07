"""
prepare_finetune_data.py
Prepares the OpenSLR52 Sinhala dataset for Whisper fine-tuning.
Creates train.csv and test.csv in ./sinhala_data/
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa

# ── Step 1: Load TSV ─────────────────────────────────────────────────────────
df = pd.read_csv(
    "./voicedataset/utt_spk_text.tsv",
    sep="\t",
    header=None,
    names=["file_id", "speaker_id", "transcription"]
)
print(f"TSV loaded: {len(df)} rows")

# ── Step 2: Build audio paths ────────────────────────────────────────────────
def get_audio_path(file_id):
    prefix = file_id[:2]
    return (
        f"./voicedataset/asr_sinhala_0/"
        f"asr_sinhala/data/{prefix}/{file_id}.flac"
    )

df["audio_path"] = df["file_id"].apply(get_audio_path)

# ── Step 3: Filter to existing files only ────────────────────────────────────
df["exists"] = df["audio_path"].apply(os.path.exists)
print(f"Files found:   {df['exists'].sum()}")
print(f"Files missing: {(~df['exists']).sum()}")
df = df[df["exists"] == True]

# ── Step 4: Sample 2000 rows ─────────────────────────────────────────────────
df_sample = df.sample(n=2000, random_state=42)

# ── Step 5: 80/20 train/test split ───────────────────────────────────────────
train_df, test_df = train_test_split(
    df_sample,
    test_size=0.2,
    random_state=42
)

# ── Step 6: Save CSVs ────────────────────────────────────────────────────────
os.makedirs("./sinhala_data", exist_ok=True)

train_df[["audio_path", "transcription"]].to_csv(
    "./sinhala_data/train.csv", index=False
)
test_df[["audio_path", "transcription"]].to_csv(
    "./sinhala_data/test.csv", index=False
)

print("\n=== DATASET READY ===")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples:  {len(test_df)}")
print("\nSample training rows:")
print(train_df[["audio_path", "transcription"]].head(3).to_string())

# ── Step 7: Verify one audio file loads ──────────────────────────────────────
sample_audio = train_df["audio_path"].iloc[0]
audio, sr = librosa.load(sample_audio, sr=16000)
print(f"\nSample audio loaded: {sample_audio}")
print(f"Duration:    {len(audio)/sr:.2f} seconds")
print(f"Sample rate: {sr} Hz")
