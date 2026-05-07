"""
fine_tune_whisper.py
Fine-tunes OpenAI Whisper-small on the OpenSLR52 Sinhala dataset.
Optimized for 4-6 GB laptop GPU.

Run from: gimhana/
    python fine_tune_whisper.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List

# ── Pre-training checks ───────────────────────────────────────────────────────
print("=" * 55)
print("  PRE-TRAINING CHECKS")
print("=" * 55)

# 1. CUDA
cuda_available = torch.cuda.is_available()
print(f"1. CUDA available     : {cuda_available}")

# 2. GPU name and VRAM
if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"2. GPU name           : {gpu_name}")
    print(f"   VRAM               : {vram_gb:.1f} GB")
else:
    print("2. GPU                : Not available — training on CPU (very slow)")

# 3. Training sample count
train_df = pd.read_csv("./sinhala_data/train.csv")
test_df  = pd.read_csv("./sinhala_data/test.csv")
print(f"3. Training samples   : {len(train_df)}")
print(f"   Test samples       : {len(test_df)}")

# 4. Estimated time
steps_per_epoch = len(train_df) // (4 * 4)   # batch=4, accum=4
total_steps     = steps_per_epoch * 5
if cuda_available:
    est_minutes = total_steps * 0.9           # ~0.9 s/step on 4-6 GB GPU
    est_label   = f"~{est_minutes/60:.0f} min  ({est_minutes:.0f} s total)"
else:
    est_label   = "~10-20 hours (CPU — not recommended)"
print(f"4. Estimated time     : {est_label}")
print(f"   Steps per epoch    : {steps_per_epoch}")
print(f"   Total steps        : {total_steps}")
print("=" * 55)
print()

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset
import evaluate

# ── Step 1: Load model and processor ─────────────────────────────────────────
print("=== Step 1: Loading Whisper-small ===")
MODEL_NAME = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="sinhala",
    task="transcribe",
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = "sinhala"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Gradient checkpointing saves ~30% VRAM at a small speed cost
model.config.use_cache = False
model.gradient_checkpointing_enable()
print("Model loaded. Gradient checkpointing enabled.")

# ── Step 2: Build HuggingFace Datasets ───────────────────────────────────────
print("\n=== Step 2: Preparing dataset ===")

def prepare_sample(batch):
    """Load FLAC audio + tokenize transcription for one sample."""
    audio, _ = librosa.load(batch["audio_path"], sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Processing training set (may take a few minutes)...")
train_dataset = Dataset.from_pandas(
    train_df[["audio_path", "transcription"]].reset_index(drop=True)
)
train_dataset = train_dataset.map(
    prepare_sample,
    remove_columns=["audio_path", "transcription"],
)

print("Processing test set...")
test_dataset = Dataset.from_pandas(
    test_df[["audio_path", "transcription"]].reset_index(drop=True)
)
test_dataset = test_dataset.map(
    prepare_sample,
    remove_columns=["audio_path", "transcription"],
)

print(f"Train dataset : {len(train_dataset)} samples ready")
print(f"Test dataset  : {len(test_dataset)} samples ready")

# ── Step 3: Data collator ─────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip leading BOS token if present
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ── Step 4: WER metric ────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}

# ── Step 5: Training arguments ────────────────────────────────────────────────
print("\n=== Step 3: Training configuration ===")
use_fp16 = cuda_available

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/whisper-sinhala",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch size = 16
    learning_rate=1e-5,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    fp16=use_fp16,
    predict_with_generate=True,
    generation_max_length=225,
    report_to=["none"],
    dataloader_num_workers=0,           # required on Windows
    remove_unused_columns=False,
)

print(f"Epochs         : {training_args.num_train_epochs}")
print(f"Batch size     : {training_args.per_device_train_batch_size}  (effective = 16 with grad accum)")
print(f"Learning rate  : {training_args.learning_rate}")
print(f"FP16           : {use_fp16}")
print(f"Output dir     : {training_args.output_dir}")

# ── Step 6: Train ─────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("\n" + "=" * 55)
print("  STARTING FINE-TUNING")
print("=" * 55)
trainer.train()

# ── Step 7: Save fine-tuned model ─────────────────────────────────────────────
SAVE_DIR = "./models/whisper-sinhala-finetuned"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"\nModel saved → {SAVE_DIR}")

# ── Step 8: Quick inference test ──────────────────────────────────────────────
print("\n=== Quick test ===")
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model=SAVE_DIR,
    device=0 if cuda_available else -1,
)
test_audio = pd.read_csv("./sinhala_data/test.csv")["audio_path"].iloc[0]
result = asr(test_audio)
print(f"Test audio : {test_audio}")
print(f"Result     : {result['text']}")
print("\nDone!")
