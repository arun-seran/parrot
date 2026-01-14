#!/usr/bin/env python3
"""
Fine-tune Whisper on custom speech recordings.

This creates an end-to-end model that directly maps speech to text,
learning from your specific examples rather than relying on post-processing.

Use this for:
- Children with speech difficulties
- Accented speech
- Unique vocabularies or terminology
- Any speech that standard models struggle with
"""

import os
import json
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import librosa
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

warnings.filterwarnings('ignore')

# Configuration
SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-tiny"  # tiny model for faster fine-tuning
OUTPUT_DIR = "./whisper-finetuned"
RECORDINGS_DIR = "Recordings"
CORRECTIONS_DIR = "Corrections"


def load_recordings(recordings_dir):
    """Load all recordings and their labels from Recordings folder.

    Expected format: filename contains the label (e.g., "hello.wav" -> "hello")
    """
    data = []

    if not os.path.exists(recordings_dir):
        print(f"Recordings directory not found: {recordings_dir}")
        return data

    for filename in os.listdir(recordings_dir):
        if not (filename.endswith('.m4a') or filename.endswith('.wav')):
            continue

        filepath = os.path.join(recordings_dir, filename)

        # Extract label from filename (remove extension and numbers)
        label = os.path.splitext(filename)[0]
        label = ''.join([c for c in label if not c.isdigit()]).strip()

        data.append({
            'audio_path': filepath,
            'text': label.lower()  # lowercase for consistency
        })

    print(f"Found {len(data)} recordings from {recordings_dir}")
    return data


def load_corrections(corrections_dir):
    """Load corrections from Corrections folder (wav files with json metadata)."""
    data = []

    if not os.path.exists(corrections_dir):
        print(f"Corrections directory not found: {corrections_dir}")
        return data

    for filename in os.listdir(corrections_dir):
        if not filename.endswith('.wav'):
            continue

        wav_path = os.path.join(corrections_dir, filename)
        json_path = os.path.join(corrections_dir, filename.replace('.wav', '.json'))

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            label = metadata.get('correct_text', '').lower().strip()
        else:
            # Fall back to filename
            label = os.path.splitext(filename)[0]
            label = label.rsplit('_', 1)[0]  # Remove timestamp
            label = label.replace('_', ' ').lower().strip()

        if label:
            data.append({
                'audio_path': wav_path,
                'text': label
            })

    print(f"Found {len(data)} corrections from {corrections_dir}")
    return data


def load_all_training_data():
    """Load both recordings and corrections."""
    recordings = load_recordings(RECORDINGS_DIR)
    corrections = load_corrections(CORRECTIONS_DIR)

    all_data = recordings + corrections
    print(f"\nTotal training samples: {len(all_data)}")

    return all_data


def convert_m4a_to_wav(m4a_path):
    """Convert m4a to wav using afconvert (macOS) or ffmpeg (other platforms)."""
    import tempfile
    import subprocess
    import platform

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()

    if platform.system() == "Darwin":
        # macOS - use built-in afconvert (no extra install needed)
        subprocess.run([
            'afconvert', '-f', 'WAVE', '-d', 'LEI16',
            m4a_path, temp_wav.name
        ], check=True, capture_output=True)
    else:
        # Linux/Windows - use ffmpeg
        subprocess.run([
            'ffmpeg', '-i', m4a_path,
            '-ar', str(SAMPLE_RATE), '-ac', '1',
            '-y', temp_wav.name
        ], check=True, capture_output=True)

    return temp_wav.name


def load_audio(audio_path):
    """Load audio file and return numpy array."""
    if audio_path.endswith('.m4a'):
        wav_path = convert_m4a_to_wav(audio_path)
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        os.unlink(wav_path)
    else:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    return audio


def augment_audio(audio, sr=SAMPLE_RATE):
    """Create augmented versions of audio for better generalization."""
    augmented = [audio]

    # Time stretch
    for rate in [0.9, 0.95, 1.05, 1.1]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        augmented.append(stretched)

    # Pitch shift
    for steps in [-2, -1, 1, 2]:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        augmented.append(shifted)

    # Add noise
    for noise_level in [0.002, 0.005, 0.01]:
        noise = np.random.normal(0, noise_level, len(audio))
        noisy = audio + noise
        augmented.append(noisy.astype(np.float32))

    # Volume variations
    for gain in [0.7, 0.85, 1.15, 1.3]:
        scaled = audio * gain
        scaled = np.clip(scaled, -1, 1)
        augmented.append(scaled.astype(np.float32))

    return augmented


def prepare_dataset(augment=True):
    """Prepare dataset for training from all sources."""
    recordings = load_all_training_data()

    audio_data = []
    texts = []

    for rec in recordings:
        print(f"  Loading: {rec['audio_path']} -> '{rec['text']}'")
        audio = load_audio(rec['audio_path'])

        if augment:
            # Create augmented versions
            augmented = augment_audio(audio)
            for aug in augmented:
                audio_data.append(aug)
                texts.append(rec['text'])
        else:
            audio_data.append(audio)
            texts.append(rec['text'])

    print(f"\nTotal samples after augmentation: {len(audio_data)}")

    # Create dataset
    dataset = Dataset.from_dict({
        'audio': audio_data,
        'text': texts,
    })

    return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper fine-tuning."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio and text
        audio_features = [f["input_features"] for f in features]
        label_features = [f["labels"] for f in features]

        # Pad audio features
        batch = self.processor.feature_extractor.pad(
            {"input_features": audio_features},
            return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt"
        )

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def preprocess_function(examples, processor):
    """Preprocess audio and text for training."""
    # Process audio
    audio = examples["audio"]

    # Get input features
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np"
    ).input_features[0]

    # Tokenize text
    labels = processor.tokenizer(examples["text"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels
    }


def compute_metrics(pred, processor):
    """Compute accuracy metric."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate accuracy
    correct = sum(1 for p, l in zip(pred_str, label_str) if p.strip().lower() == l.strip().lower())
    accuracy = correct / len(pred_str) if pred_str else 0

    return {"accuracy": accuracy}


def main():
    print("="*60)
    print("  Fine-tuning Whisper for Custom Speech Recognition")
    print("="*60)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load processor and model
    print(f"\nLoading {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Configure model for English transcription
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.suppress_tokens = []

    # Prepare dataset (from Recordings + Corrections)
    print("\nPreparing dataset with augmentation...")
    dataset = prepare_dataset(augment=True)

    # Preprocess dataset
    print("\nPreprocessing dataset...")
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, processor),
        remove_columns=dataset.column_names,
    )

    # Split into train/eval (90/10)
    split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_steps=25,
        max_steps=300,
        gradient_checkpointing=False,  # Disable for MPS compatibility
        fp16=False,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=50,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    # Train
    print("\n" + "="*60)
    print("  Starting fine-tuning...")
    print("="*60 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved to {OUTPUT_DIR}")
    print("\nFine-tuning complete!")

    # Test the model
    print("\n" + "="*60)
    print("  Testing fine-tuned model")
    print("="*60)

    model.eval()
    test_recordings = load_recordings(RECORDINGS_DIR)[:5]

    for rec in test_recordings:
        audio = load_audio(rec['audio_path'])

        input_features = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        transcription = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        expected = rec['text']
        match = "OK" if transcription.strip().lower() == expected.lower() else "MISS"
        print(f"  [{match}] Expected: '{expected}' | Got: '{transcription.strip()}'")


if __name__ == "__main__":
    main()
