#!/usr/bin/env python3
"""
Live Speech Recognition using fine-tuned Whisper.

Simplified recognizer for continuous/live transcription.
Designed to help recognize speech that standard models struggle with.
"""

import os
import sys
import warnings

import numpy as np
import librosa
import torch

warnings.filterwarnings('ignore')

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz
FINETUNED_MODEL_DIR = "./whisper-finetuned"


class LiveSpeechRecognizer:
    """Live speech recognition using fine-tuned Whisper."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or FINETUNED_MODEL_DIR
        self.processor = None
        self.model = None
        self.device = None

    def load_model(self):
        """Load the fine-tuned Whisper model."""
        if self.model is not None:
            return

        # Check if fine-tuned model exists, otherwise use base model
        if os.path.exists(self.model_dir):
            print(f"Loading fine-tuned model from {self.model_dir}...")
            model_path = self.model_dir
        else:
            print("Fine-tuned model not found, using base whisper-tiny...")
            model_path = "openai/whisper-tiny"

        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

        # Move to appropriate device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}!")

    def transcribe(self, audio_data, sample_rate=SAMPLE_RATE):
        """Transcribe audio chunk."""
        self.load_model()

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )

        audio_data = audio_data.astype(np.float32)

        # Check if audio has enough energy
        energy = np.sqrt(np.mean(audio_data**2))
        if energy < 0.005:
            return ""

        # Get input features
        inputs = self.processor.feature_extractor(
            audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        # Create attention mask
        attention_mask = torch.ones(input_features.shape[:-1], dtype=torch.long, device=self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language="en",
                task="transcribe",
                suppress_tokens=[],
                begin_suppress_tokens=[220, 50257],
            )

        transcription = self.processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription.strip()


def main():
    """Test the recognizer."""
    recognizer = LiveSpeechRecognizer()
    recognizer.load_model()
    print("\nRecognizer ready for live transcription!")


if __name__ == "__main__":
    main()
