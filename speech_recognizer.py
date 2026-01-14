#!/usr/bin/env python3
"""
Speech Recognition Helper

End-to-end speech-to-text using a fine-tuned Whisper model.
Designed to help recognize speech that standard models struggle with,
such as children with speech difficulties, accented speech, or unique vocabularies.

The model learns directly from examples - no rule-based post-processing.
"""

import os
import sys
import json
import re
import tempfile
import subprocess
import warnings
from datetime import datetime

import numpy as np
import librosa
import torch

warnings.filterwarnings('ignore')

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz
FINETUNED_MODEL_DIR = "./whisper-finetuned"
CORRECTIONS_DIR = "./Corrections"
RECORDINGS_DIR = "./Recordings"

# Phrases to filter out from transcriptions (e.g., prompts you say to elicit speech)
# Add phrases you commonly say before the target speaker talks
IGNORE_PHRASES = [
    # Example: "what do you want",
    # Example: "say it again",
]


class SpeechRecognizer:
    """End-to-end speech recognition using fine-tuned Whisper."""

    def __init__(self):
        self.finetuned_processor = None
        self.finetuned_model = None
        self.device = None

        # Speaker recognition (optional)
        self.voice_encoder = None
        self.enrolled_embedding = None
        self.voice_file = "enrolled_voice.npy"
        self.speaker_threshold = 0.75

        # Ensure corrections directory exists
        os.makedirs(CORRECTIONS_DIR, exist_ok=True)

    def load_model(self):
        """Load the fine-tuned Whisper model."""
        if self.finetuned_model is not None:
            return

        print(f"Loading fine-tuned Whisper model from {FINETUNED_MODEL_DIR}...")
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        self.finetuned_processor = WhisperProcessor.from_pretrained(FINETUNED_MODEL_DIR)
        self.finetuned_model = WhisperForConditionalGeneration.from_pretrained(FINETUNED_MODEL_DIR)

        # Move to appropriate device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.finetuned_model = self.finetuned_model.to(self.device)
        self.finetuned_model.eval()
        print(f"Model loaded on {self.device}!")

    def load_voice_encoder(self):
        """Load the voice encoder for speaker recognition."""
        if self.voice_encoder is None:
            print("Loading voice encoder...")
            from resemblyzer import VoiceEncoder
            self.voice_encoder = VoiceEncoder()
            print("Voice encoder loaded!")
        return self.voice_encoder

    def load_enrolled_voice(self):
        """Load previously enrolled voice profile."""
        if os.path.exists(self.voice_file):
            self.enrolled_embedding = np.load(self.voice_file)
            print("Enrolled voice profile loaded")
            return True
        return False

    def enroll_voice(self, audio_samples):
        """Enroll a speaker's voice from audio samples."""
        from resemblyzer import preprocess_wav

        encoder = self.load_voice_encoder()
        embeddings = []

        for audio_data, sample_rate in audio_samples:
            if sample_rate != SAMPLE_RATE:
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE
                )

            wav = preprocess_wav(audio_data, SAMPLE_RATE)
            if len(wav) > 0:
                embedding = encoder.embed_utterance(wav)
                embeddings.append(embedding)

        if embeddings:
            self.enrolled_embedding = np.mean(embeddings, axis=0)
            np.save(self.voice_file, self.enrolled_embedding)
            print(f"Voice enrolled ({len(embeddings)} samples)")
            return True
        return False

    def enroll_from_recordings(self):
        """Auto-enroll voice from recordings folder."""
        if not os.path.exists(RECORDINGS_DIR):
            return False

        print("Auto-enrolling voice from recordings...")
        audio_samples = []

        for filename in os.listdir(RECORDINGS_DIR):
            if not filename.endswith(('.m4a', '.wav')):
                continue

            filepath = os.path.join(RECORDINGS_DIR, filename)
            try:
                audio_data, sr = self._load_audio_file(filepath)
                audio_samples.append((audio_data, sr))
                print(f"  Added: {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")

        if audio_samples:
            return self.enroll_voice(audio_samples)
        return False

    def _load_audio_file(self, filepath):
        """Load an audio file, converting m4a if needed."""
        import platform

        if filepath.endswith('.m4a'):
            # Convert m4a to wav
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()
            try:
                if platform.system() == "Darwin":
                    # macOS - use built-in afconvert (no extra install needed)
                    subprocess.run([
                        'afconvert', '-f', 'WAVE', '-d', 'LEI16',
                        filepath, temp_wav.name
                    ], check=True, capture_output=True)
                else:
                    # Linux/Windows - use ffmpeg
                    subprocess.run([
                        'ffmpeg', '-i', filepath, '-ar', str(SAMPLE_RATE),
                        '-ac', '1', '-y', temp_wav.name
                    ], check=True, capture_output=True)
                audio_data, sr = librosa.load(temp_wav.name, sr=SAMPLE_RATE)
            finally:
                if os.path.exists(temp_wav.name):
                    os.unlink(temp_wav.name)
        else:
            audio_data, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        return audio_data, sr

    def extract_enrolled_speaker_audio(self, audio_data, sample_rate):
        """Extract only audio matching the enrolled speaker."""
        from resemblyzer import preprocess_wav

        if self.enrolled_embedding is None:
            return audio_data, []

        encoder = self.load_voice_encoder()

        if sample_rate != SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
            sample_rate = SAMPLE_RATE

        # Segment parameters
        segment_samples = int(1.0 * sample_rate)
        hop_samples = int(0.5 * sample_rate)

        matched_segments = []
        position = 0

        while position + segment_samples <= len(audio_data):
            segment = audio_data[position:position + segment_samples]
            energy = np.sqrt(np.mean(segment**2))

            if energy > 0.01:
                wav = preprocess_wav(segment, sample_rate)
                if len(wav) > 0:
                    embedding = encoder.embed_utterance(wav)
                    similarity = np.dot(embedding, self.enrolled_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(self.enrolled_embedding)
                    )
                    if similarity >= self.speaker_threshold:
                        matched_segments.append((position, position + segment_samples, similarity))

            position += hop_samples

        if not matched_segments:
            return np.array([]), []

        # Merge overlapping segments
        merged = []
        current_start, current_end, current_sim = matched_segments[0]

        for start, end, sim in matched_segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                current_sim = max(current_sim, sim)
            else:
                merged.append((current_start, current_end, current_sim))
                current_start, current_end, current_sim = start, end, sim
        merged.append((current_start, current_end, current_sim))

        # Extract matched audio
        filtered_audio = []
        segments_info = []

        for start_sample, end_sample, sim in merged:
            filtered_audio.append(audio_data[start_sample:end_sample])
            segments_info.append({
                'start': round(start_sample / sample_rate, 2),
                'end': round(end_sample / sample_rate, 2),
                'similarity': round(float(sim), 2)
            })

        if filtered_audio:
            silence = np.zeros(int(0.1 * sample_rate))
            result = []
            for i, seg in enumerate(filtered_audio):
                result.append(seg)
                if i < len(filtered_audio) - 1:
                    result.append(silence)
            filtered_audio = np.concatenate(result)
        else:
            filtered_audio = np.array([])

        return filtered_audio, segments_info

    def transcribe(self, audio_data, sample_rate):
        """Transcribe audio using the fine-tuned model."""
        self.load_model()

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )

        audio_data = audio_data.astype(np.float32)

        # Get input features
        input_features = self.finetuned_processor.feature_extractor(
            audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.finetuned_model.generate(
                input_features,
                language="en",
                task="transcribe",
            )

        transcription = self.finetuned_processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription.strip()

    def filter_ignored_phrases(self, text):
        """Remove ignored phrases (like prompts) from transcription."""
        text_lower = text.lower().strip()

        for phrase in IGNORE_PHRASES:
            # Remove the phrase if it appears
            text_lower = re.sub(
                re.escape(phrase) + r'[,.]?\s*',
                '',
                text_lower,
                flags=re.IGNORECASE
            )

        return text_lower.strip()

    def recognize(self, audio_data, sample_rate, filter_speaker=True) -> dict:
        """
        Recognize speech from audio.

        Returns dict with:
            - transcription: What the model recognized
            - speaker_segments: Info about detected speaker segments
            - error: Error message if any
        """
        segments_info = []
        audio_to_transcribe = audio_data

        # Filter to enrolled speaker if enabled
        if filter_speaker and self.enrolled_embedding is not None:
            filtered_audio, segments_info = self.extract_enrolled_speaker_audio(
                audio_data, sample_rate
            )

            if len(filtered_audio) == 0:
                return {
                    'transcription': '',
                    'speaker_segments': [],
                    'error': 'Could not detect enrolled speaker. Make sure the target speaker is speaking.'
                }

            audio_to_transcribe = filtered_audio
            sample_rate = SAMPLE_RATE

        # Transcribe
        raw_transcription = self.transcribe(audio_to_transcribe, sample_rate)

        if not raw_transcription:
            return {
                'transcription': '',
                'speaker_segments': segments_info,
                'error': 'Could not transcribe audio'
            }

        # Filter out prompts
        transcription = self.filter_ignored_phrases(raw_transcription)

        return {
            'transcription': transcription,
            'raw_transcription': raw_transcription,
            'speaker_segments': segments_info
        }

    def save_correction(self, audio_data, sample_rate, correct_text):
        """
        Save a correction for future retraining.

        Args:
            audio_data: The audio that was misrecognized
            sample_rate: Sample rate of the audio
            correct_text: What it should have been recognized as
        """
        from scipy.io import wavfile

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = re.sub(r'[^\w\s-]', '', correct_text).strip().replace(' ', '_')
        filename = f"{safe_text}_{timestamp}"

        # Save audio as WAV
        wav_path = os.path.join(CORRECTIONS_DIR, f"{filename}.wav")
        if sample_rate != SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
        wavfile.write(wav_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))

        # Save metadata
        meta_path = os.path.join(CORRECTIONS_DIR, f"{filename}.json")
        metadata = {
            'correct_text': correct_text.lower().strip(),
            'timestamp': timestamp,
            'sample_rate': SAMPLE_RATE
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved correction: {wav_path}")
        return True

    def get_correction_count(self):
        """Get the number of pending corrections."""
        if not os.path.exists(CORRECTIONS_DIR):
            return 0
        return len([f for f in os.listdir(CORRECTIONS_DIR) if f.endswith('.wav')])

    def retrain_model(self):
        """
        Retrain the model with corrections and original recordings.

        Returns dict with status and message.
        """
        # Check if there are corrections or recordings
        corrections_count = self.get_correction_count()
        recordings_exist = os.path.exists(RECORDINGS_DIR) and any(
            f.endswith(('.m4a', '.wav')) for f in os.listdir(RECORDINGS_DIR)
        )

        if corrections_count == 0 and not recordings_exist:
            return {
                'success': False,
                'message': 'No training data available. Add corrections or recordings first.'
            }

        print(f"Starting retraining with {corrections_count} corrections...")

        try:
            # Run the fine-tuning script
            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore'

            result = subprocess.run(
                [sys.executable, 'finetune_whisper.py'],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env
            )

            # Check for success
            if 'Fine-tuning complete' in result.stdout or result.returncode == 0:
                # Reload the model
                self.finetuned_model = None
                self.finetuned_processor = None
                self.load_model()

                return {
                    'success': True,
                    'message': f'Model retrained successfully with {corrections_count} corrections!'
                }
            else:
                # Get the last meaningful error line
                stderr_lines = [l for l in result.stderr.split('\n') if l.strip() and 'Warning' not in l]
                error_msg = stderr_lines[-1] if stderr_lines else result.stderr[:200]
                return {
                    'success': False,
                    'message': f'Retraining failed: {error_msg}'
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'message': 'Retraining timed out (exceeded 1 hour)'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Retraining error: {str(e)}'
            }


# For backward compatibility
SpeechTranslator = SpeechRecognizer


def main():
    """Command-line interface for testing."""
    recognizer = SpeechRecognizer()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test' and len(sys.argv) > 2:
            test_file = sys.argv[2]
            audio_data, sr = recognizer._load_audio_file(test_file)
            result = recognizer.recognize(audio_data, sr, filter_speaker=False)

            print(f"\nTranscription: '{result['transcription']}'")
            if result.get('error'):
                print(f"Error: {result['error']}")

        elif sys.argv[1] == 'retrain':
            result = recognizer.retrain_model()
            print(result['message'])

        else:
            print("Usage:")
            print("  python speech_recognizer.py test <file>  - Test transcription")
            print("  python speech_recognizer.py retrain      - Retrain model")
    else:
        print("Speech Recognizer (End-to-End)")
        print("=" * 50)
        print("\nUsage:")
        print("  python speech_recognizer.py test <file>")
        print("  python speech_recognizer.py retrain")


if __name__ == "__main__":
    main()
