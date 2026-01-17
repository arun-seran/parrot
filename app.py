#!/usr/bin/env python3
"""
Parrot - Live Transcription App

Continuous speech-to-text transcription using fine-tuned Whisper.
Designed to help recognize speech that standard models struggle with.

Works on MacBook and iPhone (with HTTPS for mobile microphone access).
Includes training interface to record, label, and fine-tune.
"""

import os
import sys
import socket
import base64
import json
import subprocess
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from flask import Flask, render_template, request, jsonify
from speech_recognizer import LiveSpeechRecognizer, SAMPLE_RATE

app = Flask(__name__)
recognizer = None

# Recordings directory
RECORDINGS_DIR = "./Recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def get_local_ip():
    """Get the local IP address for network access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def init_recognizer():
    """Initialize the recognizer."""
    global recognizer

    if recognizer is not None:
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Initializing Live Speech Recognizer...")
    recognizer = LiveSpeechRecognizer()
    recognizer.load_model()
    print("Ready for live transcription!")


# Initialize on module import
init_recognizer()


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe an audio chunk."""
    try:
        data = request.json
        audio_base64 = data.get('audio')

        if not audio_base64:
            return jsonify({'error': 'No audio data received'}), 400

        # Decode base64 audio (PCM float32 from client)
        audio_bytes = base64.b64decode(audio_base64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        sample_rate = data.get('sampleRate', 44100)

        # Transcribe
        text = recognizer.transcribe(audio_data, sample_rate)

        return jsonify({
            'text': text,
            'is_final': data.get('is_final', False)
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear():
    """Clear transcription history."""
    return jsonify({'success': True})


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


# ============================================================
# Training Interface Endpoints
# ============================================================

@app.route('/training')
def training():
    """Serve the training page."""
    return render_template('training.html')


@app.route('/recordings', methods=['GET'])
def list_recordings():
    """List all recordings."""
    recordings = []

    if os.path.exists(RECORDINGS_DIR):
        for filename in sorted(os.listdir(RECORDINGS_DIR)):
            if not filename.endswith('.wav'):
                continue

            wav_path = os.path.join(RECORDINGS_DIR, filename)
            json_path = wav_path.replace('.wav', '.json')

            # Get label from JSON or filename
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                label = metadata.get('label', '')
                timestamp = metadata.get('timestamp', '')
            else:
                label = os.path.splitext(filename)[0]
                timestamp = ''

            # Get file size
            size = os.path.getsize(wav_path)

            recordings.append({
                'id': filename.replace('.wav', ''),
                'filename': filename,
                'label': label,
                'timestamp': timestamp,
                'size': size
            })

    return jsonify({'recordings': recordings})


@app.route('/recordings', methods=['POST'])
def save_recording():
    """Save a new recording with label."""
    try:
        data = request.json
        audio_base64 = data.get('audio')
        label = data.get('label', '').strip()
        sample_rate = data.get('sampleRate', 44100)

        if not audio_base64:
            return jsonify({'error': 'No audio data'}), 400

        if not label:
            return jsonify({'error': 'No label provided'}), 400

        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

        # Resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.lower().replace(' ', '_')
        safe_label = ''.join(c for c in safe_label if c.isalnum() or c == '_')
        filename = f"{safe_label}_{timestamp}"

        # Save WAV file
        wav_path = os.path.join(RECORDINGS_DIR, f"{filename}.wav")
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(wav_path, SAMPLE_RATE, audio_int16)

        # Save metadata JSON
        json_path = os.path.join(RECORDINGS_DIR, f"{filename}.json")
        metadata = {
            'label': label.lower(),
            'timestamp': timestamp,
            'sample_rate': SAMPLE_RATE
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return jsonify({
            'success': True,
            'id': filename,
            'message': f"Saved: '{label}'"
        })

    except Exception as e:
        print(f"Error saving recording: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/recordings/<recording_id>', methods=['DELETE'])
def delete_recording(recording_id):
    """Delete a recording."""
    try:
        wav_path = os.path.join(RECORDINGS_DIR, f"{recording_id}.wav")
        json_path = os.path.join(RECORDINGS_DIR, f"{recording_id}.json")

        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(json_path):
            os.remove(json_path)

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def start_training():
    """Start model fine-tuning."""
    global recognizer

    try:
        # Check if there are recordings
        recordings = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.wav')]
        if not recordings:
            return jsonify({
                'success': False,
                'message': 'No recordings found. Add some training samples first.'
            }), 400

        print(f"\nStarting fine-tuning with {len(recordings)} recordings...")

        # Run fine-tuning script
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore'

        result = subprocess.run(
            [sys.executable, 'finetune_whisper.py'],
            capture_output=True,
            text=True,
            timeout=3600,
            env=env
        )

        if 'Fine-tuning complete' in result.stdout or result.returncode == 0:
            # Reload the model
            print("Reloading model...")
            recognizer = None
            init_recognizer()

            return jsonify({
                'success': True,
                'message': f'Model trained successfully with {len(recordings)} recordings!'
            })
        else:
            stderr_lines = [l for l in result.stderr.split('\n') if l.strip() and 'Warning' not in l]
            error_msg = stderr_lines[-1] if stderr_lines else result.stderr[:200]
            return jsonify({
                'success': False,
                'message': f'Training failed: {error_msg}'
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'message': 'Training timed out (exceeded 1 hour)'
        }), 500
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/train-status')
def train_status():
    """Get training data status."""
    recordings = []
    if os.path.exists(RECORDINGS_DIR):
        recordings = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.wav')]

    model_exists = os.path.exists('./whisper-finetuned')

    return jsonify({
        'recording_count': len(recordings),
        'model_exists': model_exists
    })


def main():
    """Run the Flask server."""
    port = 5001
    local_ip = get_local_ip()

    # Check for SSL certificates
    cert_file = 'cert.pem'
    key_file = 'key.pem'
    ssl_available = os.path.exists(cert_file) and os.path.exists(key_file)

    print("\n" + "="*60)
    print("  Parrot - Live Transcription")
    print("="*60)

    if ssl_available:
        print(f"\n  MacBook:  https://localhost:{port}")
        print(f"  iPhone:   https://{local_ip}:{port}")
        print("\n  (Accept the security warning on iPhone)")
    else:
        print(f"\n  MacBook:  http://localhost:{port}")
        print(f"\n  For iPhone access, generate SSL certificates:")
        print(f"  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj \"/CN=localhost\"")

    print("\n  Press the button (or Spacebar) to start.")
    print("  Speak naturally - text appears as you talk.")
    print("="*60 + "\n")

    if ssl_available:
        app.run(host='0.0.0.0', port=port, debug=False,
                ssl_context=(cert_file, key_file))
    else:
        app.run(host='127.0.0.1', port=port, debug=False)


if __name__ == '__main__':
    main()
