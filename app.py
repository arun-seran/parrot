#!/usr/bin/env python3
"""
Web-based Speech Recognizer

A Flask web application for speech recognition using fine-tuned Whisper.
Supports both local HTTPS (for mobile browser microphone access) and
production deployment (where the hosting platform handles SSL).
"""

import os
import socket
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from speech_recognizer import SpeechRecognizer, SAMPLE_RATE

app = Flask(__name__)
recognizer = None

# Store last audio for corrections
last_audio_data = None
last_sample_rate = None


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
    """Initialize the recognizer (called on module import for gunicorn)."""
    global recognizer

    if recognizer is not None:
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("Initializing Speech Recognizer...")
    recognizer = SpeechRecognizer()

    # Load enrolled voice or auto-enroll from recordings
    if recognizer.load_enrolled_voice():
        print("Speaker filtering enabled (loaded existing profile)")
    else:
        print("No voice profile found - auto-enrolling from recordings...")
        if recognizer.enroll_from_recordings():
            print("Speaker filtering enabled (enrolled from recordings)")
        else:
            print("Could not enroll voice - speaker filtering disabled")

    # Pre-load model
    print("\nPre-loading Whisper model (this may take a moment)...")
    recognizer.load_model()
    print("Recognizer initialized!")


# Initialize on module import (for gunicorn)
init_recognizer()


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():
    """Receive audio and return transcription."""
    global last_audio_data, last_sample_rate

    try:
        data = request.json
        audio_base64 = data.get('audio')

        if not audio_base64:
            return jsonify({'error': 'No audio data received'}), 400

        # Decode base64 audio (PCM float32 from client)
        audio_bytes = base64.b64decode(audio_base64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        sample_rate = data.get('sampleRate', 44100)

        # Check audio energy
        energy = np.sqrt(np.mean(audio_data**2))
        if energy < 0.005:
            return jsonify({'error': 'Audio too quiet. Please speak louder.'}), 400

        # Store for potential correction
        last_audio_data = audio_data.copy()
        last_sample_rate = sample_rate

        # Recognize speech
        result = recognizer.recognize(audio_data, sample_rate)

        if 'error' in result:
            return jsonify({
                'error': result['error'],
                'speaker_segments': result.get('speaker_segments', [])
            }), 400

        return jsonify({
            'transcription': result['transcription'],
            'raw_transcription': result.get('raw_transcription', result['transcription']),
            'speaker_segments': result.get('speaker_segments', [])
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/correct', methods=['POST'])
def correct():
    """Save a correction for future retraining."""
    global last_audio_data, last_sample_rate

    try:
        data = request.json
        correct_text = data.get('correct', '')

        if not correct_text:
            return jsonify({'error': 'Missing correct text'}), 400

        if last_audio_data is None:
            return jsonify({'error': 'No recent audio to correct'}), 400

        # Save the correction (audio + correct label)
        recognizer.save_correction(last_audio_data, last_sample_rate, correct_text)

        correction_count = recognizer.get_correction_count()

        return jsonify({
            'success': True,
            'message': f"Saved correction: '{correct_text}'",
            'correction_count': correction_count
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model with corrections."""
    try:
        result = recognizer.retrain_model()

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/retrain-status')
def retrain_status():
    """Get the number of pending corrections."""
    correction_count = recognizer.get_correction_count()
    return jsonify({
        'correction_count': correction_count
    })


@app.route('/enroll', methods=['POST'])
def enroll_voice():
    """Enroll speaker's voice from audio samples."""
    try:
        data = request.json
        audio_samples = data.get('samples', [])

        if not audio_samples:
            return jsonify({'error': 'No audio samples provided'}), 400

        # Convert samples from base64
        processed_samples = []
        for sample in audio_samples:
            audio_bytes = base64.b64decode(sample['audio'])
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            sample_rate = sample.get('sampleRate', 44100)
            processed_samples.append((audio_data, sample_rate))

        # Enroll voice
        success = recognizer.enroll_voice(processed_samples)

        if success:
            return jsonify({
                'success': True,
                'message': f'Voice enrolled with {len(processed_samples)} samples'
            })
        else:
            return jsonify({'error': 'Failed to enroll voice'}), 400

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/voice-status')
def voice_status():
    """Check if voice is enrolled."""
    enrolled = recognizer.enrolled_embedding is not None
    return jsonify({
        'enrolled': enrolled
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


def main():
    """Run the Flask server."""
    # Check if running in production (via PORT env var)
    port = int(os.environ.get('PORT', 5001))

    # Check if SSL certs exist for local HTTPS
    ssl_available = os.path.exists('cert.pem') and os.path.exists('key.pem')

    if ssl_available:
        # Local development with HTTPS (required for mobile microphone access)
        local_ip = get_local_ip()

        print("\n" + "="*60)
        print("  Speech Recognizer Web App (HTTPS)")
        print("="*60)
        print(f"\n  Local access: https://localhost:{port}")
        print(f"  Network access: https://{local_ip}:{port}")
        print(f"\n  (Make sure mobile device is on the same WiFi network)")
        print("\n  NOTE: Browser will show a security warning for self-signed cert.")
        print("  Click 'Advanced' -> 'Proceed' (or equivalent for your browser)")
        print("="*60 + "\n")

        app.run(host='0.0.0.0', port=port, debug=False,
                ssl_context=('cert.pem', 'key.pem'))
    else:
        # Production mode (SSL handled by hosting platform)
        print(f"\nStarting server on port {port}...")
        print("(No SSL certs found - running in HTTP mode)")
        print("For mobile access, deploy to a platform that provides SSL.\n")

        app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()
