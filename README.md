# Speech Helper

A speech recognition app designed for voices that standard speech-to-text models struggle with. Fine-tune OpenAI's Whisper model on your own recordings to create a personalized speech recognition system.

**Use cases:**
- Children with speech difficulties (apraxia, dysarthria, etc.)
- Accented speech
- Unique vocabularies or terminology
- Any speech that off-the-shelf models misrecognize

## How It Works

1. **Record training samples** - Collect recordings of the target speaker saying words/phrases you want to recognize
2. **Fine-tune Whisper** - The app fine-tunes OpenAI's Whisper model on your recordings
3. **Use the app** - Speak into the app and get accurate transcriptions
4. **Improve over time** - Correct mistakes through the UI, then retrain to improve accuracy

## Features

- **End-to-end learning** - No rule-based post-processing; the model learns directly from examples
- **Speaker filtering** - Optionally filter audio to only recognize a specific enrolled speaker
- **Correction workflow** - Easy UI to correct mistakes and save them for retraining
- **Mobile-friendly** - Works on mobile browsers with HTTPS
- **Self-improving** - Retrain the model from the UI as you collect more corrections

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/speech-to-text-helper.git
cd speech-to-text-helper
```

### 2. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install ffmpeg (only needed on Linux/Windows)

macOS users can skip this step - the app uses the built-in `afconvert` tool.

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

Note: ffmpeg is only used to convert `.m4a` files during training. Normal app operation doesn't require it.

### 4. Download the base Whisper model

The app uses Whisper-tiny as the base model. It will be downloaded automatically on first run, but you can pre-download it:

```bash
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; WhisperProcessor.from_pretrained('openai/whisper-tiny'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')"
```

### 5. Add your training recordings

Place your audio recordings in the `Recordings/` folder:

```
Recordings/
  hello.wav       # Will be labeled as "hello"
  goodbye.wav     # Will be labeled as "goodbye"
  water1.wav      # Will be labeled as "water"
  water2.wav      # Will be labeled as "water"
  mommy.m4a       # Will be labeled as "mommy"
```

**Recording format:**
- Supported formats: `.wav`, `.m4a`
- The filename (without extension and numbers) becomes the label
- Record multiple samples of each word for better results
- Keep recordings short (1-3 seconds per word/phrase)

### 6. Fine-tune the model

```bash
python finetune_whisper.py
```

This will:
- Load all recordings from `Recordings/`
- Apply data augmentation (pitch shift, time stretch, noise)
- Fine-tune Whisper on your data
- Save the model to `whisper-finetuned/`

Training typically takes 5-15 minutes depending on your hardware.

### 7. Run the app

```bash
python app.py
```

Open `http://localhost:5001` in your browser.

## Mobile Access (iPhone/Android)

Mobile browsers require HTTPS to access the microphone. Generate a self-signed certificate:

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

The app will automatically use HTTPS if it finds `cert.pem` and `key.pem`.

Access from your phone using your computer's IP address (e.g., `https://192.168.24.100:5001`). You'll need to accept the security warning.

## Project Structure

```
speech-to-text-helper/
├── app.py                  # Flask web application
├── speech_recognizer.py    # Core speech recognition logic
├── finetune_whisper.py     # Model fine-tuning script
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI
├── Recordings/             # Your training recordings (add your files here)
├── Corrections/            # Corrections saved from the UI
└── whisper-finetuned/      # Fine-tuned model (created after training)
```

## Configuration

Edit `speech_recognizer.py` to customize:

```python
# Phrases to filter out (e.g., prompts you say before the speaker talks)
IGNORE_PHRASES = [
    "what do you want",
    "say it again",
]
```

## Retraining

As you use the app:

1. When the model makes a mistake, tap "Wrong?" and enter the correct text
2. The correction is saved to `Corrections/`
3. Go to Settings > Retrain Model to retrain with your corrections

You can also retrain manually:

```bash
python finetune_whisper.py
```

## Voice Enrollment (Optional)

If multiple people will use the app, you can enroll a specific voice to filter out other speakers:

1. Go to Settings in the web UI
2. Record 3 voice samples from the target speaker
3. Save the voice profile

The app will then only transcribe audio that matches the enrolled voice.

## Tips for Better Results

1. **More samples = better accuracy** - Record at least 5-10 samples of each word
2. **Vary the recordings** - Different volumes, speeds, and contexts
3. **Keep it simple** - Start with single words before phrases
4. **Consistent labeling** - Use the same filename pattern for the same word
5. **Quality recordings** - Minimize background noise

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers
```

### "ffmpeg not found" (Linux/Windows only)
Install ffmpeg for your platform (see installation steps above). macOS users don't need ffmpeg.

### Model not loading
Make sure you've run `finetune_whisper.py` first to create the `whisper-finetuned/` directory.

### Mobile microphone not working
- Ensure you're using HTTPS (see Mobile Access section)
- Check that microphone permissions are granted
- Try a different browser

## Hardware Requirements

- **Minimum:** 8GB RAM, any modern CPU
- **Recommended:** 16GB RAM, Apple Silicon or NVIDIA GPU
- Training uses ~4GB VRAM (or system RAM on CPU/MPS)

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Base speech recognition model
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Model fine-tuning
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Speaker recognition
