# Parrot

Parrot is an open-source speech recognition app designed for voices that standard speech-to-text models struggle with.

It enables personalized speech recognition by fine-tuning OpenAI's Whisper model on your own recordings, allowing the system to learn directly from examples instead of relying on brittle rule-based hacks.

## Why Live Transcription?

This app was built primarily to help children with speech difficulties (due to autism) communicate. The previous version required pressing a button before each phrase, but children with autism often don't express themselves when they see adults going through the ritual of bringing up a phone and pressing buttons.

**Live transcription solves this** - you simply start listening and let the child speak naturally. The transcription happens continuously, and you can easily distinguish what the child said by looking at the final text.

## Use Cases

- Children with speech difficulties (autism, apraxia, etc.)
- Accented speech
- Unique vocabularies or terminology
- Any speech that off-the-shelf models misrecognize

## How It Works

1. **Start live transcription** - Press the button and speak naturally
2. **Train with examples** - Record words/phrases the model struggles with
3. **Fine-tune Whisper** - The app trains the model on your recordings
4. **See improvement** - The model learns to recognize those specific words

## Features

- **Live transcription** - Continuous speech-to-text as you speak
- **End-to-end learning** - No rule-based post-processing; the model learns directly from examples
- **Built-in training interface** - Record, label, and train from the web UI
- **Mobile-friendly** - Works on iPhone/Android with HTTPS
- **Self-improving** - Add more training samples to improve accuracy over time

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/arun-seran/parrot.git
cd parrot
```

### 2. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the base Whisper model

Before first use, download the base Whisper model (this only needs to be done once):

```bash
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; WhisperProcessor.from_pretrained('openai/whisper-tiny'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny'); print('Model downloaded!')"
```

This downloads ~150MB and caches it locally for future use.

### 4. Run the app

```bash
python app.py
```

Open `http://localhost:5001` in your browser.

### 5. Train the model (optional)

If the base Whisper model struggles with certain words:

1. Click "Train Model" in the app
2. Enter a word/phrase and record it being spoken
3. Add 5-10 samples per word for best results
4. Click "Start Training" (takes 5-15 minutes)

The model will now recognize those words better!

## Mobile Access (iPhone/Android)

Mobile browsers require HTTPS to access the microphone. Generate a self-signed certificate:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

The app will automatically use HTTPS if it finds `cert.pem` and `key.pem`.

Access from your phone using your computer's IP address (e.g., `https://192.168.23.34:5001`). You'll need to accept the security warning.

## Project Structure

```
parrot/
├── app.py                  # Flask web application (imports from speech_recognizer.py)
├── speech_recognizer.py    # Live speech recognition module
├── finetune_whisper.py     # Model fine-tuning script
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html          # Live transcription UI
│   └── training.html       # Training interface
├── Recordings/             # Training recordings (created when you add samples)
└── whisper-finetuned/      # Fine-tuned model (created after training)
```

All imports are local - `app.py` imports from `speech_recognizer.py` in the same directory. No external dependencies beyond the pip packages.

## Adding Training Recordings

You can add recordings via the web UI, or manually:

```
Recordings/
  hello.wav       # Will be labeled as "hello"
  goodbye.wav     # Will be labeled as "goodbye"
  water1.wav      # Will be labeled as "water"
  water2.wav      # Will be labeled as "water"
```

**Recording tips:**
- Supported formats: `.wav`, `.m4a`
- The filename (without extension and numbers) becomes the label
- Record multiple samples of each word for better results
- Keep recordings short (1-3 seconds per word/phrase)

## Manual Training

You can also train from the command line:

```bash
python finetune_whisper.py
```

This will:
- Load all recordings from `Recordings/`
- Apply data augmentation (pitch shift, time stretch, noise)
- Fine-tune Whisper on your data
- Save the model to `whisper-finetuned/`

## Installing ffmpeg (Linux/Windows only)

macOS users can skip this - the app uses the built-in `afconvert` tool.

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

Note: ffmpeg is only used to convert `.m4a` files during training.

## Tips for Better Results

1. **More samples = better accuracy** - Record at least 5-10 samples of each word
2. **Vary the recordings** - Different volumes, speeds, and contexts
3. **Keep it simple** - Start with single words before phrases
4. **Quality recordings** - Minimize background noise
5. **Be patient** - Training takes 5-15 minutes but the results are worth it

## Troubleshooting

### "No module named 'transformers'"
```bash
pip install transformers
```

### "ffmpeg not found" (Linux/Windows only)
Install ffmpeg for your platform (see installation steps above).

### Model not improving
- Add more training samples (5-10 per word minimum)
- Make sure recordings are clear with minimal background noise
- Try retraining from scratch by deleting `whisper-finetuned/`

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
