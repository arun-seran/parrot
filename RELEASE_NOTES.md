# Release Notes - Parrot v2.0.0

## Major Changes

### Live Transcription Mode

The biggest change in v2.0.0 is the shift from "tap-to-talk" to **live transcription**.

**Why?** The previous version required pressing a button before each phrase. Children with autism often don't express themselves when they see adults performing the ritual of bringing up a phone and pressing buttons. This friction prevented natural communication.

**How it works now:** Press the button once to start listening. The app transcribes continuously as the child speaks. You can easily distinguish what was said by looking at the transcription in real-time.

### Built-in Training Interface

Instead of manually adding recordings and running scripts, you can now train the model directly from the web UI:

1. Navigate to the Training page (`/training`)
2. Enter a word/phrase label
3. Record the word being spoken
4. Repeat for multiple samples
5. Click "Start Training"

This makes it much easier to improve recognition for specific words without touching the command line.

### Simplified Architecture

- **Removed speaker filtering** - No longer needed with live transcription. You can see the full conversation and easily identify who said what.
- **Removed voice enrollment** - No more recording voice profiles
- **Removed corrections workflow** - Replaced with the simpler training interface

## New Features

- **Live transcription** - Continuous speech-to-text as you speak
- **Training web UI** - Record, label, and train from the browser
- **Keyboard shortcuts** - Press Space to start/stop transcription
- **Adjustable chunk duration** - Control how often transcription updates

## Breaking Changes

- The `/translate` endpoint is now `/transcribe`
- Voice enrollment endpoints have been removed
- Corrections folder is no longer used (use Recordings folder via training UI)
- The `SpeechRecognizer` class is now `LiveSpeechRecognizer`

## Migration from v1.0

If you have existing recordings in `Corrections/`, you can move them to `Recordings/` to use them for training. Make sure each recording has a corresponding `.json` file with a `label` field:

```json
{
  "label": "water",
  "timestamp": "20240115_143022",
  "sample_rate": 16000
}
```

Or simply name the files with the label (e.g., `water1.wav`, `water2.wav`).

## Requirements

No changes to hardware or software requirements:
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- macOS, Linux, or Windows
- ffmpeg (Linux/Windows only, for .m4a conversion)
