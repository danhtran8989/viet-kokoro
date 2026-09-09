# Project Structure

```
kokoro/
├── src/
│   └── viet_tts/              # Core Python package
│       ├── __init__.py
│       ├── config.py          # Environment config loader
│       ├── core.py            # TTS utilities
│       └── _kokoro/           # Neural network model
│           ├── __init__.py
│           ├── model.py       # KModel
│           ├── modules.py     # Transformer modules
│           ├── istftnet.py    # Decoder / iSTFT network
│           └── custom_stft.py # Custom STFT implementation
│
├── apis/
│   └── fast_api/              # FastAPI REST server
│       └── app.py
│
├── apps/
│   └── gradio_app/            # Gradio web interface
│       └── app.py
│
├── scripts/
│   └── download_ckpts.py      # Model downloader
│
├── kokoro_tts/                # Original reference code
│   ├── app.py
│   ├── kokoro_vietnamese/     # Original package
│   └── requirements.txt
│
├── secrets/
│   ├── .env                   # Runtime config (git-ignored)
│   └── .env.example           # Template
│
├── ckpts/                     # Model checkpoints (git-ignored)
│
├── requirements/
│   └── requirements.txt       # All dependencies
│
├── docs/
│   ├── getting-started.md
│   ├── api.md
│   ├── download.md
│   └── structure.md
│
├── README.md
└── .gitignore
```

## Module Overview

### `src/viet_tts/config.py`

Loads configuration from `secrets/.env` using `python-dotenv`. Provides constants:
- `KOKORO_HF_REPO`, `KOKORO_BASE_REPO` - HuggingFace repos
- `KOKORO_MODEL_FILE`, `KOKORO_VOICEPACK_FILE`, `KOKORO_CONFIG_FILE` - Filenames
- `CKPT_DIR` - Path to checkpoint directory

### `src/viet_tts/core.py`

TTS utilities:
- `VOICES` - Available voice definitions
- `SAMPLE_RATE` - Audio sample rate (24000)
- `split_text()` - Split text into chunks
- `phonemize()` - Convert text to phonemes via vig2p
- `merge_audio_chunks()` - Crossfade and merge audio

### `src/viet_tts/_kokoro/`

Neural network model (`KModel`):
- `model.py` - KModel class, forward pass
- `modules.py` - Text encoder, prosody predictor
- `istftnet.py` - Decoder, generator
- `custom_stft.py` - STFT without complex ops
