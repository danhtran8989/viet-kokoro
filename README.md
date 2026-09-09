# Kokoro Vietnamese TTS

Vietnamese text-to-speech using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) fine-tuned on Vietnamese data with G2P via [vig2p](https://pypi.org/project/vig2p/).

## Project Structure

```
kokoro/
├── src/viet_tts/           # Core library
│   ├── __init__.py
│   ├── config.py           # Loads config from secrets/.env
│   ├── core.py             # VOICES, split_text, phonemize, merge_audio_chunks
│   └── _kokoro/            # KModel, neural network modules
├── apis/fast_api/          # FastAPI REST server
│   └── app.py
├── apps/gradio_app/        # Gradio web UI
│   └── app.py
├── scripts/
│   └── download_ckpts.py   # Download model checkpoints
├── kokoro_tts/             # Original reference code
├── secrets/
│   ├── .env                # Your config (git-ignored)
│   └── .env.example        # Template
├── ckpts/                  # Downloaded model files (git-ignored)
└── requirements/
    └── requirements.txt    # All dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements/requirements.txt
```

### 2. Configure

```bash
cp secrets/.env.example secrets/.env
# Edit secrets/.env with your settings
```

### 3. Download model checkpoints

```bash
python scripts/download_ckpts.py
```

### 4. Run

**FastAPI server** (port 8000):

```bash
python apis/fast_api/app.py
```

**Gradio web UI** (port 7860):

```bash
python apps/gradio_app/app.py
```

## Configuration

Environment variables in `secrets/.env`:

| Variable | Default | Description |
|---|---|---|
| `KOKORO_HF_REPO` | `contextboxai/Kokoro-Vietnamese` | HuggingFace repo for Vietnamese model |
| `KOKORO_BASE_REPO` | `hexgrad/Kokoro-82M` | HuggingFace repo for base model |
| `KOKORO_CONFIG_FILE` | `config.json` | Config filename |
| `KOKORO_MODEL_FILE` | `kokoro_vi.pth` | Model filename |
| `KOKORO_VOICEPACK_FILE` | `kokoro_vi_voicepack.pt` | Voicepack filename |

## Download Script

```bash
# Use defaults from secrets/.env
python scripts/download_ckpts.py

# Override via CLI
python scripts/download_ckpts.py --repo other/repo --model other.pth --outdir ./custom_dir

# Force re-download
python scripts/download_ckpts.py --force
```

## Available Voices

| ID | Name |
|---|---|
| `diem_trinh` | Diem Trinh |
| `hung_thinh` | Hung Thinh |
| `mai_linh` | Mai Linh |
| `mai_loan` | Mai Loan |
| `manh_dung` | Manh Dung |
| `my_yen` | My Yen |
| `ngoc_huyen` | Ngoc Huyen |
| `phat_tai` | Phat Tai |
| `thanh_dat` | Thanh Dat |
| `thuc_trinh` | Thuc Trinh |
| `tuan_ngoc` | Tuan Ngoc |
| `storyvert` | storyvert |
| `duc_an` | Duc An |
| `duc_duy` | Duc Duy |

## API Endpoints

### `GET /voices`

Returns available voices.

### `POST /tts`

Request body:

```json
{
  "text": "Xin chao ban",
  "voice": "diem_trinh",
  "speed": 1.0
}
```

Returns WAV audio file.

### `POST /tts/json`

Same request, returns JSON with metadata.
