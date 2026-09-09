# Getting Started

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- [HuggingFace account](https://huggingface.co) (for model download)

## Installation

```bash
pip install -r requirements/requirements.txt
```

## Configuration

1. Copy the example env file:

```bash
cp secrets/.env.example secrets/.env
```

2. Edit `secrets/.env` as needed. See [Configuration](../README.md#configuration) for available options.

## Downloading Models

```bash
python scripts/download_ckpts.py
```

This downloads the following files to `ckpts/`:

- `config.json` - Model configuration
- `kokoro_vi.pth` - Vietnamese model weights
- `kokoro_vi_voicepack.pt` - Voice pack
- `kokoro-v1_0.pth` - Base model weights

## Running the API Server

```bash
python apis/fast_api/app.py
```

The API starts on `http://localhost:8000`. See [API Endpoints](../README.md#api-endpoints) for details.

## Running the Gradio UI

```bash
python apps/gradio_app/app.py
```

The UI starts on `http://localhost:7860`.
