# Download Script

## Usage

```bash
python scripts/download_ckpts.py [OPTIONS]
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--repo` | `contextboxai/Kokoro-Vietnamese` | HuggingFace repo for Vietnamese model |
| `--base-repo` | `hexgrad/Kokoro-82M` | HuggingFace repo for base model |
| `--config` | `config.json` | Config filename |
| `--model` | `kokoro_vi.pth` | Model weights filename |
| `--voicepack` | `kokoro_vi_voicepack.pt` | Voice pack filename |
| `--outdir` | `ckpts/` | Output directory |
| `--force` | off | Re-download existing files |

## Examples

```bash
# Download with defaults
python scripts/download_ckpts.py

# Download to custom directory
python scripts/download_ckpts.py --outdir /data/models

# Override repo
python scripts/download_ckpts.py --repo my-org/my-model --model my_model.pth

# Force re-download
python scripts/download_ckpts.py --force
```

## Downloaded Files

The script downloads 5 files to the output directory:

1. `config.json` - From `KOKORO_HF_REPO`
2. `kokoro_vi.pth` - From `KOKORO_HF_REPO`
3. `kokoro_vi_voicepack.pt` - From `KOKORO_HF_REPO`
4. `config.json` - From `KOKORO_BASE_REPO` (base model)
5. `kokoro-v1_0.pth` - From `KOKORO_BASE_REPO` (base model)

Files that already exist are skipped unless `--force` is used.
