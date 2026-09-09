# apps/cli/main.py
import sys
import os
import re
from pathlib import Path

# Add src to path so we can import kokoro_vietnamese
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
import argparse
import numpy as np
import torch

# ==============================================================================
# PATCH: Force weights_only=False for torch.load
# Required for PyTorch 2.6+ compatibility with this specific model format.
# ==============================================================================
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# ==============================================================================

from huggingface_hub import hf_hub_download

from kokoro_vietnamese._kokoro import KModel
from kokoro_vietnamese.core import (
    SAMPLE_RATE,
    VOICES,
    split_text,
    merge_audio_chunks,
    phonemize,
    get_device,
)

# Prevent optional dependency warnings in transformers
import transformers.utils.import_utils as _import_utils
for _flag in ("_torchvision_available", "_librosa_available", "_cv2_available"):
    if hasattr(_import_utils, _flag):
        setattr(_import_utils, _flag, False)
if hasattr(_import_utils, "_torchvision_version"):
    _import_utils._torchvision_version = "N/A"

# Try to import soundfile for audio saving, fallback to scipy
try:
    import soundfile as sf
except ImportError:
    sf = None

# Try to import tqdm for progress bars, fallback to a no-op identity function
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

REPO_ID = "contextboxai/Kokoro-Vietnamese"
CKPTS_DIR = Path(__file__).resolve().parent.parent.parent / "ckpts" / "Kokoro-Vietnamese"


def _resolve_file(filename: str) -> str:
    """Resolve a checkpoint file, auto-detecting and replacing Git LFS pointers."""
    local = CKPTS_DIR / filename

    if local.exists():
        try:
            with open(local, "rb") as f:
                header = f.read(10)
            if header.startswith(b"version "):
                print(f"[WARN] {local.name} is a Git LFS pointer. Deleting and redownloading...")
                local.unlink()
            else:
                print(f"[INFO] Using local checkpoint: {local}")
                return str(local)
        except Exception as e:
            print(f"[WARN] Could not read {local}: {e}. Redownloading...")
            local.unlink()

    print(f"[INFO] Downloading from HF: {REPO_ID}/{filename}")
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(CKPTS_DIR.parent),
        local_dir_use_symlinks=False,
    )
    final_path = CKPTS_DIR / filename
    if not final_path.exists() and Path(downloaded).exists():
        import shutil
        shutil.copy(downloaded, final_path)
    return str(final_path)


def _is_lfs_pointer(filepath: Path) -> bool:
    """Check if a file is a Git LFS pointer (fake text file) instead of real data."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(10)
        return header.startswith(b"version ")
    except Exception:
        return False


def load_model_and_voices():
    """Initialize the model and preload voicepacks."""
    _config_path = _resolve_file("config.json")
    _model_path = _resolve_file("kokoro_vi.pth")

    with open(_config_path, "r", encoding="utf-8") as _f:
        _config = json.load(_f)

    _device = get_device()
    print(f"[INFO] Using device: {_device}")

    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=_config,
        model=_model_path,
    ).to(_device).eval()

    # Allow local voices.json to override default VOICES
    active_voices = VOICES
    _voices_json = CKPTS_DIR / "voices.json"
    if _voices_json.exists():
        with open(_voices_json, "r", encoding="utf-8") as _f:
            active_voices = json.load(_f)

    voicepacks = {}
    for _vname, _vinfo in active_voices.items():
        _vp_filename = _vinfo["filename"]
        _vp_path = CKPTS_DIR / _vp_filename

        # --- KEY FIX: Check voice files for LFS pointers too ---
        if _vp_path.exists() and _is_lfs_pointer(_vp_path):
            print(f"[WARN] Voice file {_vp_filename} is a Git LFS pointer. Redownloading...")
            _vp_path.unlink()

        if not _vp_path.exists():
            print(f"[INFO] Downloading voice file: {_vp_filename}")
            try:
                downloaded = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=_vp_filename,
                    local_dir=str(CKPTS_DIR.parent),
                    local_dir_use_symlinks=False,
                )
                if not _vp_path.exists() and Path(downloaded).exists():
                    import shutil
                    shutil.copy(downloaded, _vp_path)
            except Exception as e:
                print(f"[ERROR] Failed to download {_vp_filename}: {e}")
                continue
        # -------------------------------------------------------

        if _vp_path.exists():
            voicepacks[_vname] = torch.load(_vp_path, map_location="cpu", weights_only=False)
            print(f"[INFO] Loaded voice: {_vname}")
        else:
            print(f"[WARN] Voice file not found: {_vp_path}")

    print("-" * 60)
    print(f"[INFO] Available voices ({len(voicepacks)}): {', '.join(voicepacks.keys())}")
    print("-" * 60)

    return model, voicepacks, _device


def sanitize_filename(text: str, max_len: int = 20) -> str:
    """Create a safe, short filename slug from text."""
    slug = re.sub(r'[^\w\s-]', '', text.strip())
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug[:max_len]


def generate_single(text: str, voice: str, speed: float, output_path: str, model, voicepacks, device):
    """Generate audio from text and save to a single file."""
    vp = voicepacks.get(voice)
    if vp is None:
        print(f"[ERROR] Voice '{voice}' not found.")
        return False

    audio_chunks = []
    chunks = list(split_text(text))
    
    # Use tqdm for chunk processing, disable if 1 or fewer chunks
    progress_iter = tqdm(chunks, desc="Generating chunks", disable=len(chunks) <= 1)
    for index, chunk_text in enumerate(progress_iter, start=1):
        ps = phonemize(chunk_text)
        if not ps:
            continue
        if len(ps) > 510:
            print(f"[ERROR] Phoneme chunk too long ({len(ps)} > 510): {chunk_text[:80]}")
            return False
        
        with torch.no_grad():
            ref_s = vp[len(ps) - 1]
            audio = model(ps, ref_s, float(speed))
        audio_chunks.append(audio.detach().cpu().numpy())

    if not audio_chunks:
        print("[ERROR] No audio generated.")
        return False

    crossfade_samples = round(SAMPLE_RATE * 50 / 1000)
    audio = merge_audio_chunks(audio_chunks, crossfade_samples)
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if sf is not None:
        sf.write(out_path, audio, SAMPLE_RATE)
    else:
        try:
            from scipy.io import wavfile
            wavfile.write(out_path, SAMPLE_RATE, audio)
        except ImportError:
            print("[ERROR] Neither 'soundfile' nor 'scipy' is installed.")
            return False

    print(f"[SUCCESS] Saved: {out_path.resolve()}")
    return True


def generate_batch(input_file: str, voices: list, speed: float, output_dir: str, model, voicepacks, device, batch_size: int = 1):
    """Generate audio for multiple lines and multiple voices, organized by voice subfolders."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("[ERROR] Input file is empty or contains no valid text.")
        return False

    valid_voices = [v for v in voices if v in voicepacks]
    invalid_voices = [v for v in voices if v not in voicepacks]
    
    for v in invalid_voices:
        print(f"[WARN] Voice '{v}' not found. Skipping.")
    
    if not valid_voices:
        print("[ERROR] No valid voices provided.")
        return False

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-create voice directories
    voice_dirs = {}
    for voice in valid_voices:
        v_dir = out_dir / voice
        v_dir.mkdir(parents=True, exist_ok=True)
        voice_dirs[voice] = v_dir

    # Create a flat list of all generation tasks
    tasks = []
    for voice in valid_voices:
        for line_idx, text in enumerate(lines, start=1):
            tasks.append((voice, voice_dirs[voice], line_idx, text))
            
    total_jobs = len(tasks)
    success_count = 0

    print(f"\n[INFO] Starting batch generation: {len(lines)} lines × {len(valid_voices)} voices = {total_jobs} files")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Base output directory: {out_dir.resolve()}\n")

    # Process tasks in batches with tqdm progress bar
    for i in tqdm(range(0, total_jobs, batch_size), desc="Generating batches"):
        batch_tasks = tasks[i:i+batch_size]
        
        # NOTE: True model-level batching would require padding phonemes and stacking ref_s.
        # For maximum compatibility with the current KModel API, we process sequentially within the batch chunk.
        for voice, voice_dir, line_idx, text in batch_tasks:
            audio_chunks = []
            skip_line = False
            
            chunk_texts = list(split_text(text))
            for chunk_text in chunk_texts:
                ps = phonemize(chunk_text)
                if not ps:
                    continue
                if len(ps) > 510:
                    print(f"\n  [WARN] Phoneme chunk too long ({len(ps)} > 510). Skipping line {line_idx}.")
                    skip_line = True
                    break
                
                with torch.no_grad():
                    ref_s = voicepacks[voice][len(ps) - 1]
                    audio = model(ps, ref_s, float(speed))
                audio_chunks.append(audio.detach().cpu().numpy())
            
            if skip_line or not audio_chunks:
                continue

            crossfade_samples = round(SAMPLE_RATE * 50 / 1000)
            audio = merge_audio_chunks(audio_chunks, crossfade_samples)
            
            slug = sanitize_filename(text, max_len=25)
            filename = f"line_{line_idx:03d}_{slug}.wav"
            out_path = voice_dir / filename
            
            if sf is not None:
                sf.write(out_path, audio, SAMPLE_RATE)
            else:
                from scipy.io import wavfile
                wavfile.write(out_path, SAMPLE_RATE, audio)
                
            success_count += 1

    print("-" * 60)
    print(f"[INFO] Batch complete! Successfully generated {success_count}/{total_jobs} files in {out_dir.resolve()}")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro Vietnamese TTS CLI (Single & Batch Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  Single:  python apps/cli/main.py -t 'Xin chào' -o out.wav -v diem_trinh\n"
               "  Batch:   python apps/cli/main.py -i sentences.txt -o ./output_folder -v diem_trinh mai_linh -b 4"
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", "-t", type=str, help="Vietnamese text to synthesize (Single mode)")
    input_group.add_argument("--input-file", "-i", type=str, help="Path to a text file with one sentence per line (Batch mode)")
    
    parser.add_argument("--output", "-o", type=str, required=True, 
                        help="Output file path (Single mode) or base output directory (Batch mode)")
    parser.add_argument("--voice", "-v", type=str, nargs='+', default=["diem_trinh"], 
                        help="Voice name(s). Provide multiple for batch mode. Default: diem_trinh")
    parser.add_argument("--speed", "-s", type=float, default=1.0, 
                        help="Speech speed multiplier (default: 1.0, recommended: 0.75 - 1.25)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, 
                        help="Batch size for processing (default: 1). Higher values group tqdm updates and prepare for future model-level batching.")
    
    args = parser.parse_args()
    
    print("[INFO] Loading model and voices...")
    model, voicepacks, device = load_model_and_voices()
    
    if not voicepacks:
        print("[ERROR] No voices were loaded. Please check your ckpts directory.")
        sys.exit(1)

    if args.input_file:
        if not Path(args.input_file).exists():
            print(f"[ERROR] Input file not found: {args.input_file}")
            sys.exit(1)
        
        success = generate_batch(
            input_file=args.input_file,
            voices=args.voice,
            speed=args.speed,
            output_dir=args.output,
            model=model,
            voicepacks=voicepacks,
            device=device,
            batch_size=args.batch_size
        )
        sys.exit(0 if success else 1)
        
    else:
        if not args.text or not args.text.strip():
            print("[ERROR] Please provide valid text with --text.")
            sys.exit(1)
            
        chosen_voice = args.voice[0]
        if chosen_voice not in voicepacks:
            print(f"[ERROR] Voice '{chosen_voice}' not found.")
            print(f"[INFO] Available voices: {', '.join(voicepacks.keys())}")
            sys.exit(1)
            
        print(f"\n[INFO] Generating audio with voice='{chosen_voice}' at speed {args.speed}x...")
        success = generate_single(
            text=args.text,
            voice=chosen_voice,
            speed=args.speed,
            output_path=args.output,
            model=model,
            voicepacks=voicepacks,
            device=device
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()