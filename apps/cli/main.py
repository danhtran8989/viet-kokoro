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

REPO_ID = "contextboxai/Kokoro-Vietnamese"
CKPTS_DIR = Path(__file__).resolve().parent.parent.parent / "ckpts" / "Kokoro-Vietnamese"


def _resolve_file(filename: str) -> str:
    local = CKPTS_DIR / filename
    if local.exists():
        print(f"[INFO] Using local checkpoint: {local}")
        return str(local)
    print(f"[INFO] Downloading from HF: {REPO_ID}/{filename}")
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


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
        _vp_path = CKPTS_DIR / _vinfo["filename"]
        if _vp_path.exists():
            voicepacks[_vname] = torch.load(_vp_path, map_location="cpu", weights_only=True)
            print(f"[INFO] Loaded voice: {_vname}")
        else:
            print(f"[WARN] Voice file not found: {_vp_path}")

    # Automatically print available voices
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
    for index, chunk_text in enumerate(split_text(text), start=1):
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


def generate_batch(input_file: str, voices: list, speed: float, output_dir: str, model, voicepacks, device):
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
    
    total_jobs = len(lines) * len(valid_voices)
    job_idx = 1
    success_count = 0

    print(f"\n[INFO] Starting batch generation: {len(lines)} lines × {len(valid_voices)} voices = {total_jobs} files")
    print(f"[INFO] Base output directory: {out_dir.resolve()}")
    print(f"[INFO] Files will be organized into subfolders for each voice.\n")

    # Loop through voices first to create subfolders and keep console output clean
    for voice in valid_voices:
        # Create a subfolder for each voice automatically
        voice_dir = out_dir / voice
        voice_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing voice: {voice} -> {voice_dir.resolve()}")
        
        for line_idx, text in enumerate(lines, start=1):
            print(f"[{job_idx}/{total_jobs}] Line {line_idx} | Text: {text[:50]}...")
            
            audio_chunks = []
            skip_line = False
            for chunk_text in split_text(text):
                ps = phonemize(chunk_text)
                if not ps:
                    continue
                if len(ps) > 510:
                    print(f"  [WARN] Phoneme chunk too long ({len(ps)} > 510). Skipping this line.")
                    skip_line = True
                    break
                
                with torch.no_grad():
                    ref_s = voicepacks[voice][len(ps) - 1]
                    audio = model(ps, ref_s, float(speed))
                audio_chunks.append(audio.detach().cpu().numpy())
            
            if skip_line or not audio_chunks:
                print(f"  [WARN] No audio generated for this line.")
                job_idx += 1
                continue

            crossfade_samples = round(SAMPLE_RATE * 50 / 1000)
            audio = merge_audio_chunks(audio_chunks, crossfade_samples)
            
            # Create safe filename: line_001_slug.wav (voice folder handles the voice name)
            slug = sanitize_filename(text, max_len=25)
            filename = f"line_{line_idx:03d}_{slug}.wav"
            out_path = voice_dir / filename
            
            if sf is not None:
                sf.write(out_path, audio, SAMPLE_RATE)
            else:
                from scipy.io import wavfile
                wavfile.write(out_path, SAMPLE_RATE, audio)
                
            print(f"  [SUCCESS] Saved: {filename}")
            success_count += 1
            job_idx += 1

    print("-" * 60)
    print(f"[INFO] Batch complete! Successfully generated {success_count}/{total_jobs} files in {out_dir.resolve()}")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro Vietnamese TTS CLI (Single & Batch Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  Single:  python apps/cli/main.py -t 'Xin chào' -o out.wav -v diem_trinh\n"
               "  Batch:   python apps/cli/main.py -i sentences.txt -o ./output_folder -v diem_trinh mai_linh ngoc_huyen"
    )
    
    # Mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", "-t", type=str, help="Vietnamese text to synthesize (Single mode)")
    input_group.add_argument("--input-file", "-i", type=str, help="Path to a text file with one sentence per line (Batch mode)")
    
    parser.add_argument("--output", "-o", type=str, required=True, 
                        help="Output file path (Single mode) or base output directory (Batch mode)")
    parser.add_argument("--voice", "-v", type=str, nargs='+', default=["diem_trinh"], 
                        help="Voice name(s). Provide multiple for batch mode (e.g., -v voice1 voice2). Default: diem_trinh")
    parser.add_argument("--speed", "-s", type=float, default=1.0, 
                        help="Speech speed multiplier (default: 1.0, recommended: 0.75 - 1.25)")
    
    args = parser.parse_args()
    
    print("[INFO] Loading model and voices...")
    model, voicepacks, device = load_model_and_voices()
    
    if not voicepacks:
        print("[ERROR] No voices were loaded. Please check your ckpts directory.")
        sys.exit(1)

    if args.input_file:
        # Batch Mode
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
            device=device
        )
        sys.exit(0 if success else 1)
        
    else:
        # Single Mode
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