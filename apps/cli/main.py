# apps/cli/main.py
import sys
import os
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
    _voices_json = CKPTS_DIR / "voices.json"
    active_voices = VOICES
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


def generate_to_file(text: str, voice: str, speed: float, output_path: str, model, voicepacks, device):
    """Generate audio from text and save to a file."""
    if not text or not text.strip():
        print("[ERROR] Please enter valid Vietnamese text.")
        return False
    
    vp = voicepacks.get(voice)
    if vp is None:
        print(f"[ERROR] Voice '{voice}' not found or failed to load.")
        print(f"[INFO] Please choose from: {', '.join(voicepacks.keys())}")
        return False

    audio_chunks = []
    for index, chunk_text in enumerate(split_text(text), start=1):
        ps = phonemize(chunk_text)
        if not ps:
            continue
        if len(ps) > 510:
            print(f"[ERROR] Phoneme chunk too long ({len(ps)} > 510): {chunk_text[:80]}")
            return False
        
        print(f"[INFO] Processing chunk {index} ({len(ps)} phonemes)...")
        with torch.no_grad():
            ref_s = vp[len(ps) - 1]
            audio = model(ps, ref_s, float(speed))
        audio_chunks.append(audio.detach().cpu().numpy())

    if not audio_chunks:
        print("[ERROR] No audio generated.")
        return False

    print("[INFO] Merging audio chunks...")
    crossfade_samples = round(SAMPLE_RATE * 50 / 1000)
    audio = merge_audio_chunks(audio_chunks, crossfade_samples)
    
    # Ensure output directory exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save audio file
    if sf is not None:
        sf.write(out_path, audio, SAMPLE_RATE)
    else:
        try:
            from scipy.io import wavfile
            wavfile.write(out_path, SAMPLE_RATE, audio)
        except ImportError:
            print("[ERROR] Neither 'soundfile' nor 'scipy' is installed.")
            print("[INFO] Please install one of them to save audio: pip install soundfile scipy")
            return False

    print(f"[SUCCESS] Audio saved to: {out_path.resolve()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro Vietnamese TTS CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--text", "-t", 
        type=str, 
        required=True, 
        help="Vietnamese text to synthesize"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        required=True, 
        help="Output audio file path (e.g., output.wav)"
    )
    parser.add_argument(
        "--voice", "-v", 
        type=str, 
        default="diem_trinh", 
        help="Voice name (default: diem_trinh). See available voices below."
    )
    parser.add_argument(
        "--speed", "-s", 
        type=float, 
        default=1.0, 
        help="Speech speed multiplier (default: 1.0, recommended range: 0.75 - 1.25)"
    )
    
    args = parser.parse_args()
    
    print("[INFO] Loading model and voices...")
    model, voicepacks, device = load_model_and_voices()
    
    # Fallback if default voice isn't available
    chosen_voice = args.voice if args.voice in voicepacks else (list(voicepacks.keys())[0] if voicepacks else None)
    if not chosen_voice:
        print("[ERROR] No voices are available. Please check your ckpts directory.")
        sys.exit(1)
        
    if chosen_voice != args.voice:
        print(f"[WARN] Requested voice '{args.voice}' not found. Falling back to '{chosen_voice}'.")

    print(f"\n[INFO] Generating audio with voice='{chosen_voice}' at speed {args.speed}x...")
    success = generate_to_file(
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