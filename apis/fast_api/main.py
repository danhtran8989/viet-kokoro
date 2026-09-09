import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import io
import json
import struct
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from kokoro_vietnamese._kokoro import KModel
from kokoro_vietnamese.core import (
    SAMPLE_RATE,
    VOICES,
    split_text,
    merge_audio_chunks,
    phonemize,
    get_device,
)

REPO_ID = "contextboxai/Kokoro-Vietnamese"
CKPTS_DIR = Path(__file__).resolve().parent.parent.parent / "ckpts" / "Kokoro-Vietnamese"

app = FastAPI(title="Kokoro Vietnamese TTS API")

device = get_device()
print(f"[INFO] Using device: {device}")


def _resolve_file(filename: str) -> str:
    local = CKPTS_DIR / filename
    if local.exists():
        print(f"[INFO] Using local checkpoint: {local}")
        return str(local)
    print(f"[INFO] Downloading from HF: {REPO_ID}/{filename}")
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


_config_path = _resolve_file("config.json")
_model_path = _resolve_file("kokoro_vi.pth")
_voicepack_path = _resolve_file("kokoro_vi_voicepack.pt")

with open(_config_path, "r", encoding="utf-8") as _f:
    _config = json.load(_f)

model = KModel(
    repo_id="hexgrad/Kokoro-82M",
    config=_config,
    model=_model_path,
).to(device).eval()
voicepack = torch.load(_voicepack_path, map_location="cpu", weights_only=True)


class TTSRequest(BaseModel):
    text: str
    voice: str = "diem_trinh"
    speed: float = 1.0


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(audio_int16)
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<H", 2))
    buf.write(struct.pack("<H", 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_int16.tobytes())
    return buf.getvalue()


@app.post("/tts")
async def tts_generate(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    audio_chunks = []
    for chunk_text in split_text(req.text):
        ps = phonemize(chunk_text)
        if not ps:
            continue
        if len(ps) > 510:
            raise HTTPException(status_code=400, detail=f"Phoneme chunk too long ({len(ps)} > 510)")
        with torch.no_grad():
            ref_s = voicepack[len(ps) - 1]
            audio = model(ps, ref_s, float(req.speed))
        audio_chunks.append(audio.detach().cpu().numpy())

    if not audio_chunks:
        raise HTTPException(status_code=400, detail="No audio generated")

    crossfade_samples = round(SAMPLE_RATE * 50 / 1000)
    audio = merge_audio_chunks(audio_chunks, crossfade_samples)
    wav_bytes = audio_to_wav_bytes(audio, SAMPLE_RATE)

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


@app.get("/voices")
async def list_voices():
    return {"voices": VOICES}


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}
