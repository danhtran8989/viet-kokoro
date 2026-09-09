import os
import sys
import io
import json
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from viet_tts._kokoro import _q0
from viet_tts.core import (
    _z5,
    _z7,
    _z8,
    _zg,
    _zo,
)
from viet_tts.config import (
    _a,
    _e,
    _d,
    _b,
    _c,
    _f,
)

from huggingface_hub import hf_hub_download

_ra = "cuda" if torch.cuda.is_available() else "cpu"


def _rb():
    _rc = _f / _d
    _rd = _f / _b
    _re = _f / _c

    if _rc.exists() and _rd.exists() and _re.exists():
        _rf = str(_rc)
        _rg = str(_rd)
        _rh = str(_re)
    else:
        _rf = hf_hub_download(repo_id=_a, filename=_d)
        _rg = hf_hub_download(repo_id=_a, filename=_b)
        _rh = hf_hub_download(repo_id=_a, filename=_c)

    with open(_rf, "r", encoding="utf-8") as _ri:
        _rj = json.load(_ri)

    _rk = _q0(
        repo_id=_e,
        config=_rj,
        model=_rg,
    ).to(_ra).eval()

    _rl = torch.load(_rh, map_location="cpu", weights_only=True)
    return _rk, _rl


_rm, _rn = _rb()

_ro = FastAPI(title="Kokoro Vietnamese TTS API", version="1.0.0")


class _rp(BaseModel):
    text: str = Field(..., description="Vietnamese text to synthesize")
    voice: str = Field(default="diem_trinh", description="Voice name")
    speed: float = Field(default=1.0, ge=0.75, le=1.25, description="Speed multiplier")


class _rq(BaseModel):
    voice: str
    duration_seconds: float


@_ro.get("/voices")
def _rr():
    return {"voices": {k: v["label"] for k, v in _z7.items()}}


@_ro.post("/tts", response_class=StreamingResponse)
def _rs(_rt: _rp):
    if not _rt.text or not _rt.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    _ru = _z7.get(_rt.voice)
    if not _ru:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {_rt.voice}")

    _rv: list[np.ndarray] = []
    for _rw in _z8(_rt.text):
        _rx = _zo(_rw)
        if not _rx:
            continue
        if len(_rx) > 510:
            raise HTTPException(status_code=400, detail=f"Phoneme chunk too long ({len(_rx)} > 510)")
        with torch.no_grad():
            _ry = _rn[len(_rx) - 1]
            _rz = _rm(_rx, _ry, float(_rt.speed))
        _rv.append(_rz.detach().cpu().numpy())

    if not _rv:
        raise HTTPException(status_code=400, detail="Could not generate audio")

    _s0 = round(_z5 * 50 / 1000)
    _s1 = _zg(_rv, _s0)

    _s2 = io.BytesIO()
    sf.write(_s2, _s1, _z5, format="WAV")
    _s2.seek(0)

    return StreamingResponse(
        _s2,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
    )


@_ro.post("/tts/json")
def _s3(_rt: _rp):
    if not _rt.text or not _rt.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    _ru = _z7.get(_rt.voice)
    if not _ru:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {_rt.voice}")

    _rv: list[np.ndarray] = []
    for _rw in _z8(_rt.text):
        _rx = _zo(_rw)
        if not _rx:
            continue
        if len(_rx) > 510:
            raise HTTPException(status_code=400, detail=f"Phoneme chunk too long ({len(_rx)} > 510)")
        with torch.no_grad():
            _ry = _rn[len(_rx) - 1]
            _rz = _rm(_rx, _ry, float(_rt.speed))
        _rv.append(_rz.detach().cpu().numpy())

    if not _rv:
        raise HTTPException(status_code=400, detail="Could not generate audio")

    _s0 = round(_z5 * 50 / 1000)
    _s1 = _zg(_rv, _s0)

    return {
        "voice": _ru["label"],
        "sample_rate": _z5,
        "duration_seconds": round(len(_s1) / _z5, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(_ro, host="0.0.0.0", port=8000)
