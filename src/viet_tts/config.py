from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

_z = False

_y = Path(__file__).resolve().parent.parent.parent
_x = _y / "secrets" / ".env"

def _w():
    global _z
    if not _z:
        load_dotenv(_x, override=False)
        _z = True

def _v(k: str, d: str = "") -> str:
    _w()
    return os.getenv(k, d)

_a: str = _v("KOKORO_HF_REPO", "contextboxai/Kokoro-Vietnamese")
_b: str = _v("KOKORO_MODEL_FILE", "kokoro_vi.pth")
_c: str = _v("KOKORO_VOICEPACK_FILE", "kokoro_vi_voicepack.pt")
_d: str = _v("KOKORO_CONFIG_FILE", "config.json")
_e: str = _v("KOKORO_BASE_REPO", "hexgrad/Kokoro-82M")

_f: Path = _y / "ckpts"

def _g(h: str) -> Path:
    return _f / h
