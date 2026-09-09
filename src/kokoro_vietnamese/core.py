from __future__ import annotations

import re
import numpy as np

DEFAULT_HF_REPO_ID = "contextboxai/Kokoro-Vietnamese"
DEFAULT_MODEL_FILE = "kokoro_vi.pth"
DEFAULT_VOICEPACK_FILE = "kokoro_vi_voicepack.pt"
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_VOICE = "diem_trinh"
SAMPLE_RATE = 24000
DEFAULT_CROSSFADE_MS = 50

VOICES = {
    "diem_trinh": {"label": "Diễm Trinh", "filename": "voicepacks/diem_trinh.pt"},
    "hung_thinh": {"label": "Hưng Thịnh", "filename": "voicepacks/hung_thinh.pt"},
    "mai_linh": {"label": "Mai Linh", "filename": "voicepacks/mai_linh.pt"},
    "mai_loan": {"label": "Mai Loan", "filename": "voicepacks/mai_loan.pt"},
    "manh_dung": {"label": "Mạnh Dũng", "filename": "voicepacks/manh_dung.pt"},
    "my_yen": {"label": "Mỹ Yến", "filename": "voicepacks/my_yen.pt"},
    "ngoc_huyen": {"label": "Ngọc Huyền", "filename": "voicepacks/ngoc_huyen.pt"},
    "phat_tai": {"label": "Phát Tài", "filename": "voicepacks/phat_tai.pt"},
    "thanh_dat": {"label": "Thành Đạt", "filename": "voicepacks/thanh_dat.pt"},
    "thuc_trinh": {"label": "Thục Trinh", "filename": "voicepacks/thuc_trinh.pt"},
    "tuan_ngoc": {"label": "Tuấn Ngọc", "filename": "voicepacks/tuan_ngoc.pt"},
    "storyvert": {"label": "storyvert", "filename": "voicepacks/storyvert.pt"},
    "duc_an": {"label": "Đức An", "filename": "voicepacks/duc_an.pt"},
    "duc_duy": {"label": "Đức Duy", "filename": "voicepacks/duc_duy.pt"},
}


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_text(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    for match in re.finditer(r'[.!?…]+(?:[""\')])', normalized):
        end = match.end()
        if end < len(normalized) and not normalized[end].isspace():
            continue
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    remainder = normalized[start:].strip()
    if remainder:
        chunks.append(remainder)
    return chunks


def merge_audio_chunks(chunks: list[np.ndarray], crossfade_samples: int) -> np.ndarray:
    valid = [np.asarray(c, dtype=np.float32) for c in chunks if len(c) > 0]
    if not valid:
        return np.array([], dtype=np.float32)
    merged = valid[0]
    for chunk in valid[1:]:
        overlap = min(int(crossfade_samples), len(merged), len(chunk))
        if overlap <= 0:
            merged = np.concatenate([merged, chunk])
            continue
        fade_out = np.linspace(1.0, 0.0, overlap + 2, dtype=np.float32)[1:-1]
        fade_in = 1.0 - fade_out
        crossfaded = (merged[-overlap:] * fade_out) + (chunk[:overlap] * fade_in)
        merged = np.concatenate([merged[:-overlap], crossfaded, chunk[overlap:]])
    return merged.astype(np.float32, copy=False)


def phonemize(text: str) -> str:
    from vig2p import phonemize_text
    return phonemize_text(text)


def list_voices() -> list[str]:
    return sorted(VOICES)
