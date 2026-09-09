from __future__ import annotations

import re
import numpy as np

_z0 = "contextboxai/Kokoro-Vietnamese"
_z1 = "kokoro_vi.pth"
_z2 = "kokoro_vi_voicepack.pt"
_z3 = "config.json"
_z4 = "diem_trinh"
_z5 = 24000
_z6 = 50

_z7 = {
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


def _z8(t: str) -> list[str]:
    _z9 = re.sub(r"\s+", " ", t.strip())
    if not _z9:
        return []
    _za: list[str] = []
    _zb = 0
    for _zc in re.finditer(r'[.!?…]+(?:[""\')])', _z9):
        _zd = _zc.end()
        if _zd < len(_z9) and not _z9[_zd].isspace():
            continue
        _ze = _z9[_zb:_zd].strip()
        if _ze:
            _za.append(_ze)
        _zb = _zd
    _zf = _z9[_zb:].strip()
    if _zf:
        _za.append(_zf)
    return _za


def _zg(chunks: list[np.ndarray], crossfade_samples: int) -> np.ndarray:
    _zh = [np.asarray(c, dtype=np.float32) for c in chunks if len(c) > 0]
    if not _zh:
        return np.array([], dtype=np.float32)
    _zi = _zh[0]
    for _zj in _zh[1:]:
        _zk = min(int(crossfade_samples), len(_zi), len(_zj))
        if _zk <= 0:
            _zi = np.concatenate([_zi, _zj])
            continue
        _zl = np.linspace(1.0, 0.0, _zk + 2, dtype=np.float32)[1:-1]
        _zm = 1.0 - _zl
        _zn = (_zi[-_zk:] * _zl) + (_zj[:_zk] * _zm)
        _zi = np.concatenate([_zi[:-_zk], _zn, _zj[_zk:]])
    return _zi.astype(np.float32, copy=False)


def _zo(t: str) -> str:
    from vig2p import phonemize_text
    return phonemize_text(t)


def _zp() -> list[str]:
    return sorted(_z7)
