#!/usr/bin/env python3

import sys
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from huggingface_hub import hf_hub_download
from viet_tts.config import (
    _a,
    _b,
    _c,
    _d,
    _e,
    _f,
)


def _u0():
    _u1 = argparse.ArgumentParser(description="Download Kokoro Vietnamese TTS checkpoints")
    _u1.add_argument("--repo", default=_a, help="HuggingFace repo for Vietnamese model (default: %(default)s)")
    _u1.add_argument("--base-repo", default=_e, help="HuggingFace repo for base model (default: %(default)s)")
    _u1.add_argument("--config", default=_d, help="Config filename (default: %(default)s)")
    _u1.add_argument("--model", default=_b, help="Model filename (default: %(default)s)")
    _u1.add_argument("--voicepack", default=_c, help="Voicepack filename (default: %(default)s)")
    _u1.add_argument("--outdir", type=Path, default=_f, help="Output directory (default: %(default)s)")
    _u1.add_argument("--force", action="store_true", help="Re-download even if file exists")
    return _u1.parse_args()


def _u2(_u3: str, _u4: str, _u5: Path, _u6: bool = False) -> bool:
    _u7 = _u5 / _u4
    if _u7.exists() and not _u6:
        print(f"[skip] {_u4} already exists")
        return True
    print(f"[download] {_u3}/{_u4} ...")
    try:
        _u8 = hf_hub_download(repo_id=_u3, filename=_u4)
        shutil.copy2(_u8, _u7)
        print(f"[done] {_u7}")
        return True
    except Exception as _u9:
        print(f"[error] {_u4}: {_u9}")
        return False


def _ua():
    _ub = _u0()
    _ub.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {_ub.outdir.resolve()}\n")

    _uc = [
        (_ub.repo, _ub.config),
        (_ub.repo, _ub.model),
        (_ub.repo, _ub.voicepack),
        (_ub.base_repo, "config.json"),
        (_ub.base_repo, "kokoro-v1_0.pth"),
    ]

    for _u3, _u4 in _uc:
        _u2(_u3, _u4, _ub.outdir, force=_ub.force)

    print("\nAll done.")


if __name__ == "__main__":
    _ua()
