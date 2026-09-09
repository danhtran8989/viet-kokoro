import os
import sys
import json
import numpy as np
import torch
import gradio as gr

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

import transformers.utils.import_utils as _s4
for _s5 in ("_torchvision_available", "_librosa_available", "_cv2_available"):
    if hasattr(_s4, _s5):
        setattr(_s4, _s5, False)
if hasattr(_s4, "_torchvision_version"):
    _s4._torchvision_version = "N/A"


_s6 = "cuda" if torch.cuda.is_available() else "cpu"


def _s7():
    _s8 = _f / _d
    _s9 = _f / _b
    _sa = _f / _c

    if _s8.exists() and _s9.exists() and _sa.exists():
        _sb = str(_s8)
        _sc = str(_s9)
        _sd = str(_sa)
    else:
        _sb = hf_hub_download(repo_id=_a, filename=_d)
        _sc = hf_hub_download(repo_id=_a, filename=_b)
        _sd = hf_hub_download(repo_id=_a, filename=_c)

    with open(_sb, "r", encoding="utf-8") as _se:
        _sf = json.load(_se)

    _sg = _q0(
        _q1=_e,
        _q2=_sf,
        _q3=_sc,
    ).to(_s6).eval()

    _sh = torch.load(_sd, map_location="cpu", weights_only=True)
    return _sg, _sh


_si, _sj = _s7()

_sk = [(info["label"], name) for name, info in _z7.items()]

_sl = [
    [
        "Giua mot buoi chieu yen tinh, co ay ke lai cau chuyen bang mot giong noi am ap va cham roi.",
        "diem_trinh",
        1.0,
    ],
    [
        "Sang nay, thanh pho thuc day trong lan suong mong, con nhung con duong thi bat dau ron rang tieng xe.",
        "mai_linh",
        1.0,
    ],
    [
        "Neu ban lang nghe that ky, ban se nghe thay tieng mua roi nhe tren mai hien sau nha.",
        "ngoc_huyen",
        0.95,
    ],
    [
        "Ban tin hom nay ghi nhan nhieu tin hieu tich cuc tu thi truong, dac biet la nhom cong nghe va tieu dung.",
        "hung_thinh",
        1.03,
    ],
    [
        "Hanh trinh qua mien Trung de lai trong toi ky uc ve nang, gio, bien xanh va nhung bua com rat dam da.",
        "tuan_ngoc",
        1.0,
    ],
    [
        "Mot podcast hay khong chi can noi dung tot, ma con can nhip ke du cuon hut de giu nguoi nghe o lai.",
        "storyvert",
        1.0,
    ],
]


def _sm(_sn: str, _so: str, _sp: float) -> tuple:
    if not _sn or not _sn.strip():
        return None, "", "Vui long nhap van ban tieng Viet."
    _sq: list[np.ndarray] = []
    _sr: list[str] = []
    for _ss, _st in enumerate(_z8(_sn), start=1):
        _su = _zo(_st)
        if not _su:
            continue
        if len(_su) > 510:
            raise ValueError(
                f"Phoneme chunk too long ({len(_su)} > 510): {_st[:80]}"
            )
        with torch.no_grad():
            _sv = _sj[len(_su) - 1]
            _sw = _si(_su, _sv, float(_sp))
        _sr.append(f"[{_ss}] {_su}")
        _sq.append(_sw.detach().cpu().numpy())
    if not _sq:
        return None, "", "Khong tao duoc audio."
    _sx = round(_z5 * 50 / 1000)
    _sy = _zg(_sq, _sx)
    _sz = _z7.get(_so, {}).get("label", _so)
    return (_z5, _sy), "\n".join(_sr), f"Voice: {_sz}"


_t0 = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(theme=gr.themes.Citrus(), css=_t0) as _t1:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Vietnamese TTS")
        gr.Markdown(
            "Vietnamese text-to-speech demo using "
            "[contextboxai/Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese), "
            "a fine-tune of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) "
            "with Vietnamese G2P via [vig2p](https://pypi.org/project/vig2p/)."
        )
        with gr.Row():
            with gr.Column(scale=2):
                _t2 = gr.Textbox(
                    label="Vietnamese Text",
                    lines=5,
                    value=_sl[0][0],
                    placeholder="Nhap van ban tieng Viet...",
                )
                with gr.Row():
                    _t3 = gr.Dropdown(
                        label="Voice",
                        choices=_sk,
                        value="diem_trinh",
                    )
                    _t4 = gr.Slider(
                        minimum=0.75, maximum=1.25, value=1.0, step=0.01,
                        label="Speed",
                    )
                _t5 = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                _t6 = gr.Audio(label="Audio", type="numpy")
                _t7 = gr.Textbox(label="Status", interactive=False)
        _t8 = gr.Textbox(
            label="Phonemes", lines=4, interactive=False,
        )
        gr.Examples(
            examples=_sl,
            inputs=[_t2, _t3, _t4],
            outputs=[_t6, _t8, _t7],
            fn=_sm,
            cache_examples=True,
            cache_mode="lazy",
        )
        _t5.click(
            fn=_sm,
            inputs=[_t2, _t3, _t4],
            outputs=[_t6, _t8, _t7],
            api_name="generate",
        )

if __name__ == "__main__":
    _t1.launch()
