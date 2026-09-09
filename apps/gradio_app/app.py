import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
import numpy as np
import torch
import gradio as gr
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

REPO_ID = "contextboxai/Kokoro-Vietnamese"
CKPTS_DIR = Path(__file__).resolve().parent.parent.parent / "ckpts" / "Kokoro-Vietnamese"

import transformers.utils.import_utils as _import_utils
for _flag in ("_torchvision_available", "_librosa_available", "_cv2_available"):
    if hasattr(_import_utils, _flag):
        setattr(_import_utils, _flag, False)
if hasattr(_import_utils, "_torchvision_version"):
    _import_utils._torchvision_version = "N/A"


def _resolve_file(filename: str) -> str:
    local = CKPTS_DIR / filename
    if local.exists():
        print(f"[INFO] Using local checkpoint: {local}")
        return str(local)
    print(f"[INFO] Downloading from HF: {REPO_ID}/{filename}")
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


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

_voices_json = CKPTS_DIR / "voices.json"
if _voices_json.exists():
    with open(_voices_json, "r", encoding="utf-8") as _f:
        VOICES = json.load(_f)

voicepacks: dict[str, torch.Tensor] = {}
for _vname, _vinfo in VOICES.items():
    _vp_path = CKPTS_DIR / _vinfo["filename"]
    if _vp_path.exists():
        voicepacks[_vname] = torch.load(_vp_path, map_location="cpu", weights_only=True)
        print(f"[INFO] Loaded voice: {_vname} from {_vp_path}")
    else:
        print(f"[WARN] Voice file not found: {_vp_path}")

VOICE_CHOICES = [(info["label"], name) for name, info in VOICES.items()]
device = _device

DEMO_EXAMPLES = [
    [
        "Giữa một buổi chiều yên tĩnh, cô ấy kể lại câu chuyện bằng một giọng nói ấm áp và chậm rãi.",
        "diem_trinh",
        1.0,
    ],
    [
        "Sáng nay, thành phố thức dậy trong làn sương mỏng, còn những con đường thì bắt đầu rộn ràng tiếng xe.",
        "mai_linh",
        1.0,
    ],
    [
        "Nếu bạn lắng nghe thật kỹ, bạn sẽ nghe thấy tiếng mưa rơi nhẹ trên mái hiên sau nhà.",
        "ngoc_huyen",
        0.95,
    ],
    [
        "Bản tin hôm nay ghi nhận nhiều tín hiệu tích cực từ thị trường, đặc biệt là nhóm công nghệ và tiêu dùng.",
        "hung_thinh",
        1.03,
    ],
    [
        "Hành trình qua miền Trung để lại trong tôi ký ức về nắng, gió, biển xanh và những bữa cơm rất đậm đà.",
        "tuan_ngoc",
        1.0,
    ],
    [
        "Một podcast hay không chỉ cần nội dung tốt, mà còn cần nhịp kể đủ cuốn hút để giữ người nghe ở lại.",
        "storyvert",
        1.0,
    ],
]


def generate(text: str, voice: str, speed: float) -> tuple:
    if not text or not text.strip():
        return None, "", "Please enter Vietnamese text."
    vp = voicepacks.get(voice)
    if vp is None:
        return None, "", f"Voice '{voice}' not found."
    audio_chunks: list[np.ndarray] = []
    phoneme_chunks: list[str] = []
    for index, chunk_text in enumerate(split_text(text), start=1):
        ps = phonemize(chunk_text)
        if not ps:
            continue
        if len(ps) > 510:
            raise ValueError(
                f"Phoneme chunk too long ({len(ps)} > 510): {chunk_text[:80]}"
            )
        with torch.no_grad():
            ref_s = vp[len(ps) - 1]
            audio = model(ps, ref_s, float(speed))
        phoneme_chunks.append(f"[{index}] {ps}")
        audio_chunks.append(audio.detach().cpu().numpy())
    if not audio_chunks:
        return None, "", "No audio generated."
    crossfade_samples = round(SAMPLE_RATE * 50 / 1000)
    audio = merge_audio_chunks(audio_chunks, crossfade_samples)
    label = VOICES.get(voice, {}).get("label", voice)
    return (SAMPLE_RATE, audio), "\n".join(phoneme_chunks), f"Voice: {label} | Device: {device}"


CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(theme=gr.themes.Citrus(), css=CSS) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🇻🇳 Kokoro Vietnamese TTS")
        gr.Markdown(
            "Vietnamese text-to-speech using "
            "[contextboxai/Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese), "
            f"running on **{device}**."
        )
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Vietnamese Text",
                    lines=5,
                    value=DEMO_EXAMPLES[0][0],
                    placeholder="Enter Vietnamese text...",
                )
                with gr.Row():
                    voice = gr.Dropdown(
                        label="Voice",
                        choices=VOICE_CHOICES,
                        value="diem_trinh",
                    )
                    speed = gr.Slider(
                        minimum=0.75, maximum=1.25, value=1.0, step=0.01,
                        label="Speed",
                    )
                submit = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                audio_out = gr.Audio(label="Audio", type="numpy")
                status = gr.Textbox(label="Status", interactive=False)
        phonemes = gr.Textbox(
            label="Phonemes", lines=4, interactive=False,
        )
        gr.Examples(
            examples=DEMO_EXAMPLES,
            inputs=[text, voice, speed],
            outputs=[audio_out, phonemes, status],
            fn=generate,
            cache_examples=True,
            cache_mode="lazy",
        )
        submit.click(
            fn=generate,
            inputs=[text, voice, speed],
            outputs=[audio_out, phonemes, status],
            api_name="generate",
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kokoro Vietnamese TTS Gradio App")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
