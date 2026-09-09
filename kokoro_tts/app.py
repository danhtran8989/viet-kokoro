import spaces  # MUST come before any torch/CUDA-touching import
import os
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
)

REPO_ID = "contextboxai/Kokoro-Vietnamese"

# --- Disable transformers' optional-vision/audio probes that break on ZeroGPU ---
import transformers.utils.import_utils as _import_utils
for _flag in ("_torchvision_available", "_librosa_available", "_cv2_available"):
    if hasattr(_import_utils, _flag):
        setattr(_import_utils, _flag, False)
if hasattr(_import_utils, "_torchvision_version"):
    _import_utils._torchvision_version = "N/A"

# --- Download model artifacts at module scope ---
_config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
_model_path = hf_hub_download(repo_id=REPO_ID, filename="kokoro_vi.pth")
_voicepack_path = hf_hub_download(repo_id=REPO_ID, filename="kokoro_vi_voicepack.pt")

with open(_config_path, "r", encoding="utf-8") as _f:
    _config = json.load(_f)

# --- Load the KModel eagerly at module scope and move to "cuda" ---
model = KModel(
    repo_id="hexgrad/Kokoro-82M",
    config=_config,
    model=_model_path,
).to("cuda").eval()
voicepack = torch.load(_voicepack_path, map_location="cpu", weights_only=True)

VOICE_CHOICES = [(info["label"], name) for name, info in VOICES.items()]

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


@spaces.GPU(duration=30)
def generate(text: str, voice: str, speed: float) -> tuple:
    """Synthesize Vietnamese speech from text.

    Args:
        text: Vietnamese text to synthesize.
        voice: Voice name (one of the available Vietnamese voices).
        speed: Speech speed multiplier (0.75–1.25).
    """
    if not text or not text.strip():
        return None, "", "Vui lòng nhập văn bản tiếng Việt."
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
            ref_s = voicepack[len(ps) - 1]
            audio = model(ps, ref_s, float(speed))
        phoneme_chunks.append(f"[{index}] {ps}")
        audio_chunks.append(audio.detach().cpu().numpy())
    if not audio_chunks:
        return None, "", "Không tạo được audio."
    crossfade_samples = round(SAMPLE_RATE * 50 / 1000)
    audio = merge_audio_chunks(audio_chunks, crossfade_samples)
    label = VOICES.get(voice, {}).get("label", voice)
    return (SAMPLE_RATE, audio), "\n".join(phoneme_chunks), f"Voice: {label}"


CSS = """
#col-container { max-width: 1100px; margin: 0 auto; }
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(theme=gr.themes.Citrus(), css=CSS) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🇻🇳 Kokoro Vietnamese TTS")
        gr.Markdown(
            "Vietnamese text-to-speech demo using "
            "[contextboxai/Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese), "
            "a fine-tune of [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) "
            "with Vietnamese G2P via [vig2p](https://pypi.org/project/vig2p/)."
        )
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Vietnamese Text",
                    lines=5,
                    value=DEMO_EXAMPLES[0][0],
                    placeholder="Nhập văn bản tiếng Việt...",
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
    demo.launch(mcp_server=True)