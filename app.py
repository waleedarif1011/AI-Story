"""
Data-to-Story Generator — Gradio demo (CPU-friendly Flan-T5-base).

Hugging Face Space entrypoint. Boots on public ``google/flan-t5-base`` with no
private Google Drive files. Optionally set ``AI_STORY_ADAPTER`` to a local path
or public Hub LoRA repo ID.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional, Tuple

import gradio as gr

from src.inference import DEFAULT_MODEL_ID, GenerationConfig, StoryGenerator
from src.prompts import TONES, build_prompt

PRESETS: dict[str, str] = {
    "Churn model results": (
        "Churn model AUC=0.87; precision@10%=0.62; "
        "top drivers: tenure↓, support tickets↑, plan=Basic; lift vs baseline +31%."
    ),
    "Revenue anomaly": (
        "Revenue anomaly week 42: −18% vs 8-week MA; category Electronics −27%; "
        "region West −22%; no holiday shift; payment failures +9pp."
    ),
    "A/B checkout test": (
        "A/B checkout test: variant B vs A; n=42k; conversion 3.4%→3.9%; "
        "relative lift +14.7%; p=0.008; no significant AOV change."
    ),
    "Quarterly forecast": (
        "Q3 forecast: revenue $12.4M (80% PI $11.1–$13.6M); MAPE 6.2%; "
        "driver: new logo ARR +$1.1M; churn drag −$0.4M."
    ),
}

# Honest CPU expectations for Space / laptop demos (beam search on Flan-T5-base).
CPU_LATENCY_NOTE = (
    "**CPU latency (honest):** first boot downloads public "
    f"`{DEFAULT_MODEL_ID}` weights (~900MB). After load, expect roughly "
    "**5–20 seconds per story** on Hugging Face Space CPU basic / a typical laptop. "
    "No private Google Drive files are required — the demo runs on the base model "
    "unless you set `AI_STORY_ADAPTER` to a public Hub adapter."
)


@lru_cache(maxsize=1)
def get_generator() -> StoryGenerator:
    """Lazy-load base Flan-T5 (+ optional public/local adapter). Drive-free."""
    adapter = (os.environ.get("AI_STORY_ADAPTER") or "").strip() or None
    model_id = (os.environ.get("AI_STORY_MODEL_ID") or "").strip() or DEFAULT_MODEL_ID
    return StoryGenerator(model_id=model_id, adapter_path=adapter)


def apply_preset(name: str) -> str:
    return PRESETS.get(name, "")


def generate(
    findings: str,
    tone: str,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
) -> Tuple[str, str]:
    findings = (findings or "").strip()
    if not findings:
        return "", "Please enter structured findings or choose a preset."

    prompt = build_prompt(findings, tone=tone)
    try:
        generator = get_generator()
        story = generator.generate(
            findings,
            tone=tone,
            config=GenerationConfig(
                max_new_tokens=int(max_new_tokens),
                num_beams=int(num_beams),
                temperature=float(temperature),
                do_sample=float(temperature) > 0 and int(num_beams) <= 1,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — surface cleanly in the UI
        return prompt, f"Generation failed: {exc}"
    return prompt, story


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Data-to-Story Generator") as demo:
        gr.Markdown(
            f"""
            # Data-to-Story Generator
            Turn structured data-science findings into clear executive narratives.
            Powered by **{DEFAULT_MODEL_ID}** (CPU) with optional public LoRA adapters.

            {CPU_LATENCY_NOTE}
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    label="Preset scenario",
                    value="Churn model results",
                )
                findings = gr.Textbox(
                    label="Structured findings",
                    lines=8,
                    value=PRESETS["Churn model results"],
                    placeholder="Paste metrics, A/B results, KPI deltas, or an EDA summary…",
                )
                tone = gr.Radio(
                    choices=list(TONES),
                    value="Executive summary",
                    label="Tone",
                )
                with gr.Accordion("Generation settings", open=False):
                    max_new_tokens = gr.Slider(32, 512, value=256, step=8, label="max_new_tokens")
                    num_beams = gr.Slider(1, 8, value=4, step=1, label="num_beams")
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="temperature")
                run = gr.Button("Generate story", variant="primary")
            with gr.Column(scale=1):
                story = gr.Textbox(label="Generated narrative", lines=12)
                prompt_view = gr.Textbox(label="Structured prompt (side-by-side)", lines=12)

        preset.change(fn=apply_preset, inputs=preset, outputs=findings)
        run.click(
            fn=generate,
            inputs=[findings, tone, max_new_tokens, num_beams, temperature],
            outputs=[prompt_view, story],
        )
        gr.Examples(
            examples=[
                [PRESETS["Churn model results"], "Executive summary"],
                [PRESETS["Revenue anomaly"], "Stakeholder update"],
                [PRESETS["A/B checkout test"], "Technical report"],
                [PRESETS["Quarterly forecast"], "Executive summary"],
            ],
            inputs=[findings, tone],
        )
    return demo


def main(share: bool = False, server_name: Optional[str] = None) -> None:
    demo = build_ui()
    demo.launch(
        share=share,
        server_name=server_name or os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
    )


if __name__ == "__main__":
    main()
