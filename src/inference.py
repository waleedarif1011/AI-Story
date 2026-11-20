"""CPU-friendly Flan-T5 inference for data-science → narrative generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.prompts import TONES, build_prompt

DEFAULT_MODEL_ID = "google/flan-t5-base"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    num_beams: int = 4
    temperature: float = 0.7
    do_sample: bool = False


class StoryGenerator:
    """Load ``google/flan-t5-base`` (optionally with a LoRA adapter) and generate narratives."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model.to(self.device)
        self.model.eval()

    def generate(
        self,
        findings: str | Mapping[str, Any] | list[Any],
        tone: str = "Executive summary",
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate a narrative from structured findings."""
        import torch

        if tone not in TONES:
            raise ValueError(f"tone must be one of: {', '.join(TONES)}")

        cfg = config or GenerationConfig()
        prompt = build_prompt(findings, tone=tone)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "num_beams": cfg.num_beams,
        }
        if cfg.do_sample or cfg.temperature != 1.0:
            # Sampling path when temperature is used intentionally.
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(cfg.temperature, 1e-5)
            gen_kwargs["num_beams"] = 1
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**encoded, **gen_kwargs)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()


def generate_story(
    findings: str | Mapping[str, Any] | list[Any],
    tone: str = "Executive summary",
    model_id: str = DEFAULT_MODEL_ID,
    adapter_path: Optional[str] = None,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    temperature: float = 0.7,
) -> str:
    """One-shot helper that constructs a ``StoryGenerator`` and returns a narrative."""
    generator = StoryGenerator(model_id=model_id, adapter_path=adapter_path)
    return generator.generate(
        findings,
        tone=tone,
        config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=temperature > 0 and temperature != 1.0 and num_beams <= 1,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a DS narrative with Flan-T5.")
    parser.add_argument("findings", help="Free-text data-science findings")
    parser.add_argument(
        "--tone",
        default="Executive summary",
        choices=list(TONES),
        help="Narrative tone",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path or Hub ID")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    story = generate_story(
        args.findings,
        tone=args.tone,
        model_id=args.model_id,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
    )
    print(story)
