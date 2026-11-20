"""Tests for prompt formatting, JSONL schema, and mocked inference smoke."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.prompts import TONES, build_prompt, format_training_example
from src.train_lora import REQUIRED_FIELDS, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSONL = ROOT / "data" / "sample_ds_stories.jsonl"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_build_prompt_from_string():
    prompt = build_prompt("AUC=0.87; lift +31%", tone="Executive summary")
    assert "Findings:" in prompt
    assert "AUC=0.87" in prompt
    assert "Executive summary" in prompt
    assert prompt.strip().endswith("Narrative:")


def test_build_prompt_from_metric_dict():
    prompt = build_prompt(
        {"auc": 0.87, "lift_pct": 31, "drivers": ["tenure", "tickets"]},
        tone="Technical report",
    )
    assert "- auc: 0.87" in prompt
    assert "- lift_pct: 31" in prompt
    assert "Technical report" in prompt


def test_build_prompt_from_bullet_list():
    prompt = build_prompt(
        ["conversion 3.4%→3.9%", "p=0.008"],
        tone="Stakeholder update",
    )
    assert "- conversion 3.4%→3.9%" in prompt
    assert "- p=0.008" in prompt


@pytest.mark.parametrize("tone", list(TONES))
def test_all_tones_accepted(tone: str):
    prompt = build_prompt("KPI delta +5%", tone=tone)
    assert tone in prompt


def test_invalid_tone_raises():
    with pytest.raises(ValueError, match="tone"):
        build_prompt("x", tone="Poetry")


def test_empty_findings_raise():
    with pytest.raises(ValueError):
        build_prompt("")
    with pytest.raises(ValueError):
        build_prompt({})
    with pytest.raises(ValueError):
        build_prompt([])


def test_format_training_example():
    row = format_training_example(
        "AUC=0.9",
        human_storytelling_narrative="The model performs well.",
    )
    assert row["complex_data_science_result"] == "AUC=0.9"
    assert "Narrative:" in row["prompt"]
    assert row["human_storytelling_narrative"].startswith("The model")


# ---------------------------------------------------------------------------
# JSONL schema validation
# ---------------------------------------------------------------------------


def test_sample_jsonl_exists_and_has_25_rows():
    assert SAMPLE_JSONL.is_file()
    rows = load_jsonl(SAMPLE_JSONL)
    assert len(rows) == 25


def test_sample_jsonl_required_fields():
    rows = load_jsonl(SAMPLE_JSONL)
    for row in rows:
        for field in REQUIRED_FIELDS:
            assert field in row
            assert isinstance(row[field], str)
            assert row[field].strip()


def test_sample_jsonl_raw_parse():
    with SAMPLE_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            assert set(REQUIRED_FIELDS).issubset(obj.keys())


def test_load_jsonl_rejects_bad_row(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"complex_data_science_result": "only one field"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="human_storytelling_narrative"):
        load_jsonl(bad)


def test_four_presets_match_sample_themes():
    """App presets should align with the first four curated sample themes."""
    from app import PRESETS

    assert len(PRESETS) == 4
    rows = load_jsonl(SAMPLE_JSONL)
    # First four sample rows are the four demo scenarios.
    sample_inputs = {r["complex_data_science_result"] for r in rows[:4]}
    for text in PRESETS.values():
        assert text in sample_inputs


# ---------------------------------------------------------------------------
# Mocked inference smoke
# ---------------------------------------------------------------------------


def test_story_generator_generate_mocked():
    from src.inference import GenerationConfig, StoryGenerator

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    # tokenizer(...) returns a BatchEncoding-like mapping; .to on tensors mocked below
    encoded = {
        "input_ids": MagicMock(**{"to.return_value": "ids"}),
        "attention_mask": MagicMock(**{"to.return_value": "mask"}),
    }
    fake_tokenizer.side_effect = None
    fake_tokenizer.return_value = encoded
    fake_tokenizer.decode.return_value = "  Mocked executive narrative.  "

    fake_model = MagicMock()
    fake_model.generate.return_value = [[1, 2, 3]]
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
        patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=fake_model),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.no_grad", return_value=MagicMock(**{"__enter__": lambda s: None, "__exit__": lambda *a: None})),
    ):
        # Re-bind tokenizer call: StoryGenerator assigns self.tokenizer = from_pretrained(...)
        gen = StoryGenerator.__new__(StoryGenerator)
        gen.model_id = "google/flan-t5-base"
        gen.adapter_path = None
        gen.device = "cpu"
        gen.tokenizer = fake_tokenizer
        gen.model = fake_model

        # Make tokenizer callable return encoded dict with .to() on values
        def _tok(*_a, **_k):
            return {
                "input_ids": MagicMock(**{"to.return_value": "ids"}),
                "attention_mask": MagicMock(**{"to.return_value": "mask"}),
            }

        gen.tokenizer = MagicMock(side_effect=_tok)
        gen.tokenizer.decode.return_value = "  Mocked executive narrative.  "

        out = gen.generate(
            "AUC=0.87",
            tone="Executive summary",
            config=GenerationConfig(max_new_tokens=64, num_beams=2, temperature=1.0),
        )

    assert out == "Mocked executive narrative."
    fake_model.generate.assert_called_once()
    gen.tokenizer.decode.assert_called_once()


def test_build_prompt_used_by_app_generate_path():
    """Ensure app.generate returns the prompt even when the model fails."""
    from app import generate

    with patch("app.get_generator", side_effect=RuntimeError("no weights")):
        prompt, story = generate(
            "AUC=0.87; lift +31%",
            "Executive summary",
            64,
            2,
            0.7,
        )
    assert "Findings:" in prompt
    assert "Generation failed" in story
