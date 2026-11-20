"""Fine-tune google/flan-t5-base with PEFT LoRA on JSONL DS→story pairs.

Extracted from ``fine_tune_flan_t5_base.ipynb`` for local / Colab reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.prompts import build_prompt

DEFAULT_MODEL_ID = "google/flan-t5-base"
DEFAULT_DATA = Path("data/sample_ds_stories.jsonl")
REQUIRED_FIELDS = ("complex_data_science_result", "human_storytelling_narrative")


def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load and validate a JSONL training file."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            for field in REQUIRED_FIELDS:
                value = obj.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{path}:{line_no}: missing or empty required field '{field}'"
                    )
            rows.append(
                {
                    "complex_data_science_result": obj["complex_data_science_result"].strip(),
                    "human_storytelling_narrative": obj[
                        "human_storytelling_narrative"
                    ].strip(),
                }
            )
    if not rows:
        raise ValueError(f"{path}: no training examples found")
    return rows


def prepare_datasets(rows: list[dict[str, str]], val_ratio: float = 0.2, seed: int = 42):
    """Convert rows to Hugging Face datasets with instruction prompts."""
    from datasets import Dataset

    prompted = [
        {
            "input_text": build_prompt(row["complex_data_science_result"]),
            "target_text": row["human_storytelling_narrative"],
        }
        for row in rows
    ]
    dataset = Dataset.from_list(prompted)
    if len(dataset) < 5 or val_ratio <= 0:
        return dataset, None
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def tokenize_datasets(train_ds, val_ds, tokenizer, max_source_length: int, max_target_length: int):
    def _tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=max_source_length,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            truncation=True,
            padding="max_length",
            max_length=max_target_length,
        )
        # Mask pad tokens in labels so they do not contribute to loss.
        pad_id = tokenizer.pad_token_id
        label_ids = [
            [(tok if tok != pad_id else -100) for tok in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    train_tok = train_ds.map(_tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tok = None
    if val_ds is not None:
        val_tok = val_ds.map(_tokenize, batched=True, remove_columns=val_ds.column_names)
    return train_tok, val_tok


def train(
    data_path: Path = DEFAULT_DATA,
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: Path = Path("adapters/flan-t5-base-lora"),
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    max_source_length: int = 512,
    max_target_length: int = 256,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Run LoRA fine-tuning and save adapters to ``output_dir``."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    rows = load_jsonl(data_path)
    train_ds, val_ds = prepare_datasets(rows, val_ratio=val_ratio, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "k", "v"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_tok, val_tok = tokenize_datasets(
        train_ds, val_ds, tokenizer, max_source_length, max_target_length
    )

    # Disable torch dynamo compilation for broader CPU/Colab compatibility.
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.disable = True

    output_dir.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch" if val_tok is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=val_tok is not None,
        metric_for_best_model="eval_loss" if val_tok is not None else None,
        predict_with_generate=False,
        report_to=[],
        seed=seed,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapters to {output_dir}")
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Flan-T5-base with LoRA on DS→story JSONL data."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("adapters/flan-t5-base-lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train(
        data_path=args.data,
        model_id=args.model_id,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
