# Hugging Face Space deploy

The Gradio **Data-to-Story Generator** is Space-ready. A live Space was **not**
created from this agent run because no Hugging Face write token (`HF_TOKEN`) was
available in the environment.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create ai-story --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/ai-story
git push space main
```

3. Confirm the Space builds and the four presets behave as expected:
   - Churn model results
   - Revenue anomaly
   - A/B checkout test
   - Quarterly forecast
4. Optional: set Space secret `AI_STORY_ADAPTER` to a Hub LoRA repo ID after fine-tuning.
5. Update the Live Demo badge URL in `README.md` if the Space slug differs from
   `waleedarif1011/ai-story`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/` (`prompts.py`, `inference.py`, `train_lora.py`, `__init__.py`)
- `data/sample_ds_stories.jsonl` (optional for the demo; useful for docs)

Inference downloads `google/flan-t5-base` on first boot (CPU basic hardware is enough).
