# Hugging Face Space deploy

The Gradio **Data-to-Story Generator** is Space-ready and **boots without private
Google Drive files**. Default inference uses public `google/flan-t5-base`.

A live Space was **not** created from this agent run because no Hugging Face
write token (`HF_TOKEN`) was available in the environment. Do not advertise a
Live Demo URL until the Space exists.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI with `HF_TOKEN` set):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token, or export HF_TOKEN=…
huggingface-cli repo create ai-story --type space --space_sdk gradio -y
# From a clean checkout of this repo (or after merging to main):
git remote add space https://huggingface.co/spaces/waleedarif1011/ai-story
git push space main
```

3. Confirm the Space builds and the four presets behave as expected:
   - Churn model results
   - Revenue anomaly
   - A/B checkout test
   - Quarterly forecast
4. **CPU expectations:** first build/boot downloads ~900MB public Flan-T5-base
   weights; after load, generation is typically **5–20s per story** on Space
   CPU basic hardware.
5. Optional: set Space secret `AI_STORY_ADAPTER` to a **public** Hub LoRA repo ID
   after fine-tuning. Leave unset to keep the Drive-free base-model demo.
6. Once the Space is live, restore the Live Demo badge in `README.md` pointing at
   `https://huggingface.co/spaces/waleedarif1011/ai-story` (or your actual slug).

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/` (`prompts.py`, `inference.py`, `train_lora.py`, `__init__.py`)
- `data/sample_ds_stories.jsonl` (optional for the demo; useful for docs)
- `assets/` (optional; README screenshot)

Inference downloads `google/flan-t5-base` on first boot (CPU basic hardware is enough).
No Google Drive mount, private dataset, or private adapter is required to start.
