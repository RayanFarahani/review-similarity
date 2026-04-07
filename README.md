---
title: Women's Clothing Review Similarity Search
emoji: 👗
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 👗 Women's Clothing Review Similarity Search

Find similar customer reviews and detect topics (quality, fit, style, comfort)
using sentence embeddings — no API key needed, runs 100% locally.

## What you get

- **Semantic search** over real e-commerce reviews using `sentence-transformers`
- **Topic detection** (quality / fit / style / comfort) via keyword matching
- **Gradio UI** that runs locally or on Hugging Face Spaces

## How it works

1. Reviews are encoded into 384-dim vectors using `all-MiniLM-L6-v2`
2. Cosine similarity finds the closest matches in the dataset
3. Keyword matching categorises reviews by topic

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL printed in your terminal.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI (entry point) |
| `core.py` | Embedding + search logic |
| `requirements.txt` | Python dependencies |
| `womens_clothing_e-commerce_reviews.csv` | Review dataset |

## Notes

- First run will download the embedding model (`all-MiniLM-L6-v2`).
- The app samples **500 reviews** at startup for faster demo performance (configurable in `app.py`).
