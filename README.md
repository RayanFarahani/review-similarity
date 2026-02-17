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

## How it works

1. Reviews are encoded into 384-dim vectors using `all-MiniLM-L6-v2`
2. Cosine similarity finds the closest matches in the dataset
3. Keyword matching categorises reviews by topic

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI (entry point) |
| `core.py` | Embedding + search logic |
| `requirements.txt` | Python dependencies |
| `womens_clothing_e-commerce_reviews.csv` | Review dataset |
