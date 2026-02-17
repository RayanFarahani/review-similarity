"""
core.py — Reusable embedding & search logic (no UI dependencies)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── Singleton model (loaded once, reused everywhere) ─────────────────────────
_model: SentenceTransformer | None = None

def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


# ── Data loading ──────────────────────────────────────────────────────────────
def load_reviews(csv_path: str, sample: int | None = None) -> list[str]:
    """Load and clean review texts from CSV."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Review Text"]).reset_index(drop=True)
    texts = df["Review Text"].tolist()
    if sample:
        texts = texts[:sample]
    return texts


# ── Embeddings ────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str], batch_size: int = 64) -> list:
    """Return embeddings as a list of numpy vectors."""
    model = get_model()
    vecs = model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, convert_to_numpy=True)
    return list(vecs)


# ── Categorisation ────────────────────────────────────────────────────────────
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "quality" : ["quality", "material", "fabric", "durable", "cheap",
                 "well-made", "sturdy"],
    "fit"     : ["fit", "fits", "size", "sizing", "tight", "loose", "snug",
                 "small", "large", "true to size", "runs small", "runs large"],
    "style"   : ["style", "stylish", "fashion", "look", "cute", "pretty",
                 "elegant", "trendy", "chic", "design"],
    "comfort" : ["comfort", "comfortable", "soft", "cozy", "cosy", "smooth",
                 "lightweight", "breathable", "itchy", "scratchy"],
}

def categorise_reviews(texts: list[str],
                       keywords: dict | None = None) -> dict[str, list[tuple]]:
    """Return {topic: [(index, review_text), ...]}."""
    kws = keywords or TOPIC_KEYWORDS
    results: dict[str, list] = {t: [] for t in kws}
    for idx, text in enumerate(texts):
        lower = text.lower()
        for topic, kw_list in kws.items():
            if any(kw in lower for kw in kw_list):
                results[topic].append((idx, text))
    return results


# ── Similarity search ─────────────────────────────────────────────────────────
def find_similar_reviews(input_review: str,
                         all_texts: list[str],
                         emb_array: np.ndarray,
                         top_k: int = 3) -> list[str]:
    """Return the top_k most similar reviews to input_review."""
    model = get_model()
    query_vec = model.encode([input_review], convert_to_numpy=True)
    sims = cosine_similarity(query_vec, emb_array)[0]

    if input_review in all_texts:
        sims[all_texts.index(input_review)] = -1.0

    top_idx = sims.argsort()[::-1][:top_k]
    return [all_texts[i] for i in top_idx]
