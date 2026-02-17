"""
app.py — Gradio demo for Hugging Face Spaces
---------------------------------------------
Hugging Face Spaces auto-detects this file and runs it.
"""
import numpy as np
import gradio as gr
from core import load_reviews, embed_texts, find_similar_reviews, categorise_reviews

# ── Load data & build embeddings ONCE at startup ──────────────────────────────
print("Loading reviews …")
review_texts = load_reviews("womens_clothing_e-commerce_reviews.csv", sample=500)
print("Building embeddings (first run downloads ~90 MB model) …")
embeddings   = embed_texts(review_texts)
emb_matrix   = np.array(embeddings)
print(f"Ready — {len(review_texts)} reviews indexed.")


# ── Core function called by Gradio ────────────────────────────────────────────
def search_reviews(input_review: str, top_k: int) -> tuple[str, str]:
    """
    Returns:
        topics_output  : markdown string listing detected topics
        results_output : markdown string listing similar reviews
    """
    if not input_review.strip():
        return "⚠️ Please enter a review.", ""

    # Topics
    cats    = categorise_reviews([input_review])
    matched = [t for t, items in cats.items() if items]
    if not matched:
        matched = ["uncategorised"]

    pills = "  ".join(f"`{t}`" for t in matched)
    topics_md = f"**Detected topics:** {pills}"

    # Similar reviews
    similar = find_similar_reviews(input_review, review_texts, emb_matrix, top_k=top_k)
    results_md = "\n\n".join(
        f"**#{i+1}** — {rev}" for i, rev in enumerate(similar)
    )

    return topics_md, results_md


# ── Example inputs ────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Absolutely wonderful - silky and sexy and comfortable", 3],
    ["The fabric feels cheap and it runs very small", 3],
    ["Love the style but it was not comfortable to wear all day", 3],
    ["Perfect fit, true to size and great quality material", 3],
    ["Cute design but the stitching came undone after one wash", 3],
]


# ── Build the Gradio UI ───────────────────────────────────────────────────────
with gr.Blocks(
    title="👗 Women's Clothing Review Similarity Search",
    theme=gr.themes.Soft(primary_hue="violet"),
) as demo:

    gr.Markdown(
        """
        # 👗 Women's Clothing Review Similarity Search
        Enter any clothing review to instantly find the most similar customer feedback
        and detect which topics it covers (quality, fit, style, comfort).

        > **Model:** `all-MiniLM-L6-v2` · runs 100% locally · no API key needed
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            review_input = gr.Textbox(
                label="Your review",
                placeholder="e.g. Absolutely wonderful - silky and sexy and comfortable",
                lines=3,
            )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(
                label="Number of results",
                minimum=1, maximum=10, step=1, value=3,
            )

    search_btn = gr.Button("🔍 Find Similar Reviews", variant="primary", size="lg")

    with gr.Row():
        topics_output  = gr.Markdown(label="Topics")
    with gr.Row():
        results_output = gr.Markdown(label="Similar Reviews")

    # Wire up button and Enter key
    search_btn.click(
        fn=search_reviews,
        inputs=[review_input, top_k_slider],
        outputs=[topics_output, results_output],
    )
    review_input.submit(
        fn=search_reviews,
        inputs=[review_input, top_k_slider],
        outputs=[topics_output, results_output],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[review_input, top_k_slider],
        outputs=[topics_output, results_output],
        fn=search_reviews,
        cache_examples=True,
        label="Try these examples",
    )

    gr.Markdown(
        """
        ---
        **How it works:** Reviews are encoded into 384-dimensional vectors using
        `sentence-transformers`. Cosine similarity finds the closest matches.
        Topic detection uses keyword matching across quality / fit / style / comfort.
        """
    )

if __name__ == "__main__":
    demo.launch()
