"""
Notebook 03 — Sentiment Analysis Pipeline
==========================================
Run as a script:  python notebooks/03_sentiment_pipeline.py
Convert to notebook: jupytext --to notebook notebooks/03_sentiment_pipeline.py

Requires: pip install -r training/requirements.txt jupytext datasets
"""

# %% [markdown]
# # Sentiment Analysis Pipeline — SST-2 Fine-Tuned DistilBERT + VADER Ensemble
# **Dataset**: SST-2 (GLUE benchmark) — 67,349 training sentences, binary sentiment
# **Model**: DistilBERT-base-uncased fine-tuned + VADER ensemble

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 120
sys.path.insert(0, str(Path(".").resolve() / "backend"))

# %% [markdown]
# ## 1. SST-2 Dataset Exploration

# %%
from datasets import load_dataset
sst2 = load_dataset("glue", "sst2")
train_df = pd.DataFrame(sst2["train"])
val_df   = pd.DataFrame(sst2["validation"])

print(f"Train:      {len(train_df):,} samples")
print(f"Validation: {len(val_df):,} samples")
print(f"Label balance (train): {train_df['label'].value_counts().to_dict()}")
train_df.head()

# %% [markdown]
# ## 2. Sentence Length Distribution

# %%
train_df["word_count"] = train_df["sentence"].str.split().str.len()
val_df["word_count"]   = val_df["sentence"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

train_df.groupby("label")["word_count"].plot.hist(
    ax=axes[0], bins=30, alpha=0.7,
    label=["Negative (0)", "Positive (1)"], color=["#f74f4f", "#4faf7f"]
)
axes[0].set_title("Word Count by Sentiment (Train)")
axes[0].set_xlabel("Word Count")
axes[0].legend(["Negative", "Positive"])

# Label distribution
label_counts = train_df["label"].value_counts()
axes[1].bar(["Negative", "Positive"], label_counts.values,
            color=["#f74f4f", "#4faf7f"])
axes[1].set_title("Label Distribution — SST-2 Train Set")
axes[1].set_ylabel("Count")
for i, v in enumerate(label_counts.values):
    axes[1].text(i, v + 100, f"{v:,}\n({v/len(train_df)*100:.1f}%)",
                 ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("notebooks/figures/03_sst2_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Live Inference — DistilBERT + VADER Ensemble

# %%
from app.ml.sentiment import analyze

sample_texts = [
    "I absolutely love this product — it exceeded every expectation!",
    "Terrible experience. Completely broken on arrival, zero stars.",
    "It's okay, not great but not terrible either.",
    "Best purchase I've made this year. Fast shipping, perfect quality.",
    "The battery dies after 2 hours. Very disappointed.",
    "Decent value for the price, does what it says.",
    "Outstanding craftsmanship, I recommend it to everyone.",
    "Returned immediately. Cheap plastic, nothing like the photos.",
]

results = [analyze(t) for t in sample_texts]

# Build summary DataFrame
summary = pd.DataFrame([{
    "text":       t[:55] + "…",
    "label":      r["overall"]["label"],
    "score":      round(r["overall"]["score"], 3),
    "confidence": round(r["overall"]["confidence"], 3),
    "top_emotion": max(r["emotions"], key=r["emotions"].get) if r.get("emotions") else "–",
} for t, r in zip(sample_texts, results)])

print(summary.to_string(index=False))

# %% [markdown]
# ## 4. Sentiment Score Distribution — Real Samples

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Score distribution
pos_scores = [r["overall"]["score"] for r in results if r["overall"]["label"] == "positive"]
neg_scores = [r["overall"]["score"] for r in results if r["overall"]["label"] == "negative"]
axes[0].hist(pos_scores, bins=10, alpha=0.7, color="#4faf7f", label="Positive")
axes[0].hist(neg_scores, bins=10, alpha=0.7, color="#f74f4f", label="Negative")
axes[0].set_title("Sentiment Score Distribution")
axes[0].set_xlabel("Sentiment Score")
axes[0].legend()

# Emotion breakdown
if results[0].get("emotions"):
    emotions_agg = {}
    for r in results:
        for em, val in r.get("emotions", {}).items():
            emotions_agg[em] = emotions_agg.get(em, 0) + abs(val)
    em_series = pd.Series(emotions_agg).sort_values(ascending=True)
    em_series.plot.barh(ax=axes[1], color="#4f8ef7")
    axes[1].set_title("Aggregate Emotion Intensity — Sample Reviews")
    axes[1].set_xlabel("Cumulative Intensity")

plt.tight_layout()
plt.savefig("notebooks/figures/03_sentiment_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. Aspect-Level Sentiment Analysis

# %%
product_reviews = [
    "Great battery life and fast charging, but the screen is too dim.",
    "Build quality is excellent, however the software crashes constantly.",
    "Perfect camera, terrible microphone. Delivery was super fast though.",
    "Sound quality is incredible. Price is a bit high but worth it.",
]

print("=" * 60)
print("  Aspect-Level Sentiment Analysis")
print("=" * 60)
for rev in product_reviews:
    r = analyze(rev)
    print(f"\nReview: {rev[:55]}…")
    print(f"  Overall: {r['overall']['label']} (score={r['overall']['score']:.3f})")
    if r.get("aspects"):
        for asp in r["aspects"][:3]:
            print(f"  Aspect: {asp.get('aspect','?'):<15} → {asp.get('sentiment','?')}")

# %% [markdown]
# ## 6. SST-2 Validation Set Evaluation (using pretrained DistilBERT baseline)

# %%
try:
    from transformers import pipeline
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
    )

    sample_val = val_df.sample(200, random_state=42)
    preds = classifier(sample_val["sentence"].tolist(), batch_size=32)
    label_map = {"POSITIVE": 1, "NEGATIVE": 0}
    pred_labels = [label_map[p["label"]] for p in preds]
    true_labels = sample_val["label"].tolist()

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\nSST-2 Validation Accuracy (200 samples): {acc*100:.2f}%")
    print(classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"]))

except Exception as e:
    print(f"Evaluation skipped: {e}")

# %% [markdown]
# ## 7. Key Findings & Honest Assessment
#
# | Aspect | Detail |
# |---|---|
# | **Base accuracy** | DistilBERT SST-2 pretrained: ~91% on validation |
# | **Fine-tuning gain** | 1–3% improvement from domain-specific fine-tuning |
# | **VADER strength** | Fast, lexicon-based, good for short texts |
# | **DistilBERT strength** | Context-aware, handles negation and sarcasm better |
# | **Ensemble rationale** | VADER catches sentiment keywords; DistilBERT handles syntax |
# | **Limitation** | SST-2 is movie reviews — may need domain adaptation for e-commerce |

print("\n✅ Notebook 03 complete — figures saved to notebooks/figures/")
