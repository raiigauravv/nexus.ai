"""
Upgraded Sentiment Analysis Pipeline
Primary: DistilBERT (fine-tuned on SST-2) — 70% weight
Secondary: VADER — 30% weight (adds score granularity)
Enrichment: TextBlob (subjectivity), aspect extraction, emotion approximation
"""
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ── Lazy-loaded model singletons ───────────────────────────────────────────────
_distilbert = None
_vader = None
_textblob_imported = False

def _get_distilbert():
    global _distilbert
    if _distilbert is None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading DistilBERT sentiment model…")
            _distilbert = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
            logger.info("DistilBERT loaded.")
        except Exception as e:
            logger.warning(f"DistilBERT unavailable ({e}), using VADER fallback.")
            _distilbert = None
    return _distilbert

def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


# ── Aspect & Emotion data ──────────────────────────────────────────────────────
ASPECTS = {
    "Quality":          ["quality", "durable", "build", "sturdy", "cheap", "flimsy", "solid", "well-made"],
    "Price":            ["price", "expensive", "cheap", "value", "worth", "cost", "affordable", "overpriced"],
    "Performance":      ["fast", "slow", "performance", "speed", "powerful", "laggy", "smooth", "efficient"],
    "Customer Service": ["support", "service", "helpful", "response", "staff", "rude", "friendly", "ignored"],
    "Delivery":         ["shipping", "delivery", "arrived", "damaged", "late", "fast", "early", "package"],
    "Ease of Use":      ["easy", "difficult", "intuitive", "complicated", "setup", "instructions", "simple"],
    "Design":           ["design", "beautiful", "ugly", "aesthetic", "look", "appearance", "style", "color"],
    "Battery":          ["battery", "charge", "last", "drain", "life", "hours", "power"],
}

EMOTIONS = {
    "joy":          ["love", "amazing", "excellent", "happy", "great", "wonderful", "fantastic", "best", "perfect"],
    "trust":        ["reliable", "dependable", "trust", "consistent", "honest", "solid", "safe", "quality"],
    "anticipation": ["look forward", "excited", "hope", "expecting", "can't wait", "eager"],
    "sadness":      ["sad", "disappoint", "unfortunate", "regret", "miss", "unhappy", "sorry"],
    "anger":        ["angry", "furious", "terrible", "horrible", "worst", "awful", "hate", "never again"],
    "fear":         ["afraid", "worry", "concern", "anxious", "nervous", "risk", "danger", "unsafe"],
    "surprise":     ["surprised", "unexpected", "shocked", "amazed", "astonished", "wow"],
    "disgust":      ["disgusting", "gross", "nasty", "repulsive", "unacceptable", "appalling"],
}

NEGATIONS = {"not", "no", "never", "nothing", "neither", "nor", "nobody", "nowhere"}
AMPLIFIERS = {"very", "extremely", "really", "absolutely", "totally", "completely", "deeply", "incredibly"}
DIMINISHERS = {"somewhat", "kind of", "a bit", "slightly", "barely", "rarely", "hardly"}


# ── Core analysis function ─────────────────────────────────────────────────────
def analyze(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"error": "Empty text"}

    words = text.lower().split()
    word_set = set(words)

    # ── 1. Primary: DistilBERT transformer ─────────────────────────────────────
    distilbert_score = 0.0
    model_used = "VADER (fallback)"
    transformer_label = "NEUTRAL"
    transformer_confidence = 0.0

    distilbert = _get_distilbert()
    if distilbert:
        try:
            result = distilbert(text[:512])[0]
            raw_score = result["score"]
            transformer_confidence = raw_score
            if result["label"] == "POSITIVE":
                distilbert_score = raw_score
                transformer_label = "POSITIVE"
            else:
                distilbert_score = -raw_score
                transformer_label = "NEGATIVE"
            model_used = "DistilBERT SST-2"
        except Exception as e:
            logger.warning(f"DistilBERT inference failed: {e}")

    # ── 2. Secondary: VADER (adds compound granularity) ────────────────────────
    vader = _get_vader()
    vader_scores = vader.polarity_scores(text)
    vader_compound = vader_scores["compound"]

    # ── 3. Ensemble score ──────────────────────────────────────────────────────
    if distilbert and model_used == "DistilBERT SST-2":
        # 70% DistilBERT + 30% VADER for combined strength
        ensemble_score = 0.70 * distilbert_score + 0.30 * vader_compound
    else:
        # Pure VADER fallback
        from textblob import TextBlob
        blob = TextBlob(text)
        ensemble_score = 0.65 * vader_compound + 0.35 * blob.sentiment.polarity

    # ── 4. TextBlob for subjectivity ───────────────────────────────────────────
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        subjectivity = blob.sentiment.subjectivity
        textblob_polarity = blob.sentiment.polarity
    except Exception:
        subjectivity = 0.5
        textblob_polarity = 0.0

    # ── 5. Negation/amplifier adjustments ─────────────────────────────────────
    neg_count = sum(1 for w in words if w in NEGATIONS)
    amp_count = sum(1 for w in words if w in AMPLIFIERS)
    if neg_count > 0:
        ensemble_score *= max(0.3, 1 - 0.25 * neg_count)
    if amp_count > 0:
        ensemble_score = min(1.0, ensemble_score * (1 + 0.1 * amp_count)) if ensemble_score > 0 else \
                         max(-1.0, ensemble_score * (1 + 0.1 * amp_count))

    ensemble_score = max(-1.0, min(1.0, ensemble_score))

    # ── 6. Overall label ──────────────────────────────────────────────────────
    if ensemble_score > 0.15:
        overall_label, emoji = "positive", "😊"
    elif ensemble_score < -0.15:
        overall_label, emoji = "negative", "😞"
    else:
        overall_label, emoji = "neutral", "😐"

    confidence = min(1.0, abs(ensemble_score) + 0.2)

    # ── 7. Emotion approximation ───────────────────────────────────────────────
    emotions = {}
    for emotion, emo_words in EMOTIONS.items():
        hits = sum(1 for w in emo_words if w in word_set or any(w in ww for ww in words))
        emotions[emotion] = round(min(1.0, hits * 0.35 + 0.05), 3)
    # Boost the dominant emotion based on overall sentiment
    if overall_label == "positive":
        emotions["joy"] = min(1.0, emotions["joy"] + 0.3)
        emotions["trust"] = min(1.0, emotions["trust"] + 0.1)
    elif overall_label == "negative":
        emotions["anger"] = min(1.0, emotions["anger"] + 0.2)
        emotions["sadness"] = min(1.0, emotions["sadness"] + 0.15)

    # ── 8. Aspect extraction ───────────────────────────────────────────────────
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 5]
    aspect_results = []
    for aspect, kws in ASPECTS.items():
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in kws):
                sent_words = sent_lower.split()
                neg = any(n in sent_words for n in NEGATIONS)
                v = vader.polarity_scores(sent)["compound"]
                if neg and v > 0:
                    v = -v * 0.8
                label = "positive" if v > 0.05 else "negative" if v < -0.05 else "neutral"
                aspect_results.append({
                    "aspect": aspect,
                    "sentiment": label,
                    "score": round(v, 3),
                    "excerpt": sent[:100],
                })
                break

    # ── 9. Readability (Flesch-Kincaid approximation) ─────────────────────────
    n_sentences = max(1, len(sentences))
    n_words = max(1, len(words))
    avg_word_len = sum(len(w) for w in words) / n_words
    fk_score = max(0, 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (avg_word_len / 4))

    return {
        "overall": {
            "label": overall_label,
            "emoji": emoji,
            "score": round(ensemble_score, 4),
            "confidence": round(confidence, 4),
        },
        "model_info": {
            "primary_model": model_used,
            "transformer_label": transformer_label,
            "transformer_confidence": round(transformer_confidence, 4),
            "vader_compound": round(vader_compound, 4),
            "textblob_polarity": round(textblob_polarity, 4),
            "ensemble_weights": "70% DistilBERT + 30% VADER" if "DistilBERT" in model_used else "65% VADER + 35% TextBlob",
        },
        "emotions": emotions,
        "aspects": aspect_results,
        "metadata": {
            "word_count": n_words,
            "sentence_count": n_sentences,
            "subjectivity": round(subjectivity, 4),
            "readability": round(fk_score, 1),
            "negation_count": neg_count,
            "amplifier_count": amp_count,
        },
    }


# ── Backwards-compat exports ───────────────────────────────────────────────────

def get_vader():
    """Public alias for the VADER analyzer singleton (used by sentiment endpoint)."""
    return _get_vader()


SAMPLE_REVIEWS = [
    {"id": "R001", "product": "ProBook Ultra Laptop",     "rating": 5, "text": "Excellent laptop, handles everything I throw at it. Battery could be better but overall great. Worth every penny."},
    {"id": "R002", "product": "SoundWave Pro Headphones", "rating": 5, "text": "Best headphones I've ever owned. Crystal clear audio and outstanding noise cancellation."},
    {"id": "R003", "product": "UltraView 4K Monitor",     "rating": 4, "text": "Amazing 4K clarity, colors are vibrant. Perfect for my home office setup. A bit pricey but worth it."},
    {"id": "R004", "product": "SmartCharge Wireless Pad", "rating": 2, "text": "Works as expected, nothing special. Charges slowly and the coil placement is finicky. Stopped working after 3 months."},
    {"id": "R005", "product": "StreamCam 4K Pro",         "rating": 5, "text": "Incredible webcam for streaming. 4K quality is noticeable. Autofocus is fast. Software drivers crash occasionally."},
    {"id": "R006", "product": "KeyMaster Pro RGB",        "rating": 4, "text": "Satisfying keystrokes and stunning RGB lighting. A bit loud for open offices but build quality is excellent."},
    {"id": "R007", "product": "Deep Learning Book",       "rating": 4, "text": "Comprehensive and well-written. A must-read for ML practitioners, though dense with theory."},
    {"id": "R008", "product": "Atomic Habits",            "rating": 5, "text": "Changed how I think and act. Engaging writing style and transformative perspectives."},
]
