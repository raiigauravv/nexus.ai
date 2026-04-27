"""
Cross-Module Intelligence Layer
Connects the 5 NEXUS-AI ML modules so they inform each other:

  1. Sentiment → Recommendations:
     Each product has synthetic reviews. DistilBERT scores them.
     Low sentiment health → reduced recommendation score.

  2. Fraud → Recommendations:
     HIGH fraud risk → de-rank + flag luxury products (price > $300).

  3. Agent intelligence:
     Combined reasoning that pulls from both modules in one call.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Synthetic reviews per product ─────────────────────────────────────────────
# Each entry: list of (review_text, is_recent) so we can weight recent reviews more
PRODUCT_REVIEWS: Dict[str, List[str]] = {
    "P001": ["Excellent laptop, handles everything I throw at it.", "Battery could be better but overall great.", "Worth every penny for the performance.", "Runs hot after extended use — cooling system needs work.", "Screen is gorgeous, keyboard feels premium."],
    "P002": ["Best headphones I've ever owned. Crystal clear audio.", "Noise cancellation is outstanding.", "Earcups get uncomfortable after 3 hours.", "Worth the price, sound quality is superb.", "Only downside is the app is a bit buggy."],
    "P003": ["Amazing 4K clarity, colors are vibrant.", "Perfect for my home office setup.", "A bit pricey but the display quality justifies it.", "Had a dead pixel — support was helpful.", "Excellent color accuracy for design work."],
    "P004": ["Works as expected, nothing special.", "Charges a bit slowly compared to competitors.", "Convenient but the coil placement is finicky.", "It's reliable and does the job.", "Stopped working after 3 months — cheap quality."],
    "P005": ["Incredible webcam for streaming. Looks professional.", "4K quality is noticeable in video calls.", "Autofocus is fast and accurate.", "Software drivers crash occasionally.", "Setup was plug-and-play, loved it."],
    "P006": ["Satisfying keystrokes, best mechanical keyboard I've used.", "RGB lighting is stunning.", "A bit loud — not for open offices.", "Build quality is excellent, very sturdy.", "Function keys are too small."],
    "P007": ["Comprehensive and well-written. A must-read for ML practitioners.", "Dense with theory — not for beginners.", "Best deep learning book available.", "Slightly outdated in some chapters.", "Worth every page."],
    "P008": ["Changed how I think about money. Highly recommend.", "Engaging writing style, easy to read.", "Some sections feel repetitive.", "A classic personal finance book.", "Life-changing perspectives on wealth."],
    "P009": ["Essential reading for any backend engineer.", "Extremely practical and well-organized.", "A bit dry but incredibly valuable.", "My go-to reference for system design.", "Covers everything from databases to distributed systems."],
    "P010": ["Simple yet powerful framework for building better habits.", "Easy to read and implement.", "Some advice feels common sense.", "Transformative book — already seeing results.", "A bit repetitive by the end."],
    "P011": ["Great fit and comfortable for workouts.", "Durable material, survived many washes.", "A bit pricey for shorts.", "Perfect for running and gym sessions.", "Sizing runs a bit small — order up."],
    "P012": ["Keeps me dry in heavy rain. Excellent jacket.", "Stylish and functional.", "Zipper broke after 6 months.", "Great for commuting in wet weather.", "Pockets are well-placed and spacious."],
    "P013": ["Perfect for long runs — very comfortable.", "Great arch support and cushioning.", "Durability is a concern after heavy use.", "Helped improve my running pace.", "Stylish design that goes beyond the gym."],
    "P014": ["Makes the best coffee I've had at home.", "Smart features work seamlessly.", "Takes time to learn all the settings.", "Worth the investment for coffee lovers.", "Cleans itself — a huge plus."],
    "P015": ["Noticeable improvement in air quality.", "Runs quietly even on high.", "Filter replacements are expensive.", "App integration works perfectly.", "Great for allergy sufferers."],
    "P016": ["Powerful blender, handles everything.", "Easy to clean — just add water and blend.", "A bit loud but expected for the power.", "Perfect for smoothies and soups.", "Motor feels incredibly durable."],
    "P017": ["Tracks everything I need — sleep, steps, HR.", "Battery lasts over a week.", "Display is a bit small.", "GPS accuracy is impressive.", "Best fitness tracker on the market."],
    "P018": ["Perfect thickness, great grip during yoga.", "Non-slip surface is excellent.", "Carries my weight well in all poses.", "Slightly expensive for a yoga mat.", "Easy to clean and roll up."],
    "P019": ["Great variety of resistance levels.", "Durable bands that don't snap.", "Instructions could be clearer.", "Perfect for home workouts.", "Compact and easy to store."],
    "P020": ["Incredibly precise — huge improvement to my gaming.", "RGB lighting looks great.", "Cord is a bit stiff.", "Clicks are satisfying and responsive.", "Perfect weight for extended gaming sessions."],
    "P021": ["Best gaming chair I've owned. Super comfortable.", "Assembly took 2 hours but worth it.", "Lumbar support is excellent.", "Runs a bit warm in summer.", "Steel frame feels very solid."],
    "P022": ["Haptic feedback is a game changer.", "Works across all my platforms seamlessly.", "Grip feels natural after a few hours.", "Battery life is exceptional.", "Buttons have great tactile response."],
    "P023": ["Noticeably improved my skin tone in 2 weeks.", "Lightweight and absorbs quickly.", "A bit pricey but effective.", "Love the scent — not overwhelming.", "Works great under makeup."],
    "P024": ["My skin has never felt better.", "Non-greasy formula is perfect.", "Packaging is elegant.", "Lasts all day without reapplying.", "Didn't cause any breakouts — great for sensitive skin."],
    "P025": ["Excellent video quality even at night.", "Easy installation on my windshield.", "App is intuitive and well-designed.", "Caught a parking incident — saved me money.", "GPS tracking feature works well."],
}


# ── Sentiment health cache ─────────────────────────────────────────────────────
_product_sentiment_cache: Dict[str, float] = {}


def get_product_sentiment_health(product_id: str) -> float:
    """
    Returns a sentiment health score [0, 1] for a product based on its reviews.
    1.0 = universally loved, 0.0 = universally hated.
    Results are cached since they don't change at runtime.
    """
    if product_id in _product_sentiment_cache:
        return _product_sentiment_cache[product_id]

    reviews = PRODUCT_REVIEWS.get(product_id, [])
    if not reviews:
        _product_sentiment_cache[product_id] = 0.5
        return 0.5

    try:
        from app.ml.sentiment import analyze
        scores = []
        for review in reviews:
            result = analyze(review)
            raw_score = result["overall"]["score"]  # [-1, 1]
            scores.append(raw_score)

        if not scores:
            health = 0.5
        else:
            avg_score = sum(scores) / len(scores)
            # Normalize from [-1, 1] to [0, 1]
            health = round((avg_score + 1) / 2, 4)

    except Exception as e:
        logger.warning(f"Sentiment health failed for {product_id}: {e}")
        health = 0.5

    _product_sentiment_cache[product_id] = health
    return health


def get_all_product_sentiment_health() -> Dict[str, float]:
    """Pre-compute sentiment health for all products."""
    from app.ml.recommender import PRODUCTS
    results = {}
    for p in PRODUCTS:
        results[p["id"]] = get_product_sentiment_health(p["id"])
    return results


def get_sentiment_adjusted_recommendations(
    user_id: str,
    top_n: int = 6,
    fraud_risk: Optional[str] = None,  # "LOW" | "MEDIUM" | "HIGH"
) -> List[Dict]:
    """
    Hybrid recommendations with cross-module adjustment:
    - Blends sentiment health of each product into the ML score
    - Applies fraud risk de-ranking for expensive items
    """
    from app.ml.recommender import get_recommendations, PRODUCTS

    base_recs = get_recommendations(user_id, top_n=min(top_n + 4, 12))

    prod_id_map = {p["id"]: p for p in PRODUCTS}
    adjusted = []

    for rec in base_recs:
        prod_id = rec["id"]
        base_score = rec.get("score", 0.0)
        price = rec.get("price", 0)

        # ── Sentiment adjustment (20% weight) ──────────────────────────────────
        sentiment_health = get_product_sentiment_health(prod_id)
        # Sentiment health 0.5 = neutral (no change), >0.5 boosts, <0.5 penalizes
        sentiment_delta = (sentiment_health - 0.5) * 0.4  # max ±0.2 impact

        # ── Fraud risk adjustment ───────────────────────────────────────────────
        fraud_penalty = 0.0
        fraud_flag = None
        if fraud_risk == "HIGH" and price > 300:
            fraud_penalty = -0.30
            fraud_flag = "⚠️ High-value item — flagged due to account risk signals"
        elif fraud_risk == "HIGH" and price > 100:
            fraud_penalty = -0.10
            fraud_flag = "⚠️ Caution: elevated risk account"
        elif fraud_risk == "MEDIUM" and price > 500:
            fraud_penalty = -0.10
            fraud_flag = "ℹ️ Premium item — verify account status"

        # ── Final adjusted score ───────────────────────────────────────────────
        adjusted_score = round(
            max(0.0, min(1.0, base_score + sentiment_delta + fraud_penalty)), 4
        )

        # ── Sentiment health label ─────────────────────────────────────────────
        if sentiment_health >= 0.70:
            sh_label = "🟢 Loved"
        elif sentiment_health >= 0.50:
            sh_label = "🟡 Mixed"
        else:
            sh_label = "🔴 Poor reviews"

        enriched = dict(rec)
        enriched["recommendation_score"] = adjusted_score
        enriched["base_ml_score"] = base_score
        enriched["sentiment_health"] = round(sentiment_health, 3)
        enriched["sentiment_health_label"] = sh_label
        enriched["sentiment_delta"] = round(sentiment_delta, 4)
        enriched["fraud_risk"] = fraud_risk or "NONE"
        enriched["fraud_flag"] = fraud_flag
        adjusted.append(enriched)

    # Re-sort by adjusted score
    adjusted.sort(key=lambda x: x["recommendation_score"], reverse=True)
    return adjusted[:top_n]


def explain_complaints_for_category(category: str) -> str:
    """
    Agent tool: given a product category, analyze review sentiment
    and explain which products have the most complaints.
    Returns a structured text summary.
    """
    from app.ml.recommender import PRODUCTS
    from app.ml.sentiment import analyze

    category_products = [p for p in PRODUCTS if p["category"].lower() == category.lower()]
    if not category_products:
        # Try partial match
        category_products = [
            p for p in PRODUCTS
            if category.lower() in p["category"].lower() or category.lower() in p["name"].lower()
        ]

    if not category_products:
        return f"No products found in category '{category}'."

    summary_lines = [f"📊 Review Sentiment Analysis — {category}\n"]
    all_insights = []

    for prod in category_products[:6]:
        reviews = PRODUCT_REVIEWS.get(prod["id"], [])
        if not reviews:
            continue

        scores = []
        negative_reviews = []
        for review in reviews:
            result = analyze(review)
            s = result["overall"]["score"]
            scores.append(s)
            if s < -0.1:
                negative_reviews.append(review)

        avg = sum(scores) / len(scores) if scores else 0.0
        health = round((avg + 1) / 2 * 100, 1)
        neg_pct = round(len(negative_reviews) / max(1, len(reviews)) * 100)

        status = "✅ Strong" if health > 70 else "⚠️ Mixed" if health > 45 else "🔴 Poor"
        all_insights.append({
            "name": prod["name"],
            "health": health,
            "neg_pct": neg_pct,
            "negative_reviews": negative_reviews,
            "status": status,
        })

    all_insights.sort(key=lambda x: x["health"])

    for ins in all_insights:
        summary_lines.append(
            f"  {ins['status']} {ins['name']}: {ins['health']}% positive "
            f"({ins['neg_pct']}% negative reviews)"
        )
        if ins["negative_reviews"]:
            top_complaint = ins["negative_reviews"][0]
            summary_lines.append(f'    Top complaint: "{top_complaint[:80]}…"' if len(top_complaint) > 80 else f'    Top complaint: "{top_complaint}"')

    if all_insights and all_insights[0]["health"] < 50:
        worst = all_insights[0]
        summary_lines.append(
            f"\n⚡ Recommendation: Consider removing '{worst['name']}' from recommendations "
            f"— {worst['neg_pct']}% negative review rate."
        )

    return "\n".join(summary_lines)
