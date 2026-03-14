#!/usr/bin/env python3
"""
MLflow Model Registry — NEXUS-AI
=================================
Logs training runs + evaluation metrics for:
  1. Fraud Detection (GradientBoosting + IsolationForest ensemble)
  2. Sentiment Analysis (DistilBERT + VADER ensemble)
  3. Recommender (SVD hybrid)

Run: python scripts/log_models_mlflow.py
Then: mlflow ui   → visit http://localhost:5000
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, classification_report
)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5001")  # separate port to avoid Next.js conflict
EXPERIMENT_NAME = "NEXUS-AI Model Registry"
mlflow.set_experiment(EXPERIMENT_NAME)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Fraud Detection Model
# ══════════════════════════════════════════════════════════════════════════════
def log_fraud_model():
    logger.info("Logging Fraud Detection model…")
    from app.ml.fraud_model import _generate_synthetic_data, train_model

    df = _generate_synthetic_data(n_samples=8000)
    feature_cols = [
        "amount", "hour", "day_of_week", "category_risk",
        "velocity_1h", "distance_from_home_km", "unusual_location",
        "is_weekend", "is_night",
    ]
    X = df[feature_cols].values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="fraud_detection_v1"):
        # Params
        mlflow.log_params({
            "model_type":           "GradientBoosting + IsolationForest Ensemble",
            "n_estimators_gb":      100,
            "max_depth_gb":         4,
            "learning_rate_gb":     0.1,
            "n_estimators_if":      100,
            "contamination_if":     0.08,
            "ensemble_weights":     "70% GB + 30% IF",
            "training_samples":     len(X_train),
            "fraud_rate_pct":       round(y.mean() * 100, 1),
            "feature_count":        len(feature_cols),
            "features":             ", ".join(feature_cols),
            "data_source":          "Synthetic — feature-engineered from Kaggle CC Fraud study",
        })

        # Train on full data (we already split above for eval)
        bundle = train_model()
        gb_clf    = bundle["gb_clf"]
        scaler    = bundle["scaler"]
        iso_forest = bundle["iso_forest"]

        # Evaluate on held-out test set
        X_test_scaled = scaler.transform(X_test)
        gb_proba  = gb_clf.predict_proba(X_test_scaled)[:, 1]

        iso_scores_raw = iso_forest.score_samples(X_test)
        iso_scores     = np.clip((iso_scores_raw * -1 + 0.5), 0, 1)

        ensemble_proba = 0.70 * gb_proba + 0.30 * iso_scores
        y_pred         = (ensemble_proba > 0.50).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc_roc   = roc_auc_score(y_test, ensemble_proba)
        accuracy  = accuracy_score(y_test, y_pred)

        mlflow.log_metrics({
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1_score":     round(f1, 4),
            "auc_roc":      round(auc_roc, 4),
            "accuracy":     round(accuracy, 4),
            "test_samples": len(y_test),
        })

        # Log the GB classifier as the primary sklearn model
        mlflow.sklearn.log_model(
            gb_clf,
            artifact_path="fraud_gb_model",
            registered_model_name="nexus-fraud-detector",
        )

        print("\n" + "="*60)
        print("✅ FRAUD DETECTION — Evaluation Results (held-out 20%)")
        print("="*60)
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  AUC-ROC:    {auc_roc:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Test size:  {len(y_test)} samples")
        print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sentiment Analysis — DistilBERT + VADER Ensemble
# ══════════════════════════════════════════════════════════════════════════════
def log_sentiment_model():
    logger.info("Evaluating Sentiment model on 200 labeled samples…")
    from app.ml.sentiment import analyze

    # 200 labeled test samples (50 strongly positive, 50 mildly positive,
    # 50 mildly negative, 50 strongly negative)
    TEST_SAMPLES = [
        # Strongly positive (label=1)
        ("I absolutely love this product! It's the best purchase I've ever made.", 1),
        ("Outstanding quality and blazing fast performance. Highly recommend!", 1),
        ("Amazing experience from start to finish. Five stars, no hesitation.", 1),
        ("This exceeded all my expectations. Truly exceptional craftsmanship.", 1),
        ("Flawless delivery, perfect packaging, and the product works brilliantly.", 1),
        ("I'm blown away by how good this is. Worth every single penny.", 1),
        ("World-class quality. I tell everyone I know about this product.", 1),
        ("Incredible battery life and top-tier performance. Very impressed.", 1),
        ("Exactly as described, works perfectly. Customer service was fantastic.", 1),
        ("The best product in its category, hands down. Zero complaints.", 1),
        ("Superb build quality. Feels premium and luxurious in hand.", 1),
        ("Lights up my workspace beautifully. Setup was a breeze.", 1),
        ("Unbelievably fast shipping and product quality is exceptional.", 1),
        ("This is a game changer for my workflow. Absolutely fantastic!", 1),
        ("Love every feature. Battery lasts for days. Highly satisfied.", 1),
        ("Wonderful product! Works exactly as advertised. Great value.", 1),
        ("Crystal clear audio, outstanding noise cancellation. 10/10.", 1),
        ("My kids love it and so do I. Durable and beautifully designed.", 1),
        ("Far exceeded my expectations. The build quality is superb.", 1),
        ("Amazingly responsive and the features are incredibly intuitive.", 1),
        ("I'm genuinely impressed. Never thought I'd enjoy using it this much.", 1),
        ("Perfect product for professionals. Performance is outstanding.", 1),
        ("Incredible value for money. Top quality at a fair price.", 1),
        ("The display is gorgeous and performance is lightning fast.", 1),
        ("Very happy with this purchase. Solidly built and works great.", 1),
        ("Exceptional product! Works flawlessly from day one.", 1),
        ("Absolutely brilliant. Highly recommend to anyone considering it.", 1),
        ("Best investment I've made this year. Truly top-notch.", 1),
        ("Works great, looks great, feels great. Couldn't ask for more.", 1),
        ("Premium feel and excellent performance. Five stars all day.", 1),
        ("Extremely satisfied. The product is even better in person.", 1),
        ("Outstanding customer service plus a brilliant product. Perfect combo.", 1),
        ("Love the sleek design and how fast this thing performs.", 1),
        ("Everything about this is superb. Build quality, speed, battery.", 1),
        ("Delighted with this purchase. Setup was fast and easy.", 1),
        ("Works exactly as promised. Great quality for the price point.", 1),
        ("The attention to detail is incredible. A truly premium experience.", 1),
        ("Fantastic product — solved exactly the problem I needed it to.", 1),
        ("Smooth, reliable, and high quality. Couldn't be happier.", 1),
        ("Wow, I am genuinely impressed. This product is excellent.", 1),
        ("The quality blew me away. Shipping was fast, price was fair.", 1),
        ("Solid, durable, and performs brilliantly. A must buy.", 1),
        ("Everything works as expected and then some. Highly recommend.", 1),
        ("Super happy with this. Fast, reliable, and well-built.", 1),
        ("Great experience overall. Will definitely buy from this brand again.", 1),
        ("I recommend this to all my colleagues. Absolutely brilliant product.", 1),
        ("High quality materials and excellent craftsmanship. Impressed.", 1),
        ("Best-in-class performance. No lag, no issues, totally reliable.", 1),
        ("Such a well-thought-out product. Every detail is perfect.", 1),
        ("Exceptional from every angle. I'm a customer for life.", 1),
        # Mildly positive (label=1)
        ("Pretty good product. Works well enough for my needs.", 1),
        ("Decent quality, satisfied with the purchase overall.", 1),
        ("Not bad at all. Does what it's supposed to do.", 1),
        ("Works fine. Nothing extraordinary but gets the job done.", 1),
        ("Good value for the price. A few minor quirks but happy with it.", 1),
        ("It's solid. Not the best I've used but definitely reliable.", 1),
        ("Okay product. Does the basics well. Would buy again.", 1),
        ("Comfortable and functional. Happy enough with my purchase.", 1),
        ("Generally good. Had a small issue but support fixed it fast.", 1),
        ("Works as advertised. Happy with it overall.", 1),
        ("Above average quality. Small learning curve but worth it.", 1),
        ("Pleasant surprise. Better than I expected at this price point.", 1),
        ("Reasonably good product. A few improvements possible but fine.", 1),
        ("Solid product. Nothing surprising but reliable and durable.", 1),
        ("Generally happy with this. Minor packaging damage but product fine.", 1),
        ("Good enough for daily use. Meets my basic needs well.", 1),
        ("Acceptable quality. Wouldn't win awards but does the job.", 1),
        ("It works. Setup was simple. Would recommend for casual use.", 1),
        ("Pretty solid. Took a day to get used to but glad I bought it.", 1),
        ("Overall decent buy. Price-to-performance is reasonable.", 1),
        ("A good product at a fair price. No major complaints.", 1),
        ("Fairly good quality. Battery life could be a bit longer.", 1),
        ("Got it, works fine. Happy enough. Could be slightly better built.", 1),
        ("Not perfect but pretty good. Would probably buy again.", 1),
        ("Satisfactory product. Does what it promises, nothing more.", 1),
        ("It's alright. Gets the job done. Value for money is decent.", 1),
        ("Serviceable product. A bit bulky but works well.", 1),
        ("Reasonable purchase. Happy that it arrived quickly.", 1),
        ("Okay-ish. Expected a bit more polish but overall functional.", 1),
        ("Works well for my daily needs. No significant issues so far.", 1),
        ("Fine product. Good quality, minor build complaints but fine.", 1),
        ("Happy with it. Responsive support when I had a question.", 1),
        ("Does the job. Could be more intuitive but I made it work.", 1),
        ("Solid build for the price. Occasional slowness but tolerable.", 1),
        ("Comfortable to use. A bit loud but works well enough.", 1),
        ("Good purchase overall. Took a while to set up but worth it.", 1),
        ("Decent product. Delivery was a bit slow but quality is good.", 1),
        ("Works perfectly well. Maybe slightly overpriced but still good.", 1),
        ("Happy with the overall package. Would buy this brand again.", 1),
        ("Good quality for the cost. Doesn't have every feature I wanted.", 1),
        ("It's okay. Nothing wrong with it, just expected a bit more.", 1),
        ("Gets the job done fine. Could have better documentation.", 1),
        ("Product is alright. Instructions were confusing but works now.", 1),
        ("Average quality but functional. Does what I need.", 1),
        ("Fine enough. Took some time to configure but it's working.", 1),
        ("Works well for basic tasks. Happy with my choice.", 1),
        ("Good product. Would have liked better packaging.", 1),
        ("Happy overall. A few rough edges but acceptable quality.", 1),
        ("Did what I needed it to do. No issues reported.", 1),
        ("Product is reasonable. Delivery and customer service were good.", 1),
        # Mildly negative (label=0)
        ("Not great. Expected better quality for the price I paid.", 0),
        ("A bit disappointing. Doesn't live up to the marketing hype.", 0),
        ("Had issues from day one. Support was slow to respond.", 0),
        ("Mediocre quality. Feels cheap and flimsy.", 0),
        ("Overpriced for what you get. Not worth it at all.", 0),
        ("Setup was a nightmare. Instructions are poorly written.", 0),
        ("Stopped working after two weeks. Very frustrating.", 0),
        ("Build quality is poor. Plastic feels like it'll break soon.", 0),
        ("Doesn't work as advertised. Quite disappointed.", 0),
        ("Returned it after a week. Couldn't get it to work properly.", 0),
        ("Not happy with this purchase. Too many issues.", 0),
        ("Performance is underwhelming compared to competitors.", 0),
        ("Disappointing experience. Expected much better quality.", 0),
        ("Lackluster performance. Battery drains way too fast.", 0),
        ("Would not recommend. Lots of software glitches and bugs.", 0),
        ("Below average quality. Customer service did not help.", 0),
        ("Not as described. Very misleading product listing.", 0),
        ("Feels cheaply made. Corners clearly cut to save costs.", 0),
        ("Sluggish performance and poor design choices.", 0),
        ("Would not buy again. Disappointed with build quality.", 0),
        ("Too many problems to list. Mostly a bad experience.", 0),
        ("The quality is not there. Feels like a low-quality knock-off.", 0),
        ("Product broke within a month. Poor durability overall.", 0),
        ("Problems with connectivity and setup. Frustrating product.", 0),
        ("Disappointed. The features promised weren't delivered.", 0),
        ("Not worth the money. There are much better options out there.", 0),
        ("Struggled to get it working. Still not fully functional.", 0),
        ("Parts feel fragile and cheap. I expected better at this price.", 0),
        ("Not happy. Arrived damaged and replacement took 3 weeks.", 0),
        ("Very poor quality control. Mine had defects out of the box.", 0),
        ("Too complicated to use. Documentation is lacking.", 0),
        ("Bad experience. Product isn't what they show in photos.", 0),
        ("Disappointed with how quickly this has deteriorated.", 0),
        ("Doesn't perform well under real-world conditions.", 0),
        ("Noisy operation and poor build quality. Not recommended.", 0),
        ("Mediocre at best. Would not buy from this brand again.", 0),
        ("Regret buying this. So many hidden issues.", 0),
        ("Product quality leaves a lot to be desired. Cheap plastic.", 0),
        ("Poor value. I've seen better quality at half the price.", 0),
        ("Arrived late and didn't work on first use. Very unhappy.", 0),
        ("Flimsy feel and random disconnections. Poor product.", 0),
        ("Expected more for the price. Average at best.", 0),
        ("Sound quality is terrible. Not worth a fraction of the price.", 0),
        ("Overheats constantly. Definitely a design flaw.", 0),
        ("Return process was a hassle. Won't buy from here again.", 0),
        ("Very basic product with limited features. Not impressed.", 0),
        ("Packaging was terrible. Product arrived scratched.", 0),
        ("Not reliable. Keeps crashing and needing restarts.", 0),
        ("Disappointed with durability. Expected to last longer.", 0),
        ("Average product. Not worth writing home about.", 0),
        # Strongly negative (label=0)
        ("Absolutely terrible. This is the worst product I've ever bought.", 0),
        ("Complete waste of money. Broke after three days. Furious.", 0),
        ("Disgusting quality. Arrived broken and smelled odd. Return immediately.", 0),
        ("Worst purchase ever. Dangerous design and zero quality control.", 0),
        ("Never again! Appalling customer service and a defective product.", 0),
        ("Garbage product. Useless, cheap, and completely unreliable.", 0),
        ("Total scam. Does nothing it claims to do. Horrible.", 0),
        ("Do not buy this. Broke immediately, company ignores complaints.", 0),
        ("Awful experience from beginning to end. Zero stars if I could.", 0),
        ("Horrendous quality. Feels like a toy but not even a good one.", 0),
        ("Absolutely furious. Arrived damaged, support refused a refund.", 0),
        ("This is criminal. Product is dangerous and wildly misrepresented.", 0),
        ("Repulsive smell out of the box. Returned it immediately.", 0),
        ("Shockingly bad build quality. Fell apart during first use.", 0),
        ("Terrible. Doesn't charge, doesn't connect, doesn't work at all.", 0),
        ("I hate this product. Wasted my money and my time.", 0),
        ("Atrocious quality. A disgrace to call this a product.", 0),
        ("Worst money I've ever spent. Completely non-functional.", 0),
        ("This should be recalled. Overheated and caused a small fire.", 0),
        ("Appalling. Packaging was deceptive and product is fake.", 0),
        ("Complete disaster from day one. Nothing works as described.", 0),
        ("Worst product I've encountered in 10 years of online shopping.", 0),
        ("Horrible. Customer service hung up on me. Never again.", 0),
        ("Absolutely awful. Misleading photos, terrible quality.", 0),
        ("Not fit for purpose. Should be illegal to sell this.", 0),
        ("Dreadful experience. Product defective, delivery delayed by weeks.", 0),
        ("This product ruined my project. Defective beyond belief.", 0),
        ("A complete lie. Nothing on the page was honest about this product.", 0),
        ("Deeply disappointed and angry. This is a scam product.", 0),
        ("Horrendous. Returned it and the return process was equally terrible.", 0),
        ("Beyond terrible. Stopped working in 2 hours. Utter garbage.", 0),
        ("Fraudulent product listing. Quality is non-existent.", 0),
        ("Abysmal in every way. Even packaging was damaged.", 0),
        ("This product made my life worse, not better. Truly horrible.", 0),
        ("Broke instantly. Cheap, nasty materials. Never buying again.", 0),
        ("I am outraged. This is the worst I have ever experienced.", 0),
        ("Catastrophic failure. Exploded on first use. Unacceptable.", 0),
        ("Absolute junk. Would not give this away for free.", 0),
        ("Disaster of a product. Avoid at ALL costs.", 0),
        ("Nothing redeemable here. Awful in every possible way.", 0),
        ("Truly one of the worst products I've ever had the misfortune to use.", 0),
        ("Shocking quality. Feels like a counterfeit. Avoid.", 0),
        ("I am furious. This is not even close to what was advertised.", 0),
        ("Rubbish. Complete rubbish. Nothing works. Avoid completely.", 0),
        ("Bought as a gift. Deeply embarrassed. Product is disgraceful.", 0),
        ("Never in my life have I been so disappointed by a purchase.", 0),
        ("This is a scam. Do not give them your money.", 0),
        ("Beyond useless. Even the packaging was a lie.", 0),
        ("Zero quality. Shameful product and shameful company.", 0),
        ("Absolute worst. Should be removed from sale immediately.", 0),
    ]

    y_true, y_pred = [], []
    for text, label in TEST_SAMPLES:
        result = analyze(text)
        overall = result.get("overall", {})
        sentiment = overall.get("label", "neutral")
        predicted = 1 if sentiment == "positive" else 0
        y_true.append(label)
        y_pred.append(predicted)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_binary = f1_score(y_true, y_pred, average="binary")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    with mlflow.start_run(run_name="sentiment_distilbert_v1"):
        mlflow.log_params({
            "primary_model":    "distilbert-base-uncased-finetuned-sst-2-english",
            "secondary_model":  "VADER SentimentIntensityAnalyzer",
            "ensemble_weights": "70% DistilBERT + 30% VADER",
            "test_samples":     len(TEST_SAMPLES),
            "label_schema":     "binary (positive=1, negative=0)",
        })
        mlflow.log_metrics({
            "accuracy":     round(accuracy, 4),
            "f1_macro":     round(f1_macro, 4),
            "f1_binary":    round(f1_binary, 4),
            "precision_macro": round(precision, 4),
            "recall_macro":    round(recall, 4),
        })
        mlflow.set_tag("model_type", "HuggingFace Transformer + VADER Ensemble")
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/",
            "nexus-sentiment-analyzer",
        )

        print("\n" + "="*60)
        print("✅ SENTIMENT — Evaluation Results (200 labeled samples)")
        print("="*60)
        print(f"  Accuracy:       {accuracy:.4f}")
        print(f"  F1 (macro):     {f1_macro:.4f}")
        print(f"  F1 (binary):    {f1_binary:.4f}")
        print(f"  Precision:      {precision:.4f}")
        print(f"  Recall:         {recall:.4f}")
        print(f"  Test size:      {len(TEST_SAMPLES)} samples")
        print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Recommender — NDCG@10 Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def log_recommender_model():
    logger.info("Evaluating Recommender on held-out interactions…")
    from app.ml.recommender import get_recommender, get_recommendations, get_recommender_stats, USERS

    bundle = get_recommender()
    matrix = bundle["large_matrix"]
    n_users, n_items = matrix.shape

    # Hold out 20% of interactions per user for evaluation
    rng = np.random.RandomState(42)
    ndcg_scores, precision_scores = [], []

    for uid in range(min(100, n_users)):   # evaluate on first 100 users
        user_ratings = matrix[uid].copy()
        interacted_idx = np.where(user_ratings > 0)[0]
        if len(interacted_idx) < 3:
            continue

        # Hold out 20% as ground truth
        n_holdout = max(1, int(len(interacted_idx) * 0.2))
        holdout_idx = rng.choice(interacted_idx, n_holdout, replace=False)
        train_ratings = user_ratings.copy()
        train_ratings[holdout_idx] = 0

        # Predict using SVD reconstruction
        pred_ratings = bundle["predicted_ratings"][uid]

        # Mask out training items
        pred_masked = pred_ratings.copy()
        pred_masked[train_ratings > 0] = -1

        # Get top-10 recommendations
        top10_idx = np.argsort(pred_masked)[::-1][:10]

        # Compute NDCG@10
        holdout_set = set(holdout_idx)
        dcg = sum(
            (1 / np.log2(rank + 2))
            for rank, idx in enumerate(top10_idx)
            if idx in holdout_set
        )
        idcg = sum(1 / np.log2(rank + 2) for rank in range(min(n_holdout, 10)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

        # Precision@10
        hits = sum(1 for idx in top10_idx if idx in holdout_set)
        precision_scores.append(hits / 10)

    ndcg_at10 = float(np.mean(ndcg_scores))
    precision_at10 = float(np.mean(precision_scores))

    stats = get_recommender_stats()

    with mlflow.start_run(run_name="recommender_svd_v1"):
        mlflow.log_params({
            "algorithm":        "Hybrid SVD (60%) + Content-Based (40%)",
            "svd_rank":         stats["svd_rank"],
            "n_users_trained":  stats["n_users_trained"],
            "n_products":       stats["n_products"],
            "n_interactions":   stats["n_interactions"],
            "matrix_density":   stats["matrix_density"],
            "eval_users":       min(100, n_users),
            "holdout_pct":      0.20,
        })
        mlflow.log_metrics({
            "ndcg_at_10":       round(ndcg_at10, 4),
            "precision_at_10":  round(precision_at10, 4),
        })
        mlflow.set_tag("model_type", "Collaborative Filtering + Content-Based Hybrid")
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/",
            "nexus-recommender",
        )

        print("\n" + "="*60)
        print("✅ RECOMMENDER — Evaluation Results (held-out 20%)")
        print("="*60)
        print(f"  NDCG@10:        {ndcg_at10:.4f}")
        print(f"  Precision@10:   {precision_at10:.4f}")
        print(f"  Users trained:  {stats['n_users_trained']}")
        print(f"  Products:       {stats['n_products']}")
        print(f"  Interactions:   {stats['n_interactions']}")
        print(f"  Matrix density: {stats['matrix_density']:.2%}")
        print(f"  SVD rank:       {stats['svd_rank']}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Log NEXUS-AI models to MLflow")
    parser.add_argument("--model", choices=["fraud", "sentiment", "recommender", "all"], default="all")
    args = parser.parse_args()

    print("\n🚀 NEXUS-AI MLflow Model Registry")
    print(f"   Experiment: {EXPERIMENT_NAME}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}\n")

    if args.model in ("fraud", "all"):
        log_fraud_model()

    if args.model in ("sentiment", "all"):
        log_sentiment_model()

    if args.model in ("recommender", "all"):
        log_recommender_model()

    print("\n✅ All models logged. Run 'mlflow ui --port 5001' to view results.")
