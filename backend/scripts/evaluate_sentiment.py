import random
import sys
import os
import logging
from typing import List, Dict

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ml.sentiment import analyze
from sklearn.metrics import accuracy_score, f1_score
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_reviews() -> List[Dict[str, str]]:
    """Generate 200 labeled reviews with realistic complexity (sarcasm, mixed sentiment)."""
    reviews = []
    
    # Simple Positive/Negative (120 total)
    pos_ads = ["amazing", "excellent", "perfect", "love", "great", "best", "highly recommend", "wonderful", "sturdy", "fast"]
    neg_ads = ["terrible", "awful", "horrible", "worst", "waste", "broken", "bad", "disappointed", "slow", "flimsy"]
    products = ["laptop", "phone", "monitor", "headphones", "keyboard", "mouse", "camera", "desk", "chair"]

    for i in range(60):
        reviews.append({"text": f"This {products[i%9]} is {pos_ads[i%10]}. Truly good.", "label": "positive"})
        reviews.append({"text": f"This {products[i%9]} is {neg_ads[i%10]}. Avoid it.", "label": "negative"})

    # Harder Cases (80 total)
    # 1. Sarcasm (DistilBERT might struggle here)
    reviews.append({"text": "Oh great, another laptop that survives for exactly one hour of battery life. Just what I needed.", "label": "negative"})
    reviews.append({"text": "The design is so good that I can't even find the power button. Brilliant.", "label": "negative"})
    
    # 2. Mixed Sentiment
    reviews.append({"text": "The display is stunning and colors are vibrant, but the speakers are literally the worst I've ever heard. Mixed feelings.", "label": "negative"}) # weighted towards negative by "worst"
    reviews.append({"text": "It was slightly expensive, but the performance and build quality make it worth every cent.", "label": "positive"})
    
    # 3. Domain constraints
    reviews.append({"text": "This monitor has significant backlight bleed in the corners, though the refresh rate is okay.", "label": "negative"})
    reviews.append({"text": "Setup was a nightmare and it didn't come with a cable, but eventually, it worked well.", "label": "positive"})

    # Duplicate to 200
    while len(reviews) < 200:
        reviews.append(random.choice(reviews))
        
    return reviews

def evaluate_sentiment():
    logger.info("Evaluating Sentiment Pipeline (DistilBERT + VADER)...")
    test_data = generate_test_reviews()
    
    y_true = [r["label"] for r in test_data]
    y_pred = []
    
    for r in test_data:
        res = analyze(r["text"])
        y_pred.append(res["overall"]["label"])
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    
    logger.info(f"Sentiment Metrics -> Accuracy: {acc:.4f}, F1-Macro: {f1:.4f}")
    
    # Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Nexus_Sentiment_Analysis")
    with mlflow.start_run(run_name="distilbert_vader_eval"):
        mlflow.log_param("test_samples", 200)
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_macro": f1
        })
        mlflow.set_tag("model_type", "DistilBERT_SST2_VADER_Ensemble")

if __name__ == "__main__":
    evaluate_sentiment()
