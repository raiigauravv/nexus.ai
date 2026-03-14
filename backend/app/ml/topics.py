import logging
from typing import List, Dict, Any
import pandas as pd
from sqlalchemy.orm import Session
from app.models import Review, Product

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_topic_model = None

def _get_topic_model():
    global _topic_model
    if _topic_model is None:
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            logger.info("Initializing BERTopic model...")
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            _topic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=5,
                verbose=False
            )
            logger.info("BERTopic initialized.")
        except Exception as e:
            logger.warning(f"BERTopic initialization failed: {e}. Falling back to simple keyword matching.")
            _topic_model = "FALLBACK"
    return _topic_model

def get_product_complaint_themes(db: Session, product_id: str = None) -> List[Dict[str, Any]]:
    """
    Extracts thematic clusters from reviews using BERTopic.
    If product_id is provided, filters themes specifically for that product.
    """
    query = db.query(Review)
    if product_id:
        query = query.filter(Review.product_id == product_id)
    
    reviews = query.all()
    if not reviews or len(reviews) < 10:
        return [{"topic": "Insufficient Data", "count": len(reviews), "sentiment": "neutral"}]

    docs = [r.comment for r in reviews]
    model = _get_topic_model()

    if model == "FALLBACK":
        # Simple keyword fallback logic for portfolio resilience
        keywords = ["battery", "build", "software", "delivery", "price", "support"]
        themes = []
        for kw in keywords:
            count = sum(1 for d in docs if kw in d.lower())
            if count > 0:
                themes.append({"topic": f"Issues matching '{kw}'", "count": count, "relevance_score": round(count/len(docs), 2)})
        return sorted(themes, key=lambda x: x["count"], reverse=True)

    try:
        topics, probs = model.fit_transform(docs)
        freq = model.get_topic_info()
        
        results = []
        # Index 0 is often -1 (outliers), so we take higher frequency topics
        for _, row in freq.iterrows():
            topic_id = row['Topic']
            if topic_id == -1: continue # Skip noise
            
            name = row['Name'].split('_')[1:3] # Get top 2 words
            results.append({
                "topic": " ".join(name).title(),
                "count": int(row['Count']),
                "relevance_score": round(int(row['Count']) / len(docs), 2),
                "representative_docs": model.get_representative_docs(topic_id)[:2]
            })
            
            if len(results) >= 5: break # Keep top 5 themes

        return results
    except Exception as e:
        logger.error(f"BERTopic inference error: {e}")
        return [{"error": "Thematic analysis failed"}]

def should_stop_recommending(db: Session, product_id: str) -> Dict[str, Any]:
    """
    Advanced logic for Agent: Should we stop recommending X based on recent themes?
    """
    themes = get_product_complaint_themes(db, product_id)
    
    # Critical threshold: If > 30% of reviews mention a physical/software defect
    critical_keywords = ["broken", "drain", "quality", "crash", "flimsy", "flickering"]
    
    flagged = False
    reason = "Product is performing well."
    
    for theme in themes:
        if any(kw in str(theme.get("topic", "")).lower() for kw in critical_keywords):
            if theme["relevance_score"] > 0.25:
                flagged = True
                reason = f"High frequency of reports regarding '{theme['topic']}' ({int(theme['relevance_score']*100)}% of reviews)."
                break
                
    return {
        "product_id": product_id,
        "stop_recommendation": flagged,
        "primary_reason": reason,
        "discovered_themes": themes
    }
