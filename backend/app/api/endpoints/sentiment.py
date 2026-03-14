"""
Sentiment Analysis API Endpoints
POST /api/v1/sentiment/analyze          — analyze a single text
POST /api/v1/sentiment/batch            — analyze multiple texts
GET  /api/v1/sentiment/samples          — list demo reviews
GET  /api/v1/sentiment/samples/{id}     — analyze a specific demo review
"""
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.ml.sentiment import analyze, SAMPLE_REVIEWS, get_vader
from app.ml.topics import get_product_complaint_themes, should_stop_recommending

logger = logging.getLogger(__name__)
router = APIRouter()

# Pre-warm
try:
    get_vader()
    logger.info("Sentiment model ready.")
except Exception as e:
    logger.warning(f"Could not pre-warm sentiment model: {e}")


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., max_length=20)


@router.post("/sentiment/analyze")
async def analyze_text(req: TextRequest):
    """Analyze sentiment of a single text with aspect breakdown."""
    try:
        result = analyze(req.text)
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sentiment/batch")
async def batch_analyze(req: BatchRequest):
    """Analyze sentiment of multiple texts."""
    try:
        results = []
        for i, text in enumerate(req.texts):
            if not text.strip():
                continue
            result = analyze(text)
            result["index"] = i
            results.append(result)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/samples")
async def list_samples():
    """Return the list of demo product reviews."""
    return {"samples": SAMPLE_REVIEWS}


@router.get("/sentiment/samples/{review_id}")
async def analyze_sample(review_id: str):
    """Fetch and analyze a specific demo review."""
    review = next((r for r in SAMPLE_REVIEWS if r["id"] == review_id), None)
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found.")
    result = analyze(review["text"])
    return {
        "review": review,
        "analysis": result,
    }

@router.get("/sentiment/themes")
async def get_themes(product_id: str = None, db: Session = Depends(get_db)):
    """Discover thematic clusters in product reviews using BERTopic."""
    try:
        themes = get_product_complaint_themes(db, product_id)
        return {"product_id": product_id, "themes": themes}
    except Exception as e:
        logger.error(f"Topic discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/check-status/{product_id}")
async def check_product_status(product_id: str, db: Session = Depends(get_db)):
    """Advanced check: Should we stop recommending a product based on discovered complaints?"""
    try:
        status = should_stop_recommending(db, product_id)
        return status
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
