"""
Computer Vision API Endpoints
POST /api/v1/vision/analyze   — image analysis (classification, palette, edges)
POST /api/v1/vision/search    — CLIP visual product similarity search + Gemini identification
POST /api/v1/vision/identify  — Gemini Vision product identification only
GET  /api/v1/vision/clip-status — CLIP model health
"""
import base64
import logging
import asyncio
import google.generativeai as genai
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.ml.vision import analyze_image
from app.ml.visual_search import search_by_image, get_clip_status
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
MAX_SIZE_MB = 15


async def _read_and_validate(file: UploadFile) -> bytes:
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Supported: JPEG, PNG, WebP, BMP, TIFF."
        )
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Maximum is {MAX_SIZE_MB}MB."
        )
    return image_bytes


async def _gemini_identify_product(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    """
    Use Gemini Vision to identify the exact product, brand, model, and features.
    Returns structured identification data.
    """
    if not settings.GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Build the prompt for structured product identification
        prompt = """You are an expert product identification AI. Analyze this image and identify the product.

Respond in this EXACT JSON format (no markdown, just raw JSON):
{
  "identified": true,
  "product_name": "Full official product name",
  "brand": "Brand/Manufacturer",
  "model": "Specific model number or variant if visible",
  "category": "Product category (e.g. Audio, Electronics, Clothing)",
  "description": "2-3 sentence description of what you see and its key features",
  "key_features": ["feature 1", "feature 2", "feature 3"],
  "estimated_price_range": "$XX - $XX USD",
  "confidence": "high|medium|low",
  "similar_products": ["Alternative product 1", "Alternative product 2"]
}

If you cannot identify a specific product, set "identified": false and describe what you see generally.
Be specific — if you see Apple AirPods Max, say "Apple AirPods Max" not just "headphones"."""

        # Convert image to base64 for Gemini
        image_data = base64.b64encode(image_bytes).decode("utf-8")

        response = model.generate_content([
            prompt,
            {"mime_type": content_type, "data": image_data}
        ])

        # Parse the JSON response
        import json
        text = response.text.strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        result["powered_by"] = "Gemini Vision"
        return result

    except Exception as e:
        logger.error(f"Gemini vision identification failed: {e}")
        return {
            "identified": False,
            "error": str(e),
            "powered_by": "Gemini Vision",
        }


@router.post("/vision/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Upload an image and receive:
    - Scene classification, dominant color palette, edge detection visualization,
    - Image statistics (brightness, contrast, saturation, sharpness), EXIF, tags
    """
    image_bytes = await _read_and_validate(file)
    try:
        return analyze_image(image_bytes, filename=file.filename or "upload")
    except Exception as e:
        logger.error(f"Vision analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


from app.ml.recommender import PRODUCTS


@router.post("/vision/search")
async def visual_product_search(file: UploadFile = File(...), top_n: int = 5):
    """
    Two-stage visual intelligence:
    1. Gemini Vision — identifies the EXACT product (brand, model, features)
    2. CLIP ViT-B/32 — finds visually similar items from the 150-product catalog

    Returns both the identification result and catalog matches.
    """
    image_bytes = await _read_and_validate(file)
    content_type = file.content_type or "image/jpeg"

    try:
        # Run Gemini identification + CLIP search in parallel
        identification_task = asyncio.create_task(
            _gemini_identify_product(image_bytes, content_type)
        )

        # CLIP search (sync, run in executor)
        loop = asyncio.get_event_loop()
        clip_results = await loop.run_in_executor(
            None, lambda: search_by_image(image_bytes, top_n=top_n)
        )

        identification = await identification_task

        if clip_results and "error" in clip_results[0]:
            raise HTTPException(status_code=503, detail=clip_results[0]["error"])

        return {
            "query_image": file.filename,
            "model": "CLIP ViT-B/32",
            "identification": identification,        # ← Gemini Vision result
            "results": clip_results,                 # ← CLIP similarity results
            "total_indexed": len(PRODUCTS),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")


@router.post("/vision/identify")
async def identify_product(file: UploadFile = File(...)):
    """
    Gemini Vision product identification only (no CLIP search).
    Identifies brand, model, features, and price range from a product photo.
    """
    image_bytes = await _read_and_validate(file)
    content_type = file.content_type or "image/jpeg"
    try:
        result = await _gemini_identify_product(image_bytes, content_type)
        return result
    except Exception as e:
        logger.error(f"Product identification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")


@router.get("/vision/clip-status")
async def clip_status():
    """Return CLIP model and product embedding status."""
    return get_clip_status()
