"""
Computer Vision API Endpoints
POST /api/v1/vision/analyze  — image analysis (classification, palette, edges)
POST /api/v1/vision/search   — CLIP visual product similarity search
GET  /api/v1/vision/clip-status — CLIP model health
"""
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.ml.vision import analyze_image
from app.ml.visual_search import search_by_image, get_clip_status

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
    CLIP Visual Product Similarity Search.
    Upload any image — system finds visually similar products from the catalog
    using CLIP ViT-B/32 embeddings + cosine similarity.
    Returns top-N matching products with similarity scores.
    """
    image_bytes = await _read_and_validate(file)
    try:
        results = search_by_image(image_bytes, top_n=top_n)
        if results and "error" in results[0]:
            raise HTTPException(status_code=503, detail=results[0]["error"])
        return {
            "query_image": file.filename,
            "model": "CLIP ViT-B/32",
            "results": results,
            "total_indexed": len(PRODUCTS),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visual search failed: {str(e)}")


@router.get("/vision/clip-status")
async def clip_status():
    """Return CLIP model and product embedding status."""
    return get_clip_status()

