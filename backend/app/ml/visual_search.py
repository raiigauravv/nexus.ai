"""
CLIP-based Visual Product Similarity Search
Uses sentence-transformers CLIP (ViT-B/32) to encode product catalog + user images,
then returns visually similar products via cosine similarity.

Product embeddings are pre-computed at startup from programmatically
generated category-representative images (PIL-drawn).
"""
import io
import logging
import numpy as np
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── CLIP singleton ──────────────────────────────────────────────────────────────
_clip_model = None
_product_embeddings: Optional[np.ndarray] = None
_indexed_products: Optional[list] = None
_pinecone_index = None


def _init_pinecone():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    
    from app.config import settings
    if not settings.PINECONE_API_KEY:
        return None
        
    try:
        from pinecone import Pinecone, ServerlessSpec
        logger.info("Connecting to Pinecone for visual search...")
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name = settings.PINECONE_VISION_INDEX
        
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index '{index_name}' with dimension 512...")
            pc.create_index(
                name=index_name,
                dimension=512,  # CLIP ViT-B/32 generates 512-dim vectors
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        _pinecone_index = pc.Index(index_name)
        logger.info("Pinecone vision index ready.")
    except Exception as e:
        logger.warning(f"Failed to initialize Pinecone vision index: {e}")
        _pinecone_index = None
        
    return _pinecone_index


def _get_clip():
    global _clip_model
    if _clip_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading CLIP ViT-B/32 model (first run: ~350MB download)…")
            _clip_model = SentenceTransformer("clip-ViT-B-32")
            logger.info("CLIP model loaded.")
        except Exception as e:
            logger.error(f"CLIP unavailable: {e}")
            _clip_model = None
    return _clip_model


# ── Category → visual theme ────────────────────────────────────────────────────
CATEGORY_THEMES = {
    "Electronics":    {"bg": (30,  80,  180), "accent": (80,  150, 255), "text": (255, 255, 255)},
    "Books":          {"bg": (139, 90,  43),  "accent": (200, 150, 80),  "text": (255, 245, 220)},
    "Clothing":       {"bg": (180, 60,  120), "accent": (240, 140, 180), "text": (255, 255, 255)},
    "Home & Kitchen": {"bg": (50,  130, 100), "accent": (100, 200, 160), "text": (255, 255, 255)},
    "Sports":         {"bg": (200, 80,  30),  "accent": (255, 160, 80),  "text": (255, 255, 255)},
    "Gaming":         {"bg": (80,  20,  150), "accent": (180, 80,  255), "text": (255, 255, 255)},
    "Beauty":         {"bg": (200, 100, 180), "accent": (255, 180, 220), "text": (80,  20,  60)},
    "Automotive":     {"bg": (50,  50,  60),  "accent": (120, 120, 140), "text": (255, 255, 255)},
}

# Category → abstract visual shapes (to differentiate embeddings)
SHAPE_ICONS = {
    "Electronics":    "▣ ◈ ⊞",
    "Books":          "≡ ≡ ≡",
    "Clothing":       "◇ ◆ ◇",
    "Home & Kitchen": "⌂ ⌂ ⌂",
    "Sports":         "● ○ ●",
    "Gaming":         "▶ ◀ ▶",
    "Beauty":         "✦ ✧ ✦",
    "Automotive":     "⬡ ⬢ ⬡",
}


def _make_product_image(product: Dict) -> Image.Image:
    """
    Generate a visually distinct PIL image for a product that encodes:
    - Category (via background color + accent color)
    - Price tier (via image brightness)
    - Name text (semantic content for CLIP)
    """
    category = product.get("category", "Electronics")
    theme = CATEGORY_THEMES.get(category, CATEGORY_THEMES["Electronics"])
    price = product.get("price", 100)

    # Image size
    W, H = 224, 224
    img = Image.new("RGB", (W, H), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    # Gradient-like background: draw accent diagonal band
    for i in range(0, W, 4):
        alpha = max(0, min(255, int(200 * (1 - i / W))))
        r = int(theme["bg"][0] + (theme["accent"][0] - theme["bg"][0]) * i / W)
        g = int(theme["bg"][1] + (theme["accent"][1] - theme["bg"][1]) * i / W)
        b = int(theme["bg"][2] + (theme["accent"][2] - theme["bg"][2]) * i / W)
        draw.line([(i, 0), (i, H)], fill=(r, g, b))

    # Price tier indicator: top-right corner brightness overlay
    price_tier = min(5, int(price / 100))
    for t in range(price_tier):
        x = W - 20 - t * 14
        draw.ellipse([x, 8, x + 10, 18], fill=theme["text"])

    # Shape icon (geometric differntiator by category)
    shapes = SHAPE_ICONS.get(category, "● ○ ●")
    try:
        font_large = ImageFont.load_default()
    except Exception:
        font_large = None

    # Draw shape text centered
    draw.text((W // 2, H // 3), shapes, fill=theme["text"], anchor="mm", font=font_large)

    # Draw product name (word-wrapped, CLIP reads text in images)
    name = product.get("name", "")
    words = name.split()
    lines = []
    current = []
    for w in words:
        current.append(w)
        if len(" ".join(current)) > 22:
            lines.append(" ".join(current[:-1]))
            current = [w]
    if current:
        lines.append(" ".join(current))

    y_start = H // 2 + 10
    for line in lines[:3]:
        draw.text((W // 2, y_start), line, fill=theme["text"], anchor="mm", font=font_large)
        y_start += 16

    # Category label at bottom
    draw.rectangle([0, H - 28, W, H], fill=(*theme["accent"], 200))
    draw.text((W // 2, H - 14), category.upper(), fill=theme["bg"], anchor="mm", font=font_large)

    return img


def _build_product_embeddings():
    """Pre-compute CLIP embeddings for all 25 products."""
    from app.ml.recommender import PRODUCTS

    model = _get_clip()
    if model is None:
        return None, None

    logger.info("Pre-computing CLIP embeddings for product catalog…")
    images = []
    for prod in PRODUCTS:
        img = _make_product_image(prod)
        images.append(img)

    # Encode all images at once (batch for efficiency)
    embeddings = model.encode(images, batch_size=8, show_progress_bar=False)
    logger.info(f"CLIP embeddings ready: {embeddings.shape}")
    
    # Try to upsert into Pinecone
    pc_idx = _init_pinecone()
    if pc_idx:
        try:
            logger.info("Upserting product embeddings to Pinecone...")
            vectors = []
            for i, prod in enumerate(PRODUCTS):
                vectors.append((
                    prod["id"],
                    embeddings[i].tolist(),
                    {"name": prod["name"], "category": prod["category"], "price": prod["price"]}
                ))
            # Pinecone accepts up to 100-200 vectors at a time, we have 100 max so it's fine
            pc_idx.upsert(vectors=vectors)
            logger.info("Pinecone upsert complete.")
        except Exception as e:
            logger.error(f"Failed to upsert to Pinecone: {e}")

    return np.array(embeddings), PRODUCTS


def get_product_embeddings():
    """Lazy-load product embeddings."""
    global _product_embeddings, _indexed_products
    if _product_embeddings is None:
        _product_embeddings, _indexed_products = _build_product_embeddings()
    return _product_embeddings, _indexed_products


def search_by_image(image_bytes: bytes, top_n: int = 5) -> List[Dict]:
    """
    Given uploaded image bytes, return top-N visually similar products.
    Uses CLIP ViT-B/32 cosine similarity.
    """
    model = _get_clip()
    if model is None:
        return [{"error": "CLIP model not available. Install sentence-transformers + torch."}]

    product_embeddings, products = get_product_embeddings()
    if product_embeddings is None:
        return [{"error": "Product embeddings could not be computed."}]

    # Encode user image
    try:
        user_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        user_img_resized = user_img.resize((224, 224))
        user_embedding = model.encode([user_img_resized], show_progress_bar=False)[0]
    except Exception as e:
        logger.error(f"CLIP encoding failed: {e}")
        return [{"error": f"Could not encode image: {e}"}]

    # Pinecone Query (preferred)
    pc_idx = _init_pinecone()
    if pc_idx:
        try:
            res = pc_idx.query(vector=user_embedding.tolist(), top_k=top_n, include_metadata=True)
            results = []
            for rank, m in enumerate(res.matches, 1):
                prod = next((p for p in products if p["id"] == m.id), None)
                if prod:
                    p = dict(prod)
                    p["visual_similarity"] = round(float(m.score), 4)
                    p["similarity_pct"] = round(float(m.score) * 100, 1)
                    p["rank"] = rank
                    results.append(p)
            return results
        except Exception as e:
            logger.warning(f"Pinecone query failed, falling back to local numpy: {e}")

    # Local numpy fallback (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    user_emb_2d = user_embedding.reshape(1, -1)
    similarities = cosine_similarity(user_emb_2d, product_embeddings)[0]

    # Sort descending
    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        prod = dict(products[idx])
        prod["visual_similarity"] = round(float(similarities[idx]), 4)
        prod["similarity_pct"] = round(float(similarities[idx]) * 100, 1)
        prod["rank"] = rank
        results.append(prod)

    return results


def get_clip_status() -> Dict:
    """Return CLIP model and embedding status for health checks."""
    model = _get_clip()
    embeddings, products = get_product_embeddings() if model else (None, None)
    
    pc_idx = _init_pinecone()
    pinecone_stats = None
    if pc_idx:
        try:
            pinecone_stats = pc_idx.describe_index_stats().to_dict()
        except:
            pass
            
    return {
        "clip_model_loaded": model is not None,
        "model_name": "clip-ViT-B-32",
        "product_embeddings_ready": embeddings is not None,
        "embedding_dim": int(embeddings.shape[1]) if embeddings is not None else None,
        "indexed_products": len(products) if products else 0,
        "pinecone_enabled": pc_idx is not None,
        "pinecone_stats": pinecone_stats,
    }


def search_by_description(text_query: str, top_k: int = 5) -> List[Dict]:
    """
    Text-to-product visual search using CLIP's joint text+image embedding space.
    Encodes the text query and finds the most visually/semantically similar products.

    Used by:
      - The agent tool `find_visually_similar_products`
      - The pytest test suite
      - /vision/search endpoint when a text query is provided
    """
    model = _get_clip()
    if model is None:
        return [{"error": "CLIP model not available."}]

    product_embeddings, products = get_product_embeddings()
    if product_embeddings is None or products is None:
        return [{"error": "Product embeddings not ready."}]

    try:
        # CLIP's text encoder maps to the same 512-dim space as its image encoder
        text_embedding = model.encode([text_query], show_progress_bar=False)[0]
    except Exception as e:
        logger.error(f"CLIP text encoding failed: {e}")
        return [{"error": f"Encoding failed: {e}"}]

    # Pinecone Query (preferred)
    pc_idx = _init_pinecone()
    if pc_idx:
        try:
            res = pc_idx.query(vector=text_embedding.tolist(), top_k=top_k, include_metadata=True)
            results = []
            for rank, m in enumerate(res.matches, 1):
                prod = next((p for p in products if p["id"] == m.id), None)
                if prod:
                    p = dict(prod)
                    p["similarity"]     = round(float(m.score), 4)
                    p["similarity_pct"] = round(float(m.score) * 100, 1)
                    p["rank"]           = rank
                    results.append(p)
            return results
        except Exception as e:
            logger.warning(f"Pinecone text query failed, falling back to local numpy: {e}")

    # Local numpy fallback
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    text_emb_2d  = text_embedding.reshape(1, -1)
    similarities = cos_sim(text_emb_2d, product_embeddings)[0]

    top_indices  = np.argsort(similarities)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        prod = dict(products[idx])
        prod["similarity"]     = round(float(similarities[idx]), 4)
        prod["similarity_pct"] = round(float(similarities[idx]) * 100, 1)
        prod["rank"]           = rank
        results.append(prod)

    return results
