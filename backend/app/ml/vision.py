"""
Computer Vision Analysis Pipeline
Uses Pillow + NumPy for:
  - Scene/content classification (rule-based feature analysis)
  - Dominant color palette extraction
  - Image statistics (brightness, contrast, saturation, sharpness)
  - Edge detection (Sobel-like convolution)
  - Object/tag detection via color & feature heuristics
  - EXIF metadata extraction
"""
import io
import base64
import math
import logging
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageStat, ImageFilter, ImageEnhance
import numpy as np

logger = logging.getLogger(__name__)

# ── Class Labels & Rules ───────────────────────────────────────────────────────

SCENE_CLASSES = [
    "Landscape / Nature",
    "Urban / Architecture",
    "Portrait / People",
    "Food / Cuisine",
    "Animals / Wildlife",
    "Technology / Electronics",
    "Art / Abstract",
    "Indoor / Interior",
    "Night Scene",
    "Aerial / Satellite",
]

# Tag vocabularies used in heuristic tagging
WARM_TAGS = ["warm tones", "golden hour", "sunset vibes", "earthy palette"]
COOL_TAGS = ["cool tones", "blue hour", "overcast", "monochrome"]
VIBRANT_TAGS = ["vibrant", "high saturation", "colorful", "vivid"]
DARK_TAGS = ["dark scene", "low-key lighting", "nighttime", "shadows"]
BRIGHT_TAGS = ["high key", "bright", "well-lit", "overexposed"]
SHARP_TAGS = ["high detail", "sharp focus", "fine texture", "crisp edges"]
BLUR_TAGS = ["bokeh / depth of field", "motion blur", "soft focus"]


def _rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert [0,255] RGB to HSV [0,1]."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    delta = mx - mn
    v = mx
    s = (delta / mx) if mx != 0 else 0
    if delta == 0:
        h = 0.0
    elif mx == r:
        h = ((g - b) / delta) % 6
    elif mx == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4
    h = (h * 60) % 360
    return h, s, v


def _quantize_palette(img: Image.Image, n_colors: int = 6) -> List[Dict]:
    """Extract dominant colors using PIL quantize."""
    small = img.convert("RGB").resize((150, 150))
    quantized = small.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    palette_raw = quantized.getpalette()
    
    # Count pixel occurrences per palette index
    data = np.array(quantized)
    total = data.size
    colors = []
    
    for i in range(n_colors):
        count = int((data == i).sum())
        if count == 0:
            continue
        r = palette_raw[i * 3]
        g = palette_raw[i * 3 + 1]
        b = palette_raw[i * 3 + 2]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        h, s, v = _rgb_to_hsv(r, g, b)
        colors.append({
            "hex": hex_color,
            "rgb": [r, g, b],
            "percentage": round(count / total * 100, 1),
            "hue": round(h, 1),
            "saturation": round(s, 4),
            "brightness": round(v, 4),
        })
    
    colors.sort(key=lambda x: x["percentage"], reverse=True)
    return colors[:n_colors]


def _compute_image_stats(img: Image.Image) -> Dict:
    """Compute brightness, contrast, sharpness, saturation from image."""
    rgb = img.convert("RGB")
    stat = ImageStat.Stat(rgb)

    # Mean brightness (0-255)
    brightness = float(np.mean(stat.mean))

    # Contrast (std dev of all channels)
    contrast = float(np.mean(stat.stddev))

    # Sharpness: variance of Laplacian
    gray = rgb.convert("L")
    gray_arr = np.array(gray, dtype=np.float32)
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ], dtype=np.float32)
    # Manual convolution via PIL filter
    sharp_img = gray.filter(ImageFilter.SHARPEN)
    sharp_arr = np.array(sharp_img, dtype=np.float32)
    sharpness = float(np.var(sharp_arr - gray_arr) / 100.0)

    # Saturation: convert to HSV and average S channel
    hsv_arr = np.array(rgb, dtype=np.float32) / 255.0
    r, g, b = hsv_arr[:, :, 0], hsv_arr[:, :, 1], hsv_arr[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    sat = np.where(mx > 0, delta / mx, 0)
    saturation = float(np.mean(sat))

    return {
        "brightness": round(brightness / 255.0, 4),
        "brightness_pct": round(brightness / 255.0 * 100, 1),
        "contrast": round(contrast / 128.0, 4),
        "sharpness": round(min(sharpness, 1.0), 4),
        "saturation": round(saturation, 4),
        "saturation_pct": round(saturation * 100, 1),
    }


def _classify_scene(stats: Dict, palette: List[Dict], img: Image.Image) -> List[Dict]:
    """Rule-based scene classification producing probability distribution."""
    b = stats["brightness"]
    c = stats["contrast"]
    s = stats["saturation"]
    sharp = stats["sharpness"]
    w, h = img.size
    aspect = w / max(h, 1)

    # Build raw scores
    scores: Dict[str, float] = {cls: 0.0 for cls in SCENE_CLASSES}

    # Night scene: very dark
    if b < 0.25:
        scores["Night Scene"] += 0.6
        scores["Urban / Architecture"] += 0.2

    # Landscape: high saturation, green/blue dominant
    dominant_hue = palette[0]["hue"] if palette else 180
    if 60 <= dominant_hue <= 180 and s > 0.25:
        scores["Landscape / Nature"] += 0.5
    if s > 0.35 and b > 0.4:
        scores["Landscape / Nature"] += 0.2

    # Urban: high contrast, low saturation
    if c > 0.4 and s < 0.25:
        scores["Urban / Architecture"] += 0.4
    if aspect > 1.3 and c > 0.35:
        scores["Urban / Architecture"] += 0.15

    # Portrait: face-like tones (warm skin), moderate brightness
    skin_hue_palette = [p for p in palette if 5 <= p["hue"] <= 40 and p["saturation"] > 0.15]
    if skin_hue_palette and b > 0.35:
        scores["Portrait / People"] += 0.5
    if aspect < 0.85:  # portrait orientation
        scores["Portrait / People"] += 0.1

    # Food: warm/orange tones, moderate saturation
    warm_palette = [p for p in palette if 10 <= p["hue"] <= 50]
    if warm_palette and s > 0.2 and b > 0.3:
        scores["Food / Cuisine"] += 0.3
    if len(warm_palette) >= 2:
        scores["Food / Cuisine"] += 0.2

    # Technology: gray/silver tones, high contrast
    gray_tones = [p for p in palette if p["saturation"] < 0.15]
    if len(gray_tones) >= 2 and c > 0.3:
        scores["Technology / Electronics"] += 0.4
    if sharp > 0.6 and c > 0.35:
        scores["Technology / Electronics"] += 0.15

    # Art/Abstract: high saturation, multiple vivid colors
    if s > 0.4 and c > 0.4 and len(palette) >= 4:
        scores["Art / Abstract"] += 0.4
    if len({round(p["hue"] / 60) for p in palette}) >= 4:
        scores["Art / Abstract"] += 0.2

    # Animals: warm/brown tones, high sharpness
    brown_palette = [p for p in palette if 20 <= p["hue"] <= 45 and p["saturation"] > 0.2]
    if brown_palette and sharp > 0.4:
        scores["Animals / Wildlife"] += 0.35

    # Indoor: medium brightness, medium contrast
    if 0.3 < b < 0.7 and 0.2 < c < 0.5:
        scores["Indoor / Interior"] += 0.2

    # Aerial: high blue content, wide aspect
    blue_palette = [p for p in palette if 190 <= p["hue"] <= 250]
    if blue_palette and aspect > 1.4:
        scores["Aerial / Satellite"] += 0.35

    # Normalize to probabilities via softmax-like
    total = sum(scores.values()) or 1.0
    results = []
    for cls, score in sorted(scores.items(), key=lambda x: -x[1]):
        if score > 0:
            results.append({
                "label": cls,
                "confidence": round(score / total, 4),
                "score": round(score, 4),
            })

    # Normalize confidences to sum to ~1
    conf_total = sum(r["confidence"] for r in results) or 1.0
    for r in results:
        r["confidence"] = round(r["confidence"] / conf_total, 4)

    return results[:5] if results else [{"label": "General / Uncategorized", "confidence": 1.0, "score": 0.1}]


def _generate_tags(stats: Dict, palette: List[Dict], scene: str) -> List[str]:
    """Generate descriptive tags from image features."""
    tags = []
    b = stats["brightness"]
    s = stats["saturation"]
    sharp = stats["sharpness"]
    c = stats["contrast"]

    # Tone tags
    dominant_hue = palette[0]["hue"] if palette else 180
    if dominant_hue < 60 or dominant_hue > 300:
        tags += WARM_TAGS[:2]
    else:
        tags += COOL_TAGS[:2]

    if s > 0.35:
        tags.append(VIBRANT_TAGS[0])
    if b < 0.3:
        tags.append(DARK_TAGS[0])
    elif b > 0.7:
        tags.append(BRIGHT_TAGS[0])
    if sharp > 0.5:
        tags.append(SHARP_TAGS[0])
    elif sharp < 0.15:
        tags.append(BLUR_TAGS[0])
    if c > 0.5:
        tags.append("high contrast")

    # Scene tags
    tags.append(scene.lower().split("/")[0].strip())

    return list(dict.fromkeys(tags))[:8]  # deduplicate, limit


def _edge_detection_b64(img: Image.Image, max_size: int = 400) -> str:
    """Apply edge detection and return as base64 PNG."""
    gray = img.convert("L")
    # Resize for performance
    w, h = gray.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        gray = gray.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    buf = io.BytesIO()
    edges.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _thumbnail_b64(img: Image.Image, size: int = 400) -> str:
    """Return a resized base64 thumbnail."""
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    rgb_thumb = thumb.convert("RGB")
    buf = io.BytesIO()
    rgb_thumb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_image(image_bytes: bytes, filename: str = "image") -> Dict:
    """Full computer vision analysis pipeline."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    mode = img.mode
    format_ = img.format or "JPEG"

    # ── Extract EXIF ─────────────────────────────────────────────────
    exif_data = {}
    try:
        exif = img._getexif()
        if exif:
            from PIL.ExifTags import TAGS
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, str(tag_id))
                if tag in ("Make", "Model", "DateTime", "Software", "GPSInfo"):
                    exif_data[tag] = str(value)
    except Exception:
        pass

    # ── Stats & features ─────────────────────────────────────────────
    stats = _compute_image_stats(img)
    palette = _quantize_palette(img, n_colors=6)
    scene_predictions = _classify_scene(stats, palette, img)
    top_scene = scene_predictions[0]["label"] if scene_predictions else "Unknown"
    tags = _generate_tags(stats, palette, top_scene)

    # ── Visual outputs ────────────────────────────────────────────────
    edge_b64 = _edge_detection_b64(img)
    thumb_b64 = _thumbnail_b64(img)

    # ── Megapixels ────────────────────────────────────────────────────
    megapixels = round((w * h) / 1_000_000, 2)

    return {
        "filename": filename,
        "metadata": {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / max(h, 1), 3),
            "mode": mode,
            "format": format_,
            "megapixels": megapixels,
            "file_size_kb": round(len(image_bytes) / 1024, 1),
            "exif": exif_data,
        },
        "statistics": stats,
        "palette": palette,
        "classification": scene_predictions,
        "top_label": top_scene,
        "tags": tags,
        "visuals": {
            "thumbnail": thumb_b64,
            "edges": edge_b64,
        },
    }
