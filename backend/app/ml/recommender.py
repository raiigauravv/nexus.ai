"""
Recommendation Engine — Hybrid Collaborative Filtering + Content-Based
======================================================================
Training data:
  - 500 synthetic users with persona-weighted category affinities
  - 100 products across 10 categories
  - ~15,000 interaction events, affinity-weighted (purchase=5, click=3, view=1)
  - Truncated SVD (rank-50) on the full 500×100 interaction matrix

This gives the model enough signal to produce meaningful collaborative
filtering predictions that generalise beyond the 8 demo users.
"""

import numpy as np
import random
import logging
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# ── Categories ─────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Electronics", "Books", "Clothing", "Home & Kitchen",
    "Sports", "Gaming", "Beauty", "Automotive", "Music", "Health",
]

# ── Product Catalog (150 products) ────────────────────────────────────────────
PRODUCTS: List[Dict] = []

# Helper to generate products for catalog expansion
def _seed_catalog():
    global PRODUCTS
    base_data = {
        "Electronics": [
            ("ProBook Ultra Laptop 15\"", 1299.99, ["laptop", "work", "portable"]),
            ("SoundWave Pro Headphones", 249.99, ["audio", "wireless", "music"]),
            ("UltraView 4K Monitor 27\"", 449.99, ["monitor", "display", "4k"]),
            ("SmartCharge Wireless Pad", 39.99, ["charging", "wireless"]),
            ("StreamCam 4K Webcam", 129.99, ["webcam", "video"]),
            ("PixelDrop Drone 4K", 399.99, ["drone", "camera"]),
            ("NanoTab 10 Pro Tablet", 549.99, ["tablet", "portable"]),
            ("SmartHome Hub X3", 89.99, ["smart-home", "iot"]),
            ("ErgoPro VR Headset", 699.99, ["vr", "gaming", "immersive"]),
            ("PocketCast Mini Speaker", 59.99, ["audio", "bluetooth"]),
            ("LinkMaster Mesh Router", 199.99, ["network", "wifi"]),
            ("PowerBank 20k mAh", 49.99, ["battery", "travel"]),
            ("NoiseStop Earbuds", 129.99, ["audio", "anc"]),
            ("StreamDeck Pro", 149.99, ["streaming", "gaming"]),
            ("SmartLock Gen 4", 229.99, ["security", "home"])
        ],
        "Books": [
            ("Deep Learning: A Modern Approach", 49.99, ["AI", "technical"]),
            ("The Psychology of Money", 16.99, ["finance", "psychology"]),
            ("Designing Data-Intensive Apps", 55.00, ["engineering", "technical"]),
            ("Atomic Habits", 14.99, ["habits", "popular"]),
            ("Clean Code", 39.99, ["programming", "technical"]),
            ("The Lean Startup", 17.99, ["business", "startup"]),
            ("Sapiens: A Brief History", 15.99, ["history", "non-fiction"]),
            ("System Design Interview", 42.99, ["engineering", "interview"]),
            ("The Pragmatic Programmer", 44.99, ["programming", "career"]),
            ("Think and Grow Rich", 12.99, ["self-help", "finance"]),
            ("Zero to One", 18.99, ["startup", "business"]),
            ("Hackers & Painters", 24.00, ["essays", "tech"]),
            ("The Mythical Man-Month", 35.00, ["management", "engineering"]),
            ("Explainable AI", 59.99, ["AI", "ethics"]),
            ("Site Reliability Engineering", 45.00, ["SRE", "operations"])
        ],
        "Clothing": [
            ("FlexFit Athletic Shorts", 34.99, ["sports", "fitness"]),
            ("Urban Tech Jacket", 119.99, ["jacket", "outdoor"]),
            ("Performance Running Shoes", 89.99, ["running", "shoes"]),
            ("Merino Wool Sweater", 79.99, ["casual", "premium"]),
            ("Compression Leggings Pro", 54.99, ["fitness", "yoga"]),
            ("WaterResist Hiking Pants", 69.99, ["outdoor", "hiking"]),
            ("Classic Oxford Dress Shirt", 49.99, ["formal", "office"]),
            ("CoolMax Running Tank", 29.99, ["running", "breathable"]),
            ("Sherpa Fleece Hoodie", 59.99, ["casual", "warm"]),
            ("Minimalist Leather Sneakers", 94.99, ["shoes", "minimalist"]),
            ("SolarShield Cap", 24.99, ["accessory", "sun"]),
            ("QuickDry Swim Trunks", 39.99, ["swim", "beach"]),
            ("EcoCotton Tee 3-Pack", 45.00, ["casual", "sustainable"]),
            ("Heavyweight Denim Jeans", 85.00, ["casual", "classic"]),
            ("All-Weather Parka", 159.99, ["winter", "outdoor"])
        ],
        "Home & Kitchen": [
            ("Smart Coffee Maker Pro", 179.99, ["coffee", "smart"]),
            ("AirPure HEPA Purifier", 269.99, ["air", "health"]),
            ("ChefBlend Pro Blender", 89.99, ["blender", "cooking"]),
            ("InstaCook Pressure Cooker", 129.99, ["cooking", "smart"]),
            ("Sous Vide Precision Cooker", 149.99, ["cooking", "precision"]),
            ("Nordic Cast Iron Skillet", 69.99, ["cookware", "iron"]),
            ("BambooKnife Chef Set (8pc)", 89.99, ["knives", "professional"]),
            ("Smart Herb Garden Kit", 49.99, ["gardening", "iot"]),
            ("Robot Vacuum Omega 9", 349.99, ["vacuum", "cleaning"]),
            ("Modular Storage System", 79.99, ["storage", "home"]),
            ("Electric Kettle Temp Control", 59.99, ["tea", "kitchen"]),
            ("Silicone Baking Mat Set", 25.00, ["baking", "eco"]),
            ("Personal Space Heater", 45.99, ["comfort", "office"]),
            ("Memory Foam Pillow", 39.99, ["sleep", "bedroom"]),
            ("Adjustable Standing Desk", 499.00, ["office", "furniture"])
        ],
        "Sports": [
            ("PowerTrack Smart Watch", 199.99, ["tracker", "health"]),
            ("ProGrip Yoga Mat", 49.99, ["yoga", "fitness"]),
            ("Resistance Band Set (11pc)", 24.99, ["fitness", "strength"]),
            ("Carbon Fiber Road Bike", 1199.99, ["cycling", "outdoor"]),
            ("AdjustaDumbell 90lb Set", 299.99, ["weights", "home-gym"]),
            ("TrailBlazer Trekking Poles", 59.99, ["hiking", "outdoor"]),
            ("SwimPro Goggles X7", 34.99, ["swimming", "fitness"]),
            ("Pro Jump Rope (speed)", 19.99, ["cardio", "training"]),
            ("Recovery Foam Roller Set", 44.99, ["recovery", "health"]),
            ("GPS Sport Tracker Clip", 89.99, ["gps", "outdoor"]),
            ("Kettlebell Set (5-25lb)", 129.99, ["strength", "gym"]),
            ("Boxing Glove Pro Series", 75.00, ["martial-arts", "fitness"]),
            ("Basketball Indoor/Outdoor", 29.99, ["ball", "sport"]),
            ("Tennis Racket Graphite", 149.00, ["tennis", "performance"]),
            ("Hydration Pack 2L", 55.00, ["running", "hiking"])
        ],
        "Gaming": [
            ("HyperEdge Gaming Mouse", 69.99, ["mouse", "rgb"]),
            ("NovaSeat Gaming Chair", 299.99, ["chair", "ergonomic"]),
            ("Dual-Zone Controller Pro", 79.99, ["console", "haptic"]),
            ("4K 144Hz Gaming Monitor", 599.99, ["monitor", "high-refresh"]),
            ("Surround Sound Headset 7.1", 119.99, ["audio", "surround"]),
            ("LED Desk Lamp Smart RGB", 44.99, ["desk", "lighting"]),
            ("Mechanical Numpad TKL", 64.99, ["keyboard", "compact"]),
            ("PS5/Xbox Capture Card 4K", 149.99, ["streaming", "capture"]),
            ("Gaming Router AX5400", 249.99, ["network", "speed"]),
            ("Portable SSD 2TB", 169.99, ["storage", "fast"]),
            ("VR Cable Management Kit", 29.99, ["vr", "accessory"]),
            ("Cooling Pad for Laptops", 39.99, ["laptop", "thermal"]),
            ("Flight Sim Joystick", 189.00, ["simulator", "joystick"]),
            ("Racing Wheel & Pedals", 349.99, ["racing", "sim"]),
            ("Acoustic Foam Panels (12p)", 49.99, ["studio", "gaming"])
        ],
        "Beauty": [
            ("Glow Serum 30ml", 44.99, ["skincare", "vitamin-c"]),
            ("HydraMist Face Spray", 19.99, ["skincare", "refreshing"]),
            ("Rose Quartz Roller", 25.00, ["beauty", "face"]),
            ("Night Repair Cream", 59.99, ["skincare", "anti-aging"]),
            ("Organic Argan Oil", 34.99, ["hair", "skin"]),
            ("LashDefine Volumizer", 22.99, ["makeup", "eyes"]),
            ("Matte Finish Foundation", 38.00, ["makeup", "base"]),
            ("Sunscreen SPF 50 Broad", 28.50, ["skincare", "sun"]),
            ("Silk Sleep Mask", 18.00, ["accessory", "sleep"]),
            ("Charcoal Detox Mask", 14.99, ["skincare", "cleansing"]),
            ("Electric Face Cleanser", 89.00, ["beauty", "device"]),
            ("Vitamin E Lip Balm", 9.99, ["lips", "health"]),
            ("Hand Repair Salve", 12.50, ["skin", "winter"]),
            ("Sandalwood Beard Oil", 24.00, ["grooming", "beard"]),
            ("Pro Makeup Brush Set", 65.00, ["makeup", "tools"])
        ],
        "Automotive": [
            ("DashCam Pro 4K Stealth", 159.99, ["camera", "security"]),
            ("Smart OBD-II Scanner", 49.99, ["diagnostic", "tech"]),
            ("NanoWax Polish 500ml", 24.99, ["cleaning", "shine"]),
            ("Jump Starter 2000A", 99.99, ["battery", "emergency"]),
            ("Leather Interior Cleaner", 19.50, ["interior", "care"]),
            ("Magnetic Phone Car Mount", 15.99, ["accessory", "phone"]),
            ("Tire Pressure Gauge Dig", 22.00, ["safety", "tools"]),
            ("Portable Air Compressor", 65.00, ["tools", "safety"]),
            ("Sun Shade reflective X", 29.99, ["interior", "summer"]),
            ("Car Vacuum Cordless", 55.00, ["cleaning", "portable"]),
            ("Roof Rack Universal", 189.99, ["travel", "cargo"]),
            ("HEPA Cabin Air Filter", 18.00, ["maintenance", "health"]),
            ("Microfiber Cloth 10-pack", 14.99, ["cleaning", "detailing"]),
            ("LED Headlight Kit", 85.00, ["lighting", "safety"]),
            ("Backseat Organizer", 24.50, ["storage", "travel"])
        ],
        "Music": [
            ("Electric Guitar Strat-Style", 399.99, ["instrument", "rock"]),
            ("Digital Stage Piano 88", 749.99, ["instrument", "keys"]),
            ("Studio Condenser Mic", 199.99, ["recording", "vocals"]),
            ("Drum Pad Controller", 89.99, ["beats", "production"]),
            ("XLR Cable Gold 10ft", 24.99, ["audio", "pro"]),
            ("Guitar Strings (3-pack)", 18.00, ["accessory", "strings"]),
            ("Active Monitor Speakers", 299.99, ["audio", "studio"]),
            ("Ukulele Concert Wood", 65.00, ["instrument", "acoustic"]),
            ("Dynamic Karaoke Mic", 45.00, ["entertainment", "party"]),
            ("Music Stand Adjustable", 28.00, ["accessory", "stand"]),
            ("Vinyl Turntable Hi-Fi", 179.99, ["audio", "retro"]),
            ("Headphone Amp Pro", 125.00, ["audio", "hifi"]),
            ("Mandolin Sunburst", 149.99, ["instrument", "folk"]),
            ("Saxophone Reed Box", 32.00, ["accessory", "wind"]),
            ("Music Production DAW Soft", 249.00, ["software", "pro"])
        ],
        "Health": [
            ("Digital Thermometer Pro", 24.99, ["medical", "safety"]),
            ("Weighted Anxiety Blanket", 85.00, ["sleep", "mental-health"]),
            ("Omega-3 Fish Oil 180ct", 22.00, ["supplement", "wellness"]),
            ("Posture Corrector Pro", 34.00, ["back", "health"]),
            ("Diffuser & Oils Set", 49.99, ["aromatherapy", "home"]),
            ("Automatic Blood Pressure", 59.99, ["medical", "health"]),
            ("Pulse Oximeter Elite", 29.50, ["medical", "oxygen"]),
            ("Light Therapy Lamp", 68.00, ["mental-health", "winter"]),
            ("First Aid Kit (250pc)", 45.00, ["emergency", "safety"]),
            ("Natural Multivitamin", 28.00, ["supplement", "wellness"]),
            ("Blue Light Filter Glasses", 24.99, ["eyes", "tech"]),
            ("Ergonomic Keyboard Rest", 19.99, ["office", "health"]),
            ("Reusable Cold Pack", 15.00, ["recovery", "medical"]),
            ("Deep Tissue Massager Gun", 129.99, ["recovery", "fitness"]),
            ("UV Sanitizer Box", 79.99, ["hygiene", "tech"])
        ]
    }
    
    id_counter = 1
    for category, items in base_data.items():
        for name, price, tags in items:
            PRODUCTS.append({
                "id": f"P{id_counter:03d}",
                "name": name,
                "category": category,
                "price": price,
                "rating": round(random.uniform(4.0, 5.0), 1),
                "tags": tags + [category.lower()]
            })
            id_counter += 1

_seed_catalog()

# ── 10 Demo Users (displayed on the frontend) ──────────────────────────────────
USERS = [
    {"id": "U001", "name": "Alex Chen",      "avatar": "AC", "persona": "tech_professional"},
    {"id": "U002", "name": "Maria Garcia",   "avatar": "MG", "persona": "fitness_enthusiast"},
    {"id": "U003", "name": "James Wilson",   "avatar": "JW", "persona": "gamer"},
    {"id": "U004", "name": "Emma Davis",     "avatar": "ED", "persona": "bookworm"},
    {"id": "U005", "name": "Liam Brown",     "avatar": "LB", "persona": "home_chef"},
    {"id": "U006", "name": "Sophia Lee",     "avatar": "SL", "persona": "fashionista"},
    {"id": "U007", "name": "Noah Martinez",  "avatar": "NM", "persona": "outdoor_adventurer"},
    {"id": "U008", "name": "Olivia Taylor",  "avatar": "OT", "persona": "beauty_enthusiast"},
    {"id": "U009", "name": "Ethan Park",     "avatar": "EP", "persona": "musician"},
    {"id": "U010", "name": "Ava Robinson",   "avatar": "AR", "persona": "health_conscious"},
]

# Per-persona category affinity weights (sum to 1.0)
PERSONA_AFFINITIES: Dict[str, Dict[str, float]] = {
    "tech_professional":    {"Electronics": 0.45, "Books": 0.25, "Gaming": 0.15, "Health": 0.05, "Music": 0.05, "Automotive": 0.05},
    "fitness_enthusiast":   {"Sports": 0.40, "Health": 0.25, "Clothing": 0.20, "Home & Kitchen": 0.10, "Beauty": 0.05},
    "gamer":                {"Gaming": 0.50, "Electronics": 0.25, "Music": 0.10, "Books": 0.10, "Automotive": 0.05},
    "bookworm":             {"Books": 0.55, "Home & Kitchen": 0.15, "Health": 0.10, "Beauty": 0.10, "Clothing": 0.10},
    "home_chef":            {"Home & Kitchen": 0.45, "Books": 0.20, "Health": 0.15, "Beauty": 0.10, "Sports": 0.10},
    "fashionista":          {"Clothing": 0.45, "Beauty": 0.30, "Sports": 0.10, "Home & Kitchen": 0.10, "Books": 0.05},
    "outdoor_adventurer":   {"Sports": 0.40, "Clothing": 0.25, "Automotive": 0.15, "Health": 0.10, "Electronics": 0.10},
    "beauty_enthusiast":    {"Beauty": 0.50, "Clothing": 0.25, "Health": 0.15, "Home & Kitchen": 0.10},
    "musician":             {"Music": 0.40, "Electronics": 0.30, "Books": 0.15, "Gaming": 0.10, "Health": 0.05},
    "health_conscious":     {"Health": 0.35, "Sports": 0.25, "Home & Kitchen": 0.20, "Beauty": 0.10, "Books": 0.10},
}

# ── Synthetic Interaction Matrix (500 users × 100 products) ───────────────────

# Map product IDs to indices
_PROD_ID_TO_IDX = {p["id"]: i for i, p in enumerate(PRODUCTS)}
_CAT_TO_PROD_INDICES: Dict[str, List[int]] = {}
for _i, _p in enumerate(PRODUCTS):
    _CAT_TO_PROD_INDICES.setdefault(_p["category"], []).append(_i)

# 500 user personas for the latent interaction matrix
_EXTENDED_PERSONAS = list(PERSONA_AFFINITIES.keys())
_RNG = random.Random(42)

def _generate_large_interaction_matrix() -> np.ndarray:
    """
    Generate a 500×100 interaction matrix with realistic affinity-weighted
    purchase/click/view events.

    Event weights:
      purchase  → rating 4.5–5.0 (strong positive signal)
      click     → rating 2.5–3.5 (mild signal)
      view      → rating 0.5–1.5 (weak signal / noise)
    """
    n_users = 500
    n_items = len(PRODUCTS)
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    cat_names = list(_CAT_TO_PROD_INDICES.keys())

    for uid in range(n_users):
        # Assign a random persona (with slight variation)
        base_persona = _EXTENDED_PERSONAS[uid % len(_EXTENDED_PERSONAS)]
        affinities = PERSONA_AFFINITIES[base_persona].copy()

        # Add small random noise to affinities to differentiate users
        for cat in list(affinities.keys()):
            affinities[cat] = max(0.01, affinities[cat] + _RNG.gauss(0, 0.03))
        total = sum(affinities.values())
        affinities = {k: v / total for k, v in affinities.items()}

        # Simulate 8–25 interactions per user
        n_interactions = _RNG.randint(8, 25)
        interacted = set()

        for _ in range(n_interactions):
            # Pick category by affinity probability
            cats = list(affinities.keys())
            weights = [affinities.get(c, 0.01) for c in cats]
            cat = _RNG.choices(cats, weights=weights, k=1)[0]

            if cat not in _CAT_TO_PROD_INDICES:
                continue
            prod_indices = _CAT_TO_PROD_INDICES[cat]
            if not prod_indices:
                continue

            # Pick a product from that category, weighted by its rating
            prod_ratings = [PRODUCTS[i]["rating"] for i in prod_indices]
            item_idx = _RNG.choices(prod_indices, weights=prod_ratings, k=1)[0]

            if item_idx in interacted:
                continue
            interacted.add(item_idx)

            # Assign event type probabilistically
            event_roll = _RNG.random()
            if event_roll > 0.65:       # 35% purchase
                rating = _RNG.uniform(4.0, 5.0)
            elif event_roll > 0.30:     # 35% click
                rating = _RNG.uniform(2.0, 3.5)
            else:                       # 30% view only
                rating = _RNG.uniform(0.3, 1.5)

            matrix[uid, item_idx] = round(rating, 1)

    return matrix


# ── Model Singleton ────────────────────────────────────────────────────────────
_rec_bundle: dict | None = None


def _build_item_feature_matrix() -> np.ndarray:
    """Build item feature vectors from tags and category (for content-based)."""
    all_tags = sorted({tag for p in PRODUCTS for tag in p["tags"]})
    all_categories = sorted({p["category"] for p in PRODUCTS})
    tag_idx = {t: i for i, t in enumerate(all_tags)}
    cat_idx = {c: i for i, c in enumerate(all_categories)}

    n_items = len(PRODUCTS)
    n_features = len(all_tags) + len(all_categories)
    matrix = np.zeros((n_items, n_features))

    for pi, prod in enumerate(PRODUCTS):
        for tag in prod["tags"]:
            if tag in tag_idx:
                matrix[pi, tag_idx[tag]] = 1.0
        cat_offset = len(all_tags)
        if prod["category"] in cat_idx:
            matrix[pi, cat_offset + cat_idx[prod["category"]]] = 2.0  # stronger category weight

    return normalize(matrix)


def train_recommender() -> dict:
    """
    Train the hybrid recommendation engine with Ranking Evaluation (NDCG@10).
    """
    logger.info("Training recommendation engine on 500-user interaction matrix…")

    full_matrix = _generate_large_interaction_matrix()
    item_features = _build_item_feature_matrix()

    # ── Ranking Evaluation Split ─────────
    # Zero out 15% of interactions for testing
    train_matrix = full_matrix.copy()
    nonzero_coords = np.argwhere(full_matrix > 0)
    np.random.seed(42)
    test_indices = np.random.choice(len(nonzero_coords), int(0.15 * len(nonzero_coords)), replace=False)
    
    test_coords = nonzero_coords[test_indices]
    for r, c in test_coords:
        train_matrix[r, c] = 0

    # Truncated SVD (rank-50)
    k = min(50, min(train_matrix.shape) - 1)
    sparse_train = csr_matrix(train_matrix)
    U, sigma, Vt = svds(sparse_train, k=k)

    # Full predicted rating matrix
    predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
    predicted_ratings = np.clip(predicted_ratings, 0, 5)
    
    # ── Calculate NDCG@10 (Ranking Metric) ──
    def dcg(labels):
        return sum([l / np.log2(i + 2) for i, l in enumerate(labels)])

    ndcg_list = []
    for uid in range(full_matrix.shape[0]):
        # Truth: the items this user actually interacted with in the test set
        user_test_coords = test_coords[test_coords[:, 0] == uid]
        if len(user_test_coords) == 0: continue
        
        # Predicted: top 10 items for this user
        user_preds = predicted_ratings[uid]
        top_10_idx = np.argsort(user_preds)[::-1][:10]
        
        # Relevance: 1 if item was in test set, 0 otherwise
        test_item_indices = set(user_test_coords[:, 1])
        relevance = [1 if idx in test_item_indices else 0 for idx in top_10_idx]
        
        actual_dcg = dcg(relevance)
        ideal_relevance = sorted(relevance, reverse=True)
        ideal_dcg = dcg(ideal_relevance)
        
        if ideal_dcg > 0:
            ndcg_list.append(actual_dcg / ideal_dcg)

    final_ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0.0

    # ── Inject Visual Semantics ──
    try:
        from app.ml.visual_search import get_product_embeddings
        visual_emb, _ = get_product_embeddings()
        if visual_emb is not None:
            visual_emb_norm = normalize(visual_emb)
            item_features = normalize(np.hstack((item_features, visual_emb_norm * 1.5)))
    except Exception: pass

    item_sim_matrix = cosine_similarity(item_features)
    demo_user_row = {u["id"]: i for i, u in enumerate(USERS)}

    bundle = {
        "large_matrix":       full_matrix,
        "predicted_ratings":  predicted_ratings,
        "item_sim_matrix":    item_sim_matrix,
        "item_features":      item_features,
        "prod_id_to_idx":     {p["id"]: i for i, p in enumerate(PRODUCTS)},
        "user_id_to_idx":     demo_user_row,
        "n_users_trained":    full_matrix.shape[0],
        "svd_rank":           k,
        "ndcg_10":           final_ndcg,
    }

    # MLflow
    try:
        import mlflow
        import os
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
        mlflow.set_experiment("Nexus_Recommendations")
        with mlflow.start_run(run_name="hybrid_train_eval"):
            mlflow.log_param("svd_rank", k)
            mlflow.log_metrics({
                "n_users": full_matrix.shape[0],
                "matrix_density": float((full_matrix > 0).sum()) / full_matrix.size,
                "ndcg_10": final_ndcg
            })
    except Exception: pass

    print(f"--- RECOMMENDER METRICS ---")
    print(f"NDCG@10: {final_ndcg:.4f}")
    logger.info(f"Recommender ready. NDCG@10={final_ndcg:.4f}")
    return bundle


def get_recommender() -> dict:
    global _rec_bundle
    if _rec_bundle is None:
        _rec_bundle = train_recommender()
    return _rec_bundle


def get_recommendations(user_id: str, top_n: int = 6) -> List[Dict]:
    """
    Hybrid recommendation: 60% collaborative filtering (SVD) + 40% content-based.
    Excludes items the user has already interacted with.
    """
    bundle = get_recommender()
    uid = bundle["user_id_to_idx"].get(user_id)
    if uid is None:
        return []

    n_items = len(PRODUCTS)

    # Collaborative scores (from SVD reconstruction)
    cf_scores = bundle["predicted_ratings"][uid]

    # Content-based: items similar to ones this user liked
    user_ratings = bundle["large_matrix"][uid]
    liked_mask = user_ratings >= 3.5
    cb_scores = np.zeros(n_items)
    if liked_mask.any():
        liked_indices = np.where(liked_mask)[0]
        for li in liked_indices:
            if li < bundle["item_sim_matrix"].shape[0]:
                cb_scores += bundle["item_sim_matrix"][li] * user_ratings[li]
        cb_scores /= liked_mask.sum()

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    hybrid_scores = 0.60 * norm(cf_scores) + 0.40 * norm(cb_scores)

    # Exclude already-seen items
    seen_mask = user_ratings > 0
    hybrid_scores[seen_mask] = -1

    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        if hybrid_scores[idx] < 0:
            continue
        prod = dict(PRODUCTS[idx])
        prod["recommendation_score"] = round(float(hybrid_scores[idx]), 4)
        prod["cf_score"] = round(float(norm(cf_scores)[idx]), 4)
        prod["cb_score"] = round(float(norm(cb_scores)[idx]), 4)
        prod["match_reason"] = _explain(prod, bundle, uid)
        results.append(prod)

    return results


def get_similar_items(product_id: str, top_n: int = 5) -> List[Dict]:
    """Return content-based similar items."""
    bundle = get_recommender()
    idx = bundle["prod_id_to_idx"].get(product_id)
    if idx is None:
        return []

    sim_scores = bundle["item_sim_matrix"][idx].copy()
    sim_scores[idx] = -1

    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    results = []
    for i in top_indices:
        if sim_scores[i] < 0:
            continue
        prod = dict(PRODUCTS[i])
        prod["similarity_score"] = round(float(sim_scores[i]), 4)
        results.append(prod)
    return results


def get_trending(top_n: int = 8) -> List[Dict]:
    """Return most popular items by average predicted rating across all 500 users."""
    bundle = get_recommender()
    avg_scores = bundle["predicted_ratings"].mean(axis=0)
    interaction_counts = (bundle["large_matrix"] > 0).sum(axis=0)

    # Bayesian blend: predicted quality × interaction volume
    score = 0.6 * (avg_scores / avg_scores.max()) + 0.4 * (interaction_counts / interaction_counts.max())
    top_indices = np.argsort(score)[::-1][:top_n]

    results = []
    for idx in top_indices:
        prod = dict(PRODUCTS[idx])
        prod["trending_score"] = round(float(score[idx]), 4)
        prod["interaction_count"] = int(interaction_counts[idx])
        results.append(prod)
    return results


def _explain(prod: Dict, bundle: Dict, uid: int) -> str:
    """Generate a human-readable match reason."""
    user_ratings = bundle["large_matrix"][uid]
    liked_indices = np.where(user_ratings >= 4.0)[0]
    liked_products = [PRODUCTS[i] for i in liked_indices if i < len(PRODUCTS)]

    prod_tags = set(prod["tags"])
    for lp in liked_products:
        shared = prod_tags & set(lp["tags"])
        if shared:
            return f"Because you liked {lp['name']}"

    if liked_products:
        return f"Popular in {prod['category']}"
    return "Trending across all users"


def get_recommender_stats() -> dict:
    """Return training stats for MLflow logging and API responses."""
    bundle = get_recommender()
    matrix = bundle["large_matrix"]
    return {
        "n_users_trained": int(matrix.shape[0]),
        "n_products":      int(matrix.shape[1]),
        "n_interactions":  int((matrix > 0).sum()),
        "matrix_density":  round(float((matrix > 0).sum()) / matrix.size, 4),
        "svd_rank":        bundle["svd_rank"],
        "algorithm":       f"Hybrid SVD (rank={bundle['svd_rank']}, 60%) + Content-Based (40%)",
    }
