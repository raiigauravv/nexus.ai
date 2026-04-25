"""
CLIP-based Visual Product Similarity Search
Uses sentence-transformers CLIP (ViT-B/32) to encode product catalog + user images,
then returns visually similar products via cosine similarity.

KEY UPGRADE: Product embeddings are now generated from rich TEXT descriptions
using CLIP's text encoder (same 512-dim space as image encoder). This means a 
real photo of Apple AirPods Max will correctly match "premium over-ear wireless
headphones with cushioned ear cups" — unlike PIL-drawn colored rectangles which
produce low-quality embeddings incompatible with real product photography.
"""
import io
import logging
import numpy as np
from typing import List, Dict, Optional
from PIL import Image

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
            logger.info("Loading CLIP ViT-B/32 model (first run: ~350MB download)...")
            _clip_model = SentenceTransformer("clip-ViT-B-32")
            logger.info("CLIP model loaded.")
        except Exception as e:
            logger.error(f"CLIP unavailable: {e}")
            _clip_model = None
    return _clip_model


# ── Rich visual product descriptions for CLIP text embedding ──────────────────
# CLIP's text encoder operates in the same 512-dim embedding space as its image encoder.
# Detailed visual descriptions produce far better cross-modal image↔text matching.
PRODUCT_VISUAL_DESCRIPTIONS = {
    # Electronics — audio/headphones cluster
    "SoundWave Pro Headphones": "premium over-ear wireless headphones with large cushioned ear cups and adjustable headband in black and silver",
    "NoiseStop Earbuds": "small in-ear wireless earbuds true wireless with silicone ear tips in a glossy white charging case",
    "PocketCast Mini Speaker": "a small cylindrical portable bluetooth speaker in fabric mesh with metallic accents and LED indicator",
    "ErgoPro VR Headset": "a virtual reality headset with large display lenses foam face padding fabric strap worn on the head",
    # Electronics — computing
    "ProBook Ultra Laptop 15\"": "a sleek silver ultra-thin laptop computer with large display screen keyboard and trackpad on white background",
    "UltraView 4K Monitor 27\"": "a large widescreen flat panel computer monitor with thin bezel on a metallic stand",
    "NanoTab 10 Pro Tablet": "a slim tablet computer with large touchscreen display and thin aluminum chassis in space grey",
    "SmartCharge Wireless Pad": "a small flat round wireless charging pad in matte black with subtle LED ring indicator",
    "StreamCam 4K Webcam": "a compact digital webcam camera with glass lens mounted on a flexible clip stand in black",
    "PixelDrop Drone 4K": "a quadcopter drone with four propellers foldable arms camera gimbal stabilizer in dark grey",
    "SmartHome Hub X3": "a small smart home speaker hub with fabric mesh wrapping and colored LED ring light on the base",
    "LinkMaster Mesh Router": "a white triangular mesh wifi router with LED status indicator lights and antenna ports",
    "PowerBank 20k mAh": "a rectangular portable battery charger power bank in matte black with multiple USB ports",
    "StreamDeck Pro": "a small desktop control panel with programmable LCD key buttons and USB cable for streaming",
    "SmartLock Gen 4": "a digital keypad smart door lock in brushed nickel with touchscreen display and deadbolt mechanism",
    # Books
    "Deep Learning: A Modern Approach": "a thick technical hardcover textbook about deep learning neural networks with diagram on cover",
    "The Psychology of Money": "a popular bestseller paperback book with gold lettering on a simple clean cover design",
    "Designing Data-Intensive Apps": "a technical O'Reilly programming book about databases with an animal on the cover",
    "Atomic Habits": "a bright paperback self-help book with bold typography on a clean white and yellow cover",
    "Clean Code": "a software programming book with clean code principles on a dark professional cover",
    "The Lean Startup": "a business startup methodology book with a modern cover and bold title typography",
    "Sapiens: A Brief History": "a popular non-fiction history of humankind book with striking artistic cover artwork",
    "System Design Interview": "a technical software system design interview preparation book for engineers",
    "The Pragmatic Programmer": "a classic software programming career book with a crab on the cover",
    "Think and Grow Rich": "a vintage self-help finance success book with gold embossed lettering on dark cover",
    "Zero to One": "a startup business strategy book by Peter Thiel with bold minimal geometric design cover",
    "Hackers & Painters": "a collection of tech essays book with yellow and red color blocked cover design",
    "The Mythical Man-Month": "a classic software engineering project management book with anniversary edition cover",
    "Explainable AI": "a modern artificial intelligence ethics and machine learning book with abstract technology cover",
    "Site Reliability Engineering": "the Google SRE operations engineering book with a butterfly on the cover",
    # Clothing
    "FlexFit Athletic Shorts": "lightweight athletic running shorts in bright neon colors with mesh ventilation panels",
    "Urban Tech Jacket": "a modern technical outdoor jacket with multiple zippered pockets and hood in dark charcoal grey",
    "Performance Running Shoes": "colorful athletic running shoes with thick cushioned sole and breathable mesh upper",
    "Merino Wool Sweater": "a soft fine knitted merino wool pullover sweater in neutral oatmeal cream color",
    "Compression Leggings Pro": "high-waist black compression yoga leggings with textured fabric side panel detail",
    "WaterResist Hiking Pants": "durable water resistant cargo hiking pants in olive military green with reinforced knees",
    "Classic Oxford Dress Shirt": "a crisp white formal cotton oxford dress shirt with button-down collar and placket",
    "CoolMax Running Tank": "a moisture-wicking sleeveless athletic running tank top in bright performance color",
    "Sherpa Fleece Hoodie": "a soft thick fleece pullover hoodie with sherpa teddy lining in warm oatmeal beige",
    "Minimalist Leather Sneakers": "clean all-white minimalist genuine leather low-top sneakers with thin white rubber sole",
    "SolarShield Cap": "a lightweight sun protection baseball cap with UV blocking brim in khaki tan",
    "QuickDry Swim Trunks": "board shorts swimwear in tropical print pattern with elastic drawstring waist",
    "EcoCotton Tee 3-Pack": "basic organic cotton crew neck t-shirt in solid neutral color multipack",
    "Heavyweight Denim Jeans": "classic straight leg dark wash denim jeans with traditional five-pocket construction",
    "All-Weather Parka": "a heavy insulated winter parka coat with fur-trimmed hood in dark navy blue",
    # Home & Kitchen
    "Smart Coffee Maker Pro": "a modern brushed stainless steel programmable drip coffee maker with glass carafe and display",
    "AirPure HEPA Purifier": "a tall cylindrical white air purifier with HEPA filter and soft LED night light ring",
    "ChefBlend Pro Blender": "a powerful kitchen countertop blender with tall clear plastic pitcher and brushed metal base",
    "InstaCook Pressure Cooker": "a silver stainless steel electric multi-use pressure cooker instant pot with lid and sealing ring",
    "Sous Vide Precision Cooker": "a cylindrical immersion sous vide precision cooker with digital temperature display and clamp",
    "Nordic Cast Iron Skillet": "a heavy seasoned black cast iron frying skillet pan with long handle and pour spout",
    "BambooKnife Chef Set (8pc)": "a set of professional stainless steel chef kitchen knives with dark bamboo wood handles in a block",
    "Smart Herb Garden Kit": "a small white countertop hydroponic herb garden kit with LED grow light panel",
    "Robot Vacuum Omega 9": "a flat circular robotic vacuum cleaner in glossy black with infrared sensors",
    "Modular Storage System": "white modular cube open shelving unit system with clean Scandinavian design",
    "Electric Kettle Temp Control": "a matte black gooseneck electric pour-over kettle with digital temperature display",
    "Silicone Baking Mat Set": "a flat non-stick food-grade silicone baking sheet mat in light beige with measurement rings",
    "Personal Space Heater": "a compact white ceramic electric oscillating space heater with front grille vents",
    "Memory Foam Pillow": "a white contoured ergonomic memory foam sleeping pillow with breathable cover",
    "Adjustable Standing Desk": "a motorized electric height-adjustable sit-stand desk with white top and grey metal frame",
    # Sports
    "PowerTrack Smart Watch": "a sporty round face digital smartwatch with heart rate sensor and silicone rubber sport band",
    "ProGrip Yoga Mat": "a thick 6mm non-slip premium purple yoga exercise mat unrolled flat on floor",
    "Resistance Band Set (11pc)": "a set of colorful latex resistance exercise bands with different resistance levels",
    "Carbon Fiber Road Bike": "a lightweight carbon fiber drop-bar road racing bicycle in vivid red and white",
    "AdjustaDumbell 90lb Set": "adjustable dial-select dumbbell weight set with storage tray for home gym",
    "TrailBlazer Trekking Poles": "collapsible telescoping aluminum hiking trekking poles with cork grip and wrist straps",
    "SwimPro Goggles X7": "hydrodynamic low-profile competitive swimming goggles with dark tinted anti-fog lenses",
    "Pro Jump Rope (speed)": "a speed jump rope with thin steel wire cable and ergonomic lightweight handles",
    "Recovery Foam Roller Set": "a high-density cylindrical foam massage roller in black for muscle recovery",
    "GPS Sport Tracker Clip": "a small bright orange waterproof GPS outdoor sport activity tracker clip device",
    "Kettlebell Set (5-25lb)": "a set of black cast iron kettlebells on a rack arranged by weight for strength training",
    "Boxing Glove Pro Series": "professional red genuine leather boxing gloves with foam padding and velcro wrist strap",
    "Basketball Indoor/Outdoor": "a regulation size orange leather basketball with traditional black channel seam lines",
    "Tennis Racket Graphite": "a lightweight graphite composite tennis racket with string bed and cushion grip handle",
    "Cooling Pad for Laptops": "a laptop cooling pad stand with built-in USB-powered fans and adjustable height in black",
    # Gaming
    "4K 144Hz Gaming Monitor": "a curved ultrawide QHD gaming monitor with RGB LED ambient lighting strip on back",
    "Mechanical RGB Keyboard": "a full-size mechanical gaming keyboard with per-key RGB LED backlit keys",
    "Precision Gaming Mouse": "an ergonomic high-DPI optical gaming mouse with RGB logo light and programmable buttons",
    "Gaming Chair Pro": "a high-back racing-style ergonomic gaming chair in red and black with lumbar pillow",
    "Surround Sound Headset 7.1": "a large over-ear surround sound gaming headset with cushioned earcups and bendable boom microphone",
    "StreamDeck Plus": "a content creator streaming control pad with touch strip and programmable LCD button display",
    "Capture Card 4K USB-C": "a compact HDMI video game capture card with USB-C connectivity in sleek black",
    "Controller Charge Dock": "a dual controller charging dock station stand with LED charge indicator lights",
    "High Speed SSD M.2": "a compact M.2 2280 NVMe solid state drive chip on PCIe circuit board",
    "Portable SSD 2TB": "a slim pocket-sized external solid-state drive in brushed metallic grey with USB-C port",
    # Beauty
    "Vitamin C Glow Serum": "a small amber glass dropper bottle of brightening vitamin C facial serum with gold cap",
    "Hyaluronic Acid Moisturizer": "a white luxurious face moisturizer cream in a clean minimalist airless pump bottle",
    "Retinol Night Cream": "an elegant anti-aging retinol night cream in a dark blue glass jar with gold lid",
    "Natural SPF 50 Sunscreen": "a white mineral sunscreen lotion tube with clean botanical skincare branding",
    "Electric Face Massager": "a small pink rose quartz face roller massager gua sha lifting device",
    "Lash Lift & Tint Kit": "a beauty eyelash lift treatment professional kit with silicone pads and solution bottles",
    "Rose Clay Face Mask": "a pink rose clay face mask in a small jar with dried rose petals and natural ingredient label",
    "Collagen Eye Patches": "a pair of white hydrogel under-eye collagen face mask patches on white background",
    "Bamboo Charcoal Cleanser": "a black bamboo activated charcoal facial cleanser in a matte tube with natural branding",
    "Vitamin E Lip Oil": "a small clear lip gloss nourishing oil tube with vitamin E and doe foot applicator wand",
    "Micro Jade Roller Set": "a green jade face massage roller and matching gua sha stone tool set",
    "Hydrating Sheet Mask Box": "a box of individually wrapped Korean hydrating face sheet masks with fruit illustration",
    "Aromatherapy Eye Pillow": "a soft lavender filled fabric eye mask pillow for relaxation and meditation",
    "Probiotic Toner Mist": "a glass spray bottle facial toner mist with natural probiotic and botanical ingredients",
    "Detox Hair Scalp Mask": "a treatment tube of deep conditioning hair and scalp detox mask with clay",
    # Automotive
    "DashCam Pro 4K": "a compact black 4K car dash camera mounted on windshield with wide angle lens",
    "Car Phone Mount MagSafe": "a magnetic magsafe phone holder car air vent mount in black chrome finish",
    "Portable Jump Starter": "a black and yellow portable car battery jump starter pack with heavy duty clamp cables",
    "Seat Cover Leather Set": "custom fit black premium leather car seat covers with beige contrast stitching and piping",
    "Car Vacuum Cordless": "a handheld cordless mini car vacuum cleaner in black with motorized brush attachment",
    "LED Interior Light Strip": "flexible RGB color-changing LED strip lights installed inside car interior",
    "OBD2 Diagnostic Scanner": "a small plug-in OBD2 Bluetooth car diagnostic scanner tool in black",
    "Floor Mat All Weather Set": "custom all-weather heavy duty rubber floor mat set for car interior in black",
    "Tire Inflator Portable": "a compact portable digital auto tire inflator air compressor with pressure gauge",
    "Steering Wheel Cover": "a black genuine leather steering wheel cover with perforated grip and contrast stitching",
    "Smart Radar Detector": "a sleek windshield-mounted laser radar speed detector in gloss black",
    "Car Organizer Console": "a black leatherette center console car organizer box with lid and multiple compartments",
    "USB Car Charger 65W": "a compact multi-port USB-A and USB-C fast car charger adapter plugged into 12V socket",
    "Backup Camera System": "a wireless rear-view backup parking camera with monitor LCD display and night vision",
    "Carbon Fiber Wrap Film": "a roll of black carbon fiber textured vinyl wrap adhesive film for automotive detailing",
    # Music
    "Acoustic Guitar Yamaha F310": "a full-size dreadnought acoustic guitar in natural gloss wood finish with steel strings",
    "MIDI Controller 25-Key": "a compact 25-key mini MIDI keyboard controller with octave shift knobs in black",
    "Studio Monitor Speakers 5\"": "a pair of active near-field studio monitor reference speakers in black for audio mixing",
    "Condenser Microphone Kit": "a large-diaphragm condenser studio recording microphone on shock mount with pop filter screen",
    "Headphone Amp Pro": "a desktop headphone amplifier in brushed aluminum with large volume knob and 6.35mm jack",
    "Electric Guitar Fender Strat": "a classic Fender Stratocaster electric guitar in three-tone sunburst finish with maple neck",
    "DJ Controller Entry Level": "a DJ mixer controller with motorized jog wheels crossfader and performance pads in black",
    "Violin Bow & Rosin Set": "a carbon fiber violin bow with a small tin of rosin for string instrument playing",
    "Drum Practice Pad Kit": "an 8-piece drum practice pad training kit with drumsticks and stand",
    "Ukulele Soprano Natural": "a small compact soprano ukulele in light natural mahogany wood with aquila nylon strings",
    "Audio Interface USB 2-in": "a compact red USB audio recording interface with XLR mic inputs and headphone output",
    "Noise Cancelling Earphones": "in-ear noise cancelling wired earphones with interchangeable tips and inline remote microphone",
    "Portable Recorder Zoom H5": "a professional handheld portable digital audio recorder with interchangeable XLR capsule",
    "Music Stand Adjustable": "a lightweight folding metal music stand for sheet music with adjustable height and tray",
    "Pop Filter Screen": "a microphone pop filter with circular nylon mesh screen and flexible gooseneck clamp mount",
    # Health
    "Blood Pressure Monitor": "a digital automatic upper arm cuff blood pressure monitor with large LCD digital display",
    "Pulse Oximeter Clip": "a small white finger clip pulse oximeter device that reads blood oxygen SpO2 percentage",
    "Ultrasonic Humidifier": "a cool-mist ultrasonic humidifier in white with LED night light and whisper-quiet operation",
    "Acupressure Mat Set": "a purple foam acupressure therapy mat with plastic spiky massage points and neck pillow",
    "Posture Corrector Brace": "an adjustable black back posture corrector brace with cross-back shoulder strap design",
    "Sleep Sound Machine": "a compact white noise sleep machine with speaker grille and multiple sound setting buttons",
    "First Aid Kit Complete": "a red zipper first aid emergency kit bag with clearly labelled medical supplies inside",
    "Massage Gun Deep Tissue": "a handheld percussion massage gun in black with multiple interchangeable head attachments",
    "Meditation Cushion Zafu": "a round traditional zafu meditation floor cushion in dark navy blue buckwheat hull filling",
    "Sauna Blanket Infrared": "a portable far-infrared thermal sauna blanket in silver foil for home detox therapy",
    "Continuous Glucose Monitor": "a small white wearable continuous glucose monitoring sensor patch worn on the upper arm",
    "LED Light Therapy Mask": "a full-face LED phototherapy mask with multiple colored light modes for skincare treatment",
    "Foam Percussion Stick": "a foam massage roller stick tool for post-run leg muscle recovery and myofascial release",
    "Melatonin Sleep Diffuser": "a bedside plug-in sleep diffuser device with melatonin and lavender cartridge and nightlight",
    "Immunity Supplement Pack": "a white bottle of daily vitamin D3 zinc immune support supplement capsules",
}


def _build_product_embeddings():
    """Pre-compute CLIP text embeddings for all products.
    
    WHY TEXT INSTEAD OF IMAGES:
    - CLIP's text encoder maps to the SAME 512-dim space as its image encoder
    - Rich visual descriptions ("large cushioned over-ear headphones") semantically
      match real product photographs of AirPods Max, earbuds, headphones, etc.
    - PIL-drawn colored rectangles are in a completely different visual distribution
      from real product photography, causing low similarity scores
    """
    from app.ml.recommender import PRODUCTS

    model = _get_clip()
    if model is None:
        return None, None

    logger.info(f"Pre-computing CLIP text embeddings for {len(PRODUCTS)} products...")
    
    descriptions = []
    for prod in PRODUCTS:
        desc = PRODUCT_VISUAL_DESCRIPTIONS.get(
            prod["name"],
            f"a {prod['category'].lower()} product: {prod['name']}"
        )
        descriptions.append(desc)

    # Encode all text descriptions in one batch
    embeddings = model.encode(descriptions, batch_size=32, show_progress_bar=False)
    logger.info(f"CLIP text embeddings ready: {embeddings.shape}")
    
    # Upsert into Pinecone (invalidates old PIL-image vectors)
    pc_idx = _init_pinecone()
    if pc_idx:
        try:
            logger.info("Upserting text-based product embeddings to Pinecone...")
            vectors = []
            for i, prod in enumerate(PRODUCTS):
                vectors.append((
                    prod["id"],
                    embeddings[i].tolist(),
                    {"name": prod["name"], "category": prod["category"], "price": prod["price"]}
                ))
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
    Uses CLIP ViT-B/32 image encoder → cosine similarity against text-embedded catalog.
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
        except Exception:
            pass
            
    return {
        "clip_model_loaded": model is not None,
        "model_name": "clip-ViT-B-32",
        "embedding_strategy": "text-descriptions",
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
    """
    model = _get_clip()
    if model is None:
        return [{"error": "CLIP model not available."}]

    product_embeddings, products = get_product_embeddings()
    if product_embeddings is None or products is None:
        return [{"error": "Product embeddings not ready."}]

    try:
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
