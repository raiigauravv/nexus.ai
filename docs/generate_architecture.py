"""
NEXUS-AI Architecture Diagram — Pillow renderer
Matches the original dark-blue premium aesthetic with gradient borders & glow.
Run: python3 docs/generate_architecture.py
"""
from PIL import Image, ImageDraw, ImageFont
import math, os, sys

W, H = 1600, 1080
OUT  = os.path.join(os.path.dirname(__file__), "architecture.png")

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = (8,  13,  28)
PANEL    = (11, 20,  45)
BORDER   = (30, 70, 160)
GLOW_BLU = (41, 121, 255)

# model accent colours
RED    = (239,  68,  68)
PURP   = (139,  92, 246)
GOLD   = (234, 179,   8)
GREEN  = ( 34, 197,  94)
TEAL   = ( 20, 184, 166)
BLUE   = ( 96, 165, 250)
ORANGE = (251, 146,  60)
PINK   = (236,  72, 153)

TEXT_H  = (224, 240, 255)
TEXT_M  = (148, 180, 220)
TEXT_D  = ( 90, 120, 155)

# ── Canvas ────────────────────────────────────────────────────────────────────
img  = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# ── Gradient background (top dark-blue to near-black at bottom) ───────────────
for y in range(H):
    t = y / H
    r = int(8  + (5  - 8 ) * t)
    g = int(13 + (8  - 13) * t)
    b = int(28 + (18 - 28) * t)
    draw.line([(0, y), (W, y)], fill=(r, g, b))

draw = ImageDraw.Draw(img)   # refresh after pixel ops

# ── Font helpers ──────────────────────────────────────────────────────────────
def _font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_TITLE  = _font(32, bold=True)
F_SECT   = _font(15, bold=True)
F_HEAD   = _font(14, bold=True)
F_SUB    = _font(12)
F_SMALL  = _font(11)
F_METRIC = _font(13, bold=True)
F_BADGE  = _font(13, bold=True)
F_LOGO   = _font(11, bold=True)

def text_w(s, font):
    bb = font.getbbox(s)
    return bb[2] - bb[0]

def cx_text(draw, cx, y, s, font, fill):
    x = cx - text_w(s, font) // 2
    draw.text((x, y), s, font=font, fill=fill)

# ── Drawing primitives ────────────────────────────────────────────────────────
def rounded_rect(draw, x0, y0, x1, y1, r=12, fill=None, outline=None, width=2):
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill,
                            outline=outline, width=width)

def glow_rect(x0, y0, x1, y1, colour, r=12, layers=4, fill=PANEL):
    """Draw a box with a multi-layer glow border."""
    for i in range(layers, 0, -1):
        alpha = int(40 + i * 18)
        cr, cg, cb = colour
        c = (cr, cg, cb)
        pad = i * 2
        try:
            draw.rounded_rectangle(
                [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
                radius=r + pad, outline=c, width=1, fill=None
            )
        except Exception:
            pass
    rounded_rect(draw, x0, y0, x1, y1, r=r, fill=fill, outline=colour, width=2)

def divider(x0, x1, y, colour=BORDER):
    draw.line([(x0, y), (x1, y)], fill=colour, width=1)

def dot(cx, cy, r, colour):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)

def hex_icon(cx, cy, size, colour):
    """Draw a simple hexagon icon."""
    pts = []
    for i in range(6):
        a = math.radians(60 * i - 30)
        pts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    draw.polygon(pts, outline=colour, fill=(*colour[:3], 40) if len(colour) == 4 else None)
    draw.polygon(pts, outline=colour)

def diamond_icon(cx, cy, size, colour):
    pts = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
    draw.polygon(pts, outline=colour)
    inner = int(size * 0.55)
    pts2 = [(cx, cy - inner), (cx + inner, cy), (cx, cy + inner), (cx - inner, cy)]
    draw.polygon(pts2, fill=colour)

def circle_icon(cx, cy, size, colour):
    draw.ellipse([cx - size, cy - size, cx + size, cy + size], outline=colour, width=2)
    inner = int(size * 0.5)
    draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=colour)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
cx = W // 2
cx_text(draw, cx, 18, "NEXUS-AI SYSTEM ARCHITECTURE", F_TITLE, TEXT_H)
cx_text(draw, cx, 60, "A Premium High-Fidelity AI Orchestration Platform", F_SUB, TEXT_M)
draw.line([(100, 88), (W - 100, 88)], fill=BORDER, width=1)

# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER  (y: 98 – 490)
# ══════════════════════════════════════════════════════════════════════════════
IL_X0, IL_Y0, IL_X1, IL_Y1 = 22, 98, W - 22, 490
glow_rect(IL_X0, IL_Y0, IL_X1, IL_Y1, GLOW_BLU, r=16, layers=3,
          fill=(11, 22, 52))
cx_text(draw, cx, IL_Y0 + 12, "NEXUS-AI: Intelligence Layer", F_SECT, BLUE)

# 4 model boxes
BOX_Y0 = IL_Y0 + 44
BOX_H  = IL_Y1 - BOX_Y0 - 14
BOX_W  = 362
GAP    = 20
starts = [IL_X0 + 14, IL_X0 + 14 + BOX_W + GAP,
          IL_X0 + 14 + (BOX_W + GAP) * 2,
          IL_X0 + 14 + (BOX_W + GAP) * 3]

MODELS = [
    {
        "title":   "Fraud Detection",
        "sub":     "XGBoost + Isolation Forest",
        "colour":  RED,
        "icon":    "hex",
        "lines": [
            ("SMOTE balanced  ·  284K transactions",  TEXT_D,  F_SMALL),
            ("F1: 0.9091",                            RED,     F_METRIC),
            ("AUC-ROC: 0.98",                         TEXT_M,  F_SMALL),
            ("Kaggle CC Fraud Dataset",               TEXT_D,  F_SMALL),
        ],
    },
    {
        "title":   "Hybrid Recommender",
        "sub":     "SVD + CLIP Content-Based",
        "colour":  PURP,
        "icon":    "diamond",
        "lines": [
            ("MovieLens 1M  ·  1M ratings",           TEXT_D,  F_SMALL),
            ("SVD rank-50  ·  NDCG@10",               PURP,    F_METRIC),
            ("Real-time ALS Kafka update",            TEXT_M,  F_SMALL),
            ("No full retrain needed",                TEXT_D,  F_SMALL),
        ],
    },
    {
        "title":   "Thematic Sentiment",
        "sub":     "RoBERTa + DistilBERT + VADER",
        "colour":  GOLD,
        "icon":    "circle",
        "lines": [
            ("BERTopic  ·  Aspect & emotion level",   TEXT_D,  F_SMALL),
            ("Accuracy: 91.3%",                       GOLD,    F_METRIC),
            ("cardiffnlp/roberta-sentiment",          TEXT_M,  F_SMALL),
            ("Ensemble fallback chain",               TEXT_D,  F_SMALL),
        ],
    },
    {
        "title":   "RAG Pipeline",
        "sub":     "LangChain + Pinecone",
        "colour":  GREEN,
        "icon":    "diamond",
        "lines": [
            ("Gemini embedding-001  ·  3072d",        TEXT_D,  F_SMALL),
            ("Faithfulness ≥ 0.90 (Ragas)",           GREEN,   F_METRIC),
            ("nexus-ai-rag  ·  Pinecone index",       TEXT_M,  F_SMALL),
            ("10-item eval dataset",                  TEXT_D,  F_SMALL),
        ],
    },
]

for m, bx in zip(MODELS, starts):
    glow_rect(bx, BOX_Y0, bx + BOX_W, BOX_Y0 + BOX_H,
              m["colour"], r=12, layers=3, fill=(14, 22, 48))

    # Title
    cx_text(draw, bx + BOX_W // 2, BOX_Y0 + 12, m["title"],  F_HEAD, m["colour"])
    cx_text(draw, bx + BOX_W // 2, BOX_Y0 + 32, m["sub"],    F_SMALL, TEXT_M)

    divider(bx + 14, bx + BOX_W - 14, BOX_Y0 + 52, m["colour"])

    # Icon
    icon_cy = BOX_Y0 + 52 + 62
    icon_cx = bx + BOX_W // 2
    if m["icon"] == "hex":
        hex_icon(icon_cx, icon_cy, 30, m["colour"])
    elif m["icon"] == "diamond":
        diamond_icon(icon_cx, icon_cy, 26, m["colour"])
    else:
        circle_icon(icon_cx, icon_cy, 26, m["colour"])

    divider(bx + 14, bx + BOX_W - 14, BOX_Y0 + 52 + 110, m["colour"])

    # Text lines
    ty = BOX_Y0 + 52 + 118
    for txt, col, fnt in m["lines"]:
        cx_text(draw, bx + BOX_W // 2, ty, txt, fnt, col)
        ty += 20

# ══════════════════════════════════════════════════════════════════════════════
# AI AGENT  (centre, y: 508 – 658)
# ══════════════════════════════════════════════════════════════════════════════
AG_W, AG_H = 460, 148
AG_X0 = (W - AG_W) // 2
AG_Y0 = 506

# Arrows from model boxes to agent
ARROW_COL = (41, 100, 200)
for bx in starts:
    ax = bx + BOX_W // 2
    draw.line([(ax, IL_Y1), (ax, AG_Y0 - 4)], fill=ARROW_COL, width=2)
    draw.polygon([(ax, AG_Y0 + 4), (ax - 7, AG_Y0 - 8), (ax + 7, AG_Y0 - 8)],
                 fill=ARROW_COL)

glow_rect(AG_X0, AG_Y0, AG_X0 + AG_W, AG_Y0 + AG_H,
          GLOW_BLU, r=18, layers=5, fill=(10, 24, 60))

# Glowing orb
OC_X, OC_Y = AG_X0 + AG_W // 2, AG_Y0 + 46
for r in range(28, 8, -4):
    alpha = max(0, 180 - r * 5)
    draw.ellipse([OC_X - r, OC_Y - r, OC_X + r, OC_Y + r],
                 fill=(20, 80, 200))
dot(OC_X, OC_Y, 14, (80, 160, 255))
dot(OC_X, OC_Y, 6,  (200, 230, 255))

cx_text(draw, AG_X0 + AG_W // 2, AG_Y0 + 6,  "AI AGENT",           F_HEAD, TEXT_H)
cx_text(draw, AG_X0 + AG_W // 2, AG_Y0 + 26, "LangGraph StateGraph",F_BADGE, BLUE)

# LangGraph badge
BDG_W, BDG_H = 240, 28
BDG_X = AG_X0 + (AG_W - BDG_W) // 2
BDG_Y = AG_Y0 + 76
rounded_rect(draw, BDG_X, BDG_Y, BDG_X + BDG_W, BDG_Y + BDG_H,
             r=8, fill=(15, 40, 100), outline=BLUE, width=1)
cx_text(draw, BDG_X + BDG_W // 2, BDG_Y + 6,
        ">>  LangGraph  +  LangChain", F_LOGO, BLUE)

cx_text(draw, AG_X0 + AG_W // 2, AG_Y0 + 112,
        "agent node  <->  tools node  ·  8 ML tools  ·  SSE streaming  ·  session memory",
        F_SMALL, TEXT_D)

# Arrow from agent down to infra
MID_X = W // 2
draw.line([(MID_X, AG_Y0 + AG_H), (MID_X, AG_Y0 + AG_H + 26)],
          fill=ARROW_COL, width=2)
draw.polygon([(MID_X, AG_Y0 + AG_H + 34),
              (MID_X - 7, AG_Y0 + AG_H + 20),
              (MID_X + 7, AG_Y0 + AG_H + 20)], fill=ARROW_COL)

# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE  (bottom-left)
# ══════════════════════════════════════════════════════════════════════════════
INF_Y0 = AG_Y0 + AG_H + 40
INF_Y1 = H - 22
INF_X0 = 22
INF_X1 = 850
glow_rect(INF_X0, INF_Y0, INF_X1, INF_Y1, BORDER, r=14, layers=2,
          fill=(8, 18, 38))
cx_text(draw, (INF_X0 + INF_X1) // 2, INF_Y0 + 10, "INFRASTRUCTURE",
        F_SECT, BLUE)

SERV = [
    ("FastAPI",          "API Gateway",             TEAL),
    ("PostgreSQL",       "User / App Data",          BLUE),
    ("Redis",            "Cache / Broker",           RED),
    ("MinIO",            "Object Storage",           ORANGE),
    ("Kafka +\nZookeeper","Event Streaming\nReal-time ALS", GOLD),
]
SBW = 148; SBH = 130; SBY = INF_Y0 + 38
sbx_start = INF_X0 + 22
sbx_gap   = (INF_X1 - INF_X0 - 44 - SBW * 5) // 4

for i, (name, sub, col) in enumerate(SERV):
    sbx = sbx_start + i * (SBW + sbx_gap)
    glow_rect(sbx, SBY, sbx + SBW, SBY + SBH, col, r=10, layers=2,
              fill=(10, 20, 42))
    if "\n" in name:
        parts = name.split("\n")
        cx_text(draw, sbx + SBW // 2, SBY + 10, parts[0], F_BADGE, col)
        cx_text(draw, sbx + SBW // 2, SBY + 26, parts[1], F_BADGE, col)
    else:
        cx_text(draw, sbx + SBW // 2, SBY + 14, name, F_BADGE, col)

    diamond_icon(sbx + SBW // 2, SBY + SBH // 2 + 8, 18, col)

    divider(sbx + 10, sbx + SBW - 10, SBY + SBH - 36, col)
    if "\n" in sub:
        parts = sub.split("\n")
        cx_text(draw, sbx + SBW // 2, SBY + SBH - 32, parts[0], F_SMALL, TEXT_D)
        cx_text(draw, sbx + SBW // 2, SBY + SBH - 16, parts[1], F_SMALL, TEXT_D)
    else:
        cx_text(draw, sbx + SBW // 2, SBY + SBH - 22, sub, F_SMALL, TEXT_D)

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY  (bottom-right)
# ══════════════════════════════════════════════════════════════════════════════
OBS_X0 = 870
OBS_X1 = W - 22
glow_rect(OBS_X0, INF_Y0, OBS_X1, INF_Y1, BORDER, r=14, layers=2,
          fill=(8, 18, 38))
cx_text(draw, (OBS_X0 + OBS_X1) // 2, INF_Y0 + 10,
        "OBSERVABILITY", F_SECT, BLUE)

OBS_ITEMS = [
    ("MLflow",         "Model Registry\nExperiment Tracking", PURP),
    ("Drift Monitor",  "PSI across 8 models\n/api/v1/monitor/drift", TEAL),
    ("Ragas Eval",     "RAG faithfulness\nhallucination tracking", GREEN),
]
OBW = (OBS_X1 - OBS_X0 - 44 - 2 * 14) // 3
obx_start = OBS_X0 + 22

for i, (name, sub, col) in enumerate(OBS_ITEMS):
    obx = obx_start + i * (OBW + 14)
    glow_rect(obx, SBY, obx + OBW, SBY + SBH, col, r=10, layers=2,
              fill=(10, 20, 42))
    cx_text(draw, obx + OBW // 2, SBY + 14, name, F_BADGE, col)
    circle_icon(obx + OBW // 2, SBY + SBH // 2 + 8, 18, col)
    divider(obx + 10, obx + OBW - 10, SBY + SBH - 36, col)
    parts = sub.split("\n")
    cx_text(draw, obx + OBW // 2, SBY + SBH - 32, parts[0], F_SMALL, TEXT_D)
    if len(parts) > 1:
        cx_text(draw, obx + OBW // 2, SBY + SBH - 16, parts[1], F_SMALL, TEXT_D)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND  (bottom-right corner)
# ══════════════════════════════════════════════════════════════════════════════
LX, LY = W - 220, INF_Y1 - 78
rounded_rect(draw, LX - 10, LY - 8, W - 10, INF_Y1 - 6,
             r=8, fill=(8, 16, 36), outline=BORDER, width=1)
for i, (col, lbl) in enumerate([(BLUE, "Data Flow / Control"),
                                  (PURP, "Orchestration"),
                                  (GOLD, "Kafka Event")]):
    yl = LY + i * 20
    draw.line([(LX, yl + 6), (LX + 28, yl + 6)], fill=col, width=2)
    draw.polygon([(LX + 32, yl + 6), (LX + 24, yl + 2), (LX + 24, yl + 10)],
                 fill=col)
    draw.text((LX + 38, yl), lbl, font=F_SMALL, fill=TEXT_M)

# ── Save ──────────────────────────────────────────────────────────────────────
img.save(OUT, "PNG", dpi=(144, 144))
print(f"✅  Saved  →  {OUT}  ({W}×{H}px)")
