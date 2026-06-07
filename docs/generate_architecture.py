"""
Generate NEXUS-AI Architecture Diagram
Corrected version: LangGraph, XGBoost, Kafka, Drift Monitoring
Run: python3 docs/generate_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#0a0e1a")
ax.set_facecolor("#0a0e1a")

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "panel":       "#111827",
    "panel_border":"#1e3a5f",
    "intel_bg":    "#0f1f3d",
    "intel_border":"#1e4080",
    "fraud":       "#1a2744",
    "fraud_acc":   "#e53e3e",
    "rec":         "#1a2744",
    "rec_acc":     "#805ad5",
    "sent":        "#1a2744",
    "sent_acc":    "#d69e2e",
    "rag":         "#1a2744",
    "rag_acc":     "#38a169",
    "agent":       "#1e3a5f",
    "agent_acc":   "#4299e1",
    "infra":       "#12202a",
    "infra_border":"#1a4060",
    "obs":         "#12202a",
    "obs_border":  "#1a4060",
    "kafka":       "#1a2744",
    "kafka_acc":   "#f6ad55",
    "fastapi":     "#1a2744",
    "fastapi_acc": "#48bb78",
    "pg":          "#1a2744",
    "pg_acc":      "#63b3ed",
    "redis":       "#1a2744",
    "redis_acc":   "#fc8181",
    "minio":       "#1a2744",
    "minio_acc":   "#f6ad55",
    "mlflow":      "#1a2744",
    "mlflow_acc":  "#9f7aea",
    "drift":       "#1a2744",
    "drift_acc":   "#4fd1c5",
    "text_bright": "#f0f6ff",
    "text_mid":    "#a0b4cc",
    "text_dim":    "#607080",
    "arrow_data":  "#4299e1",
    "arrow_ctrl":  "#9f7aea",
    "arrow_event": "#f6ad55",
}


def box(x, y, w, h, fc, ec, radius=0.3, lw=1.5, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=3,
    )
    ax.add_patch(p)
    return p


def label(x, y, text, size=9, color="#f0f6ff", weight="normal", ha="center", va="center", zorder=5):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=zorder,
            fontfamily="DejaVu Sans")


def arrow(x0, y0, x1, y1, color, lw=1.5, style="->", zorder=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=zorder)


def section_label(x, y, text, size=8, color=None):
    ax.text(x, y, text, fontsize=size,
            color=color or C["text_dim"],
            ha="center", va="center", style="italic", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
label(10, 13.4, "NEXUS-AI SYSTEM ARCHITECTURE", size=18,
      color=C["text_bright"], weight="bold")
label(10, 13.0, "LangGraph · XGBoost · RAG · Kafka · MLflow · Drift Monitoring",
      size=9.5, color=C["text_mid"])

# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE LAYER (outer panel)
# ══════════════════════════════════════════════════════════════════════════════
box(0.4, 7.5, 19.2, 4.9, C["intel_bg"], C["intel_border"], radius=0.5, lw=2)
label(10, 12.05, "NEXUS-AI: Intelligence Layer", size=10,
      color=C["agent_acc"], weight="bold")

# ── 4 model boxes ─────────────────────────────────────────────────────────────
MODEL_Y   = 8.1
MODEL_H   = 3.7
MODEL_W   = 4.0
GAP       = 0.46
starts    = [0.7, 5.16, 9.62, 14.08]

# Fraud Detection
bx = starts[0]
box(bx, MODEL_Y, MODEL_W, MODEL_H, C["fraud"], C["fraud_acc"], lw=2)
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.45, "Fraud Detection", size=10,
      color="#fc8181", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.95, "XGBoost + Isolation Forest", size=8,
      color=C["text_mid"])
label(bx + MODEL_W/2, MODEL_Y + 2.35, "⬡", size=28, color="#e53e3e")
label(bx + MODEL_W/2, MODEL_Y + 1.6,  "SMOTE balanced", size=7.5, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 1.2,  "F1: 0.9091  AUC-ROC: 0.98", size=7.5,
      color="#fc8181", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + 0.65, "284K real transactions", size=7, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 0.25, "Kaggle CC Fraud Dataset", size=7, color=C["text_dim"])

# Hybrid Recommender
bx = starts[1]
box(bx, MODEL_Y, MODEL_W, MODEL_H, C["rec"], C["rec_acc"], lw=2)
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.45, "Hybrid Recommender", size=10,
      color="#b794f4", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.95, "SVD + CLIP Content-Based", size=8,
      color=C["text_mid"])
label(bx + MODEL_W/2, MODEL_Y + 2.35, "◈", size=26, color="#805ad5")
label(bx + MODEL_W/2, MODEL_Y + 1.6,  "MovieLens 1M  (1M ratings)", size=7.5, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 1.2,  "SVD rank-50  ·  NDCG@10", size=7.5,
      color="#b794f4", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + 0.65, "Real-time ALS Kafka update", size=7, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 0.25, "No full retrain needed", size=7, color=C["text_dim"])

# Thematic Sentiment
bx = starts[2]
box(bx, MODEL_Y, MODEL_W, MODEL_H, C["sent"], C["sent_acc"], lw=2)
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.45, "Thematic Sentiment", size=10,
      color="#f6e05e", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.95, "RoBERTa + DistilBERT + VADER", size=7.5,
      color=C["text_mid"])
label(bx + MODEL_W/2, MODEL_Y + 2.35, "◉", size=26, color="#d69e2e")
label(bx + MODEL_W/2, MODEL_Y + 1.6,  "BERTopic thematic clustering", size=7.5, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 1.2,  "Accuracy: 91.3%", size=7.5,
      color="#f6e05e", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + 0.65, "Aspect + emotion breakdown", size=7, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 0.25, "cardiffnlp/roberta-sentiment", size=7, color=C["text_dim"])

# RAG
bx = starts[3]
box(bx, MODEL_Y, MODEL_W, MODEL_H, C["rag"], C["rag_acc"], lw=2)
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.45, "RAG Pipeline", size=10,
      color="#68d391", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + MODEL_H - 0.95, "LangChain + Pinecone", size=8,
      color=C["text_mid"])
label(bx + MODEL_W/2, MODEL_Y + 2.35, "◈", size=26, color="#38a169")
label(bx + MODEL_W/2, MODEL_Y + 1.6,  "Gemini embedding-001 (3072d)", size=7.5, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 1.2,  "Faithfulness ≥ 0.90 (Ragas)", size=7.5,
      color="#68d391", weight="bold")
label(bx + MODEL_W/2, MODEL_Y + 0.65, "nexus-ai-rag Pinecone index", size=7, color=C["text_dim"])
label(bx + MODEL_W/2, MODEL_Y + 0.25, "Ragas eval · 10-item dataset", size=7, color=C["text_dim"])

# ══════════════════════════════════════════════════════════════════════════════
# AI AGENT (centre)
# ══════════════════════════════════════════════════════════════════════════════
AX, AY, AW, AH = 7.2, 4.5, 5.6, 2.7
box(AX, AY, AW, AH, C["agent"], C["agent_acc"], radius=0.5, lw=2.5)

# Glowing circle
circle = plt.Circle((AX + AW/2, AY + AH/2 + 0.2), 0.75,
                     color="#1a4080", zorder=4)
ax.add_patch(circle)
circle2 = plt.Circle((AX + AW/2, AY + AH/2 + 0.2), 0.6,
                      color="#2a5faa", zorder=4)
ax.add_patch(circle2)
label(AX + AW/2, AY + AH/2 + 0.2, "⬡", size=20, color="#90cdf4", zorder=5)

label(AX + AW/2, AY + AH - 0.38, "AI AGENT", size=12,
      color=C["text_bright"], weight="bold")
label(AX + AW/2, AY + AH - 0.72, "LangGraph StateGraph", size=9.5,
      color="#90cdf4", weight="bold")
label(AX + AW/2, AY + 0.55, "agent node  ↔  tools node", size=8, color=C["text_mid"])
label(AX + AW/2, AY + 0.22, "8 ML tools · SSE streaming · session memory", size=7.5,
      color=C["text_dim"])

# LangGraph badge
badge_x, badge_y = AX + AW/2 - 1.5, AY + 1.1
box(badge_x, badge_y, 3.0, 0.5, "#1a3a6a", "#4299e1", radius=0.2, lw=1.5)
label(badge_x + 1.5, badge_y + 0.25, ">>  LangGraph + LangChain", size=8,
      color="#90cdf4", weight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE (bottom-left)
# ══════════════════════════════════════════════════════════════════════════════
IX, IY, IW, IH = 0.4, 0.3, 11.8, 3.9
box(IX, IY, IW, IH, C["infra"], C["infra_border"], radius=0.4, lw=2)
label(IX + IW/2, IY + IH - 0.3, "INFRASTRUCTURE", size=10,
      color=C["agent_acc"], weight="bold")

# 5 infra service boxes
SW, SH = 1.9, 2.4
SY = IY + 0.35
sx_list = [0.6, 2.65, 4.7, 6.75, 8.8]
services = [
    ("FastAPI",      "API Gateway",        "#48bb78", "⬡"),
    ("PostgreSQL",   "User / App Data",    "#63b3ed", "◈"),
    ("Redis",        "Cache / Broker",     "#fc8181", "◈"),
    ("MinIO",        "Object Storage\nModel Artifacts", "#f6ad55", "◈"),
    ("Kafka +\nZookeeper", "Event Streaming\nReal-time ALS", "#f6ad55", "◉"),
]
for (name, sub, acc, ico), sx in zip(services, sx_list):
    box(sx, SY, SW, SH, C["panel"], acc, radius=0.3, lw=1.5)
    label(sx + SW/2, SY + SH - 0.32, name, size=8.5, color=acc, weight="bold")
    label(sx + SW/2, SY + SH/2 + 0.1, ico, size=22, color=acc)
    label(sx + SW/2, SY + 0.55, sub, size=7, color=C["text_dim"])

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY (bottom-right)
# ══════════════════════════════════════════════════════════════════════════════
OX, OY, OW, OH = 12.6, 0.3, 7.0, 3.9
box(OX, OY, OW, OH, C["obs"], C["obs_border"], radius=0.4, lw=2)
label(OX + OW/2, OY + OH - 0.3, "OBSERVABILITY", size=10,
      color=C["agent_acc"], weight="bold")

obs = [
    ("MLflow",         "Model Registry\nExperiment Tracking", "#9f7aea", "◈"),
    ("Drift Monitor",  "PSI across 8 models\n/api/v1/monitor/drift", "#4fd1c5", "◉"),
    ("Ragas Eval",     "RAG faithfulness\nhallucination tracking", "#68d391", "◈"),
]
ox_list = [12.8, 14.9, 17.0]
for (name, sub, acc, ico), ox in zip(obs, ox_list):
    box(ox, SY, SW, SH, C["panel"], acc, radius=0.3, lw=1.5)
    label(ox + SW/2, SY + SH - 0.32, name, size=8.5, color=acc, weight="bold")
    label(ox + SW/2, SY + SH/2 + 0.1, ico, size=22, color=acc)
    label(ox + SW/2, SY + 0.55, sub, size=7, color=C["text_dim"])

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════════════════════
# Model boxes → Agent (downward)
for bx, mid in zip(starts, [2.7, 7.16, 11.62, 16.08]):
    arrow(mid, MODEL_Y, mid, AY + AH, C["arrow_data"], lw=1.5)

# Agent → Infrastructure
arrow(AX + AW/2, AY, AX + AW/2, IY + IH, C["arrow_ctrl"], lw=1.8)

# Kafka → Agent (purchase event feedback)
arrow(9.75, IY + IH, 9.6, AY, C["arrow_event"], lw=1.5)

# Agent → Observability
arrow(AX + AW, AY + AH/2, OX, AY + AH/2 - 0.5, C["arrow_ctrl"], lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    Line2D([0], [0], color=C["arrow_data"],  lw=2, label="Data Flow"),
    Line2D([0], [0], color=C["arrow_ctrl"],  lw=2, label="Control / Orchestration"),
    Line2D([0], [0], color=C["arrow_event"], lw=2, label="Kafka Event"),
]
leg = ax.legend(handles=legend_items, loc="lower right",
                framealpha=0.15, facecolor="#0d1117",
                edgecolor="#1e3a5f", fontsize=8,
                labelcolor=C["text_mid"])

plt.tight_layout(pad=0.3)
out = "docs/architecture.png"
fig.savefig(out, dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"✅  Saved → {out}")
