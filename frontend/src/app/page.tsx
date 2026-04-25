"use client";

import { useState } from "react";
import AgentHub from "@/components/agent/AgentHub";
import ChatUI from "@/components/chat/ChatUI";
import DocumentUpload from "@/components/chat/DocumentUpload";
import FraudDashboard from "@/components/fraud/FraudDashboard";
import RecommendationDashboard from "@/components/recommendation/RecommendationDashboard";
import SentimentDashboard from "@/components/sentiment/SentimentDashboard";
import VisionDashboard from "@/components/vision/VisionDashboard";
import { MessageSquare, ShieldAlert, Cpu, BarChart2, Eye, Brain, Zap } from "lucide-react";
import AuthMenu from "@/components/auth/AuthMenu";

const MODULES = [
  {
    id: "agent",
    name: "NEXUS Agent",
    icon: <Brain className="w-4 h-4" />,
    status: "live" as const,
    desc: "Unified AI orchestrator",
    color: "purple",
  },
  {
    id: "rag",
    name: "Document Q&A",
    icon: <MessageSquare className="w-4 h-4" />,
    status: "live" as const,
    desc: "RAG-powered document intelligence",
    color: "violet",
  },
  {
    id: "fraud",
    name: "Fraud Detection",
    icon: <ShieldAlert className="w-4 h-4" />,
    status: "live" as const,
    desc: "Real-time ML fraud scoring",
    color: "rose",
  },
  {
    id: "recommendation",
    name: "Recommendation",
    icon: <Cpu className="w-4 h-4" />,
    status: "live" as const,
    desc: "Personalized ML recommendations",
    color: "purple",
  },
  {
    id: "sentiment",
    name: "Sentiment Pipeline",
    icon: <BarChart2 className="w-4 h-4" />,
    status: "live" as const,
    desc: "NLP sentiment analysis at scale",
    color: "cyan",
  },
  {
    id: "vision",
    name: "Computer Vision",
    icon: <Eye className="w-4 h-4" />,
    status: "live" as const,
    desc: "CLIP visual product search",
    color: "amber",
  },
];

const MODULE_BANNERS: Record<string, { title: string; subtitle: string; gradient: string; icon: React.ReactNode }> = {
  agent: {
    title: "NEXUS Agent",
    subtitle: "The unified intelligence layer. Ask anything — the agent automatically routes to Sentiment, Fraud Detection, Recommendations, Document Q&A, Trending Products, and Visual Search.",
    gradient: "linear-gradient(135deg, rgba(109,40,217,0.6) 0%, rgba(124,58,237,0.4) 50%, rgba(6,182,212,0.2) 100%)",
    icon: <Brain className="w-7 h-7" />,
  },
  rag: {
    title: "Document Q&A",
    subtitle: "Upload PDF documents and ask questions. NEXUS retrieves the most relevant context from your knowledge base via Pinecone vector search and synthesizes grounded answers in real-time.",
    gradient: "linear-gradient(135deg, rgba(109,40,217,0.5) 0%, rgba(139,92,246,0.3) 100%)",
    icon: <MessageSquare className="w-7 h-7" />,
  },
  fraud: {
    title: "Real-Time Fraud Detection",
    subtitle: "HistGradientBoosting ensemble model scores transactions in real-time. F1: 0.909 · AUC-ROC: 0.990 · Precision: 0.984 — trained on 284k real transactions.",
    gradient: "linear-gradient(135deg, rgba(244,63,94,0.5) 0%, rgba(220,38,38,0.3) 100%)",
    icon: <ShieldAlert className="w-7 h-7" />,
  },
  recommendation: {
    title: "Recommendation Engine",
    subtitle: "Hybrid SVD collaborative filtering + content-based cosine similarity with real-time ALS embedding updates via Kafka. NDCG@10: 0.338.",
    gradient: "linear-gradient(135deg, rgba(124,58,237,0.5) 0%, rgba(109,40,217,0.3) 100%)",
    icon: <Cpu className="w-7 h-7" />,
  },
  sentiment: {
    title: "Sentiment Analysis Pipeline",
    subtitle: "Multi-layer NLP engine: VADER lexicon (65%) + TextBlob pattern analysis (35%). Detects overall sentiment, emotion distribution, aspect-level opinions, and subjectivity scoring.",
    gradient: "linear-gradient(135deg, rgba(6,182,212,0.5) 0%, rgba(8,145,178,0.3) 100%)",
    icon: <BarChart2 className="w-7 h-7" />,
  },
  vision: {
    title: "Computer Vision",
    subtitle: "CLIP ViT-B/32 visual similarity search across 150-product catalog. Upload an image to find visually similar products via 512-dim embedding space.",
    gradient: "linear-gradient(135deg, rgba(245,158,11,0.4) 0%, rgba(251,191,36,0.2) 100%)",
    icon: <Eye className="w-7 h-7" />,
  },
};

export default function Home() {
  const [activeModule, setActiveModule] = useState("agent");

  const banner = MODULE_BANNERS[activeModule];

  return (
    <main style={{ minHeight: "100vh", background: "var(--bg-base)", position: "relative", zIndex: 1 }}>

      {/* ── Navbar ──────────────────────────────────────────────────────── */}
      <nav style={{
        background: "rgba(5,5,15,0.85)",
        borderBottom: "1px solid rgba(139,92,246,0.2)",
        backdropFilter: "blur(20px)",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}>
        <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 60 }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 32, height: 32,
              background: "linear-gradient(135deg, #7c3aed, #6d28d9)",
              borderRadius: 8,
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 20px rgba(124,58,237,0.5)",
            }}>
              <Zap className="w-4 h-4" style={{ color: "#e9d5ff" }} />
            </div>
            <span style={{
              fontWeight: 800,
              fontSize: 18,
              letterSpacing: "-0.02em",
              background: "linear-gradient(135deg, #c4b5fd 0%, #a78bfa 50%, #7c3aed 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}>
              NEXUS-AI
            </span>
            <span style={{
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "#7c6dab",
              border: "1px solid rgba(139,92,246,0.25)",
              padding: "2px 8px",
              borderRadius: 99,
              background: "rgba(139,92,246,0.08)",
            }}>
              Enterprise
            </span>
          </div>

          {/* Auth Menu */}
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--text-muted)", paddingRight: 16, borderRight: "1px solid rgba(255,255,255,0.1)" }}>
              <span className="status-dot green" />
              <span className="hidden sm:inline">All systems operational</span>
            </div>
            <AuthMenu />
          </div>
        </div>
      </nav>

      {/* ── Body ────────────────────────────────────────────────────────── */}
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "28px 24px", display: "grid", gridTemplateColumns: "240px 1fr", gap: 24 }}>

        {/* ── Sidebar ─────────────────────────────────────────────────── */}
        <aside>
          <p className="section-label" style={{ paddingLeft: 4 }}>Platform Modules</p>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {MODULES.map((mod) => {
              const isActive = activeModule === mod.id;
              return (
                <button
                  key={mod.id}
                  onClick={() => setActiveModule(mod.id)}
                  className={`nav-item${isActive ? " active" : ""}`}
                  style={{ textAlign: "left" }}
                >
                  <span style={{ color: isActive ? "#a78bfa" : "var(--text-muted)", flexShrink: 0 }}>
                    {mod.icon}
                  </span>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: isActive ? "#c4b5fd" : "var(--text-secondary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {mod.name}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {mod.desc}
                    </div>
                  </div>
                  {isActive && (
                    <span style={{ marginLeft: "auto", width: 6, height: 6, borderRadius: "50%", background: "#a78bfa", boxShadow: "0 0 8px #a78bfa", flexShrink: 0 }} />
                  )}
                </button>
              );
            })}
          </div>

          {/* Version info */}
          <div style={{ marginTop: 32, padding: "14px", background: "rgba(139,92,246,0.06)", border: "1px solid rgba(139,92,246,0.15)", borderRadius: 10 }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 8 }}>System</div>
            {[
              { label: "Backend", val: "FastAPI v0.111" },
              { label: "Models", val: "3 active" },
              { label: "MLflow", val: "v2.14" },
              { label: "Kafka", val: "Connected" },
            ].map(({ label, val }) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                <span style={{ color: "var(--text-muted)" }}>{label}</span>
                <span style={{ color: "var(--text-secondary)", fontWeight: 500 }}>{val}</span>
              </div>
            ))}
          </div>
        </aside>

        {/* ── Main content ─────────────────────────────────────────────── */}
        <div className="fade-in-up" key={activeModule} style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Module banner */}
          <div style={{
            padding: "24px 28px",
            borderRadius: 16,
            background: banner.gradient,
            border: "1px solid rgba(139,92,246,0.25)",
            position: "relative",
            overflow: "hidden",
          }}>
            {/* Noise glow */}
            <div style={{
              position: "absolute", inset: 0,
              background: "radial-gradient(ellipse 80% 120% at 100% 50%, rgba(255,255,255,0.04) 0%, transparent 70%)",
              pointerEvents: "none",
            }} />
            <div style={{ position: "relative", display: "flex", alignItems: "flex-start", gap: 16 }}>
              <span style={{ color: "rgba(255,255,255,0.8)", flexShrink: 0, marginTop: 2 }}>{banner.icon}</span>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                  <h1 style={{ fontSize: 22, fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>{banner.title}</h1>
                  <span className="badge-live">Live</span>
                </div>
                <p style={{ fontSize: 13, color: "rgba(255,255,255,0.65)", lineHeight: 1.6, maxWidth: 680 }}>{banner.subtitle}</p>
              </div>
            </div>
          </div>

          {/* Module content */}
          {activeModule === "agent" && <AgentHub />}
          {activeModule === "rag" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 20 }}>
              <DocumentUpload />
              <ChatUI />
            </div>
          )}
          {activeModule === "fraud" && <FraudDashboard />}
          {activeModule === "recommendation" && <RecommendationDashboard />}
          {activeModule === "sentiment" && <SentimentDashboard />}
          {activeModule === "vision" && <VisionDashboard />}
        </div>
      </div>
    </main>
  );
}
