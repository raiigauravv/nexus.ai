"use client";

import { useState } from "react";
import AgentHub from "@/components/agent/AgentHub";
import ChatUI from "@/components/chat/ChatUI";
import DocumentUpload from "@/components/chat/DocumentUpload";
import FraudDashboard from "@/components/fraud/FraudDashboard";
import RecommendationDashboard from "@/components/recommendation/RecommendationDashboard";
import SentimentDashboard from "@/components/sentiment/SentimentDashboard";
import VisionDashboard from "@/components/vision/VisionDashboard";
import { MessageSquare, ShieldAlert, Cpu, BarChart2, Eye, Brain } from "lucide-react";

const MODULES = [
  {
    id: "agent",
    name: "NEXUS Agent",
    icon: <Brain className="w-4 h-4" />,
    status: "live" as const,
    desc: "Unified AI orchestrator",
  },
  {
    id: "rag",
    name: "Document Q&A",
    icon: <MessageSquare className="w-4 h-4" />,
    status: "live" as const,
    desc: "RAG-powered document intelligence",
  },
  {
    id: "fraud",
    name: "Fraud Detection",
    icon: <ShieldAlert className="w-4 h-4" />,
    status: "live" as const,
    desc: "Real-time ML fraud scoring",
  },
  {
    id: "recommendation",
    name: "Recommendation",
    icon: <Cpu className="w-4 h-4" />,
    status: "live" as const,
    desc: "Personalized ML recommendations",
  },
  {
    id: "sentiment",
    name: "Sentiment Pipeline",
    icon: <BarChart2 className="w-4 h-4" />,
    status: "live" as const,
    desc: "NLP sentiment analysis at scale",
  },
  {
    id: "vision",
    name: "Computer Vision",
    icon: <Eye className="w-4 h-4" />,
    status: "live" as const,
    desc: "Image classification & detection",
  },
];

export default function Home() {
  const [activeModule, setActiveModule] = useState("agent");

  return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      {/* Navbar */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <span className="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-violet-600">
                NEXUS-AI
              </span>
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600 border border-gray-200">
                Enterprise Platform
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              All systems operational
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">

          {/* Sidebar: Module List */}
          <div className="lg:col-span-1 space-y-3">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-1 mb-4">
              Platform Modules
            </h2>
            {MODULES.map((mod) => {
              const isActive = activeModule === mod.id;
              const isLive = mod.status === "live";
              return (
                <button
                  key={mod.id}
                  onClick={() => isLive && setActiveModule(mod.id)}
                  disabled={!isLive}
                  className={`w-full text-left rounded-xl border p-3.5 transition-all duration-200 ${
                    isActive
                      ? "bg-indigo-50 border-indigo-200 shadow-sm shadow-indigo-100"
                      : isLive
                      ? "bg-white border-gray-100 hover:border-indigo-100 hover:bg-indigo-50/30 cursor-pointer"
                      : "bg-gray-50 border-gray-100 opacity-60 cursor-not-allowed"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div
                      className={`flex items-center gap-2 font-medium text-sm ${isActive ? "text-indigo-700" : isLive ? "text-gray-700" : "text-gray-400"}`}
                    >
                      <span className={isActive ? "text-indigo-600" : "text-gray-400"}>
                        {mod.icon}
                      </span>
                      {mod.name}
                    </div>
                    {isLive ? (
                      <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500" />
                      </span>
                    ) : (
                      <span className="text-xs text-gray-400 bg-gray-100 rounded px-1.5 py-0.5">Soon</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 pl-6">{mod.desc}</p>
                </button>
              );
            })}
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">

            {/* RAG Module */}
            {activeModule === "rag" && (
              <>
                <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-xl p-6 text-white shadow-md">
                  <div className="flex items-center gap-3 mb-2">
                    <MessageSquare className="w-6 h-6 text-indigo-200" />
                    <h1 className="text-2xl font-bold">Document Q&A (RAG)</h1>
                  </div>
                  <p className="text-indigo-100 text-sm">
                    Upload PDF documents and ask questions. The NEXUS-AI agent retrieves
                    the most relevant context from your knowledge base and synthesizes
                    a grounded answer in real-time.
                  </p>
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                  <div className="lg:col-span-2">
                    <DocumentUpload />
                  </div>
                  <div className="lg:col-span-3">
                    <ChatUI />
                  </div>
                </div>
              </>
            )}

            {/* NEXUS Agent Module */}
            {activeModule === "agent" && (
              <>
                <div className="bg-gradient-to-r from-indigo-700 via-purple-700 to-violet-700 rounded-xl p-6 text-white shadow-md overflow-hidden relative">
                  <div className="absolute inset-0 opacity-10">
                    <div className="absolute top-0 right-0 w-64 h-64 rounded-full bg-white transform translate-x-16 -translate-y-16" />
                    <div className="absolute bottom-0 left-0 w-48 h-48 rounded-full bg-white transform -translate-x-16 translate-y-16" />
                  </div>
                  <div className="relative">
                    <div className="flex items-center gap-3 mb-2">
                      <Brain className="w-6 h-6 text-indigo-200" />
                      <h1 className="text-2xl font-bold">NEXUS Agent</h1>
                      <span className="text-xs bg-white/20 border border-white/30 px-2.5 py-0.5 rounded-full font-semibold text-indigo-100 ml-1">
                        Orchestrator
                      </span>
                    </div>
                    <p className="text-indigo-100 text-sm">
                      The unified intelligence layer. Ask anything in natural language — the agent
                      automatically routes to Sentiment, Fraud Detection, Recommendations, Document Q&A,
                      Trending Products, Smart Cross-Module Recs, Product Complaints, and Visual Search (CLIP).
                    </p>
                  </div>
                </div>
                <AgentHub />
              </>
            )}

            {/* Computer Vision Module */}
            {activeModule === "vision" && (
              <>
                <div className="bg-gradient-to-r from-rose-600 to-orange-500 rounded-xl p-6 text-white shadow-md">
                  <div className="flex items-center gap-3 mb-2">
                    <Eye className="w-6 h-6 text-rose-200" />
                    <h1 className="text-2xl font-bold">Computer Vision</h1>
                  </div>
                  <p className="text-rose-100 text-sm">
                    Upload any image for real-time scene classification, dominant color palette extraction,
                    edge detection, and image quality analysis — all powered by a custom Pillow + NumPy CV pipeline.
                  </p>
                </div>
                <VisionDashboard />
              </>
            )}

            {/* Sentiment Pipeline Module */}
            {activeModule === "sentiment" && (
              <>
                <div className="bg-gradient-to-r from-teal-600 to-cyan-600 rounded-xl p-6 text-white shadow-md">
                  <div className="flex items-center gap-3 mb-2">
                    <BarChart2 className="w-6 h-6 text-teal-200" />
                    <h1 className="text-2xl font-bold">Sentiment Analysis Pipeline</h1>
                  </div>
                  <p className="text-teal-100 text-sm">
                    Multi-layer NLP engine: VADER lexicon (65%) + TextBlob pattern analysis (35%).
                    Detects overall sentiment, emotion distribution, and aspect-level opinions.
                  </p>
                </div>
                <SentimentDashboard />
              </>
            )}

            {/* Recommendation Engine Module */}
            {activeModule === "recommendation" && (
              <>
                <div className="bg-gradient-to-r from-violet-600 to-purple-600 rounded-xl p-6 text-white shadow-md">
                  <div className="flex items-center gap-3 mb-2">
                    <Cpu className="w-6 h-6 text-violet-200" />
                    <h1 className="text-2xl font-bold">Recommendation Engine</h1>
                  </div>
                  <p className="text-violet-100 text-sm">
                    Hybrid SVD collaborative filtering + content-based cosine similarity. Personalized
                    recommendations for each user based on their interaction history and item features.
                  </p>
                </div>
                <RecommendationDashboard />
              </>
            )}

            {/* Fraud Detection Module */}
            {activeModule === "fraud" && (
              <>
                <div className="bg-gradient-to-r from-red-600 to-rose-600 rounded-xl p-6 text-white shadow-md">
                  <div className="flex items-center gap-3 mb-2">
                    <ShieldAlert className="w-6 h-6 text-red-200" />
                    <h1 className="text-2xl font-bold">Real-Time Fraud Detection</h1>
                  </div>
                  <p className="text-red-100 text-sm">
                    An ensemble of Isolation Forest + Gradient Boosting models analyzes
                    transactions in real-time, scoring each one for fraud probability
                    with explainable AI reasoning.
                  </p>
                </div>
                <FraudDashboard />
              </>
            )}

          </div>
        </div>
      </div>
    </main>
  );
}
