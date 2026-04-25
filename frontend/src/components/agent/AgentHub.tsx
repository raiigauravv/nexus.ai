"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  RefreshCw,
  Bot,
  User,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Sparkles,
  Brain,
  Trash2,
  AlertTriangle,
} from "lucide-react";

const BACKEND = "http://localhost:8000/api/v1";

interface ToolStep {
  tool: string;
  input: string;
  output: string;
  status: "running" | "done";
}

interface AgentMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolSteps: ToolStep[];
  status: "thinking" | "using-tools" | "writing" | "done" | "error";
  statusText: string;
}

const TOOL_META: Record<string, { icon: string; accent: string; label: string }> = {
  analyze_sentiment:             { icon: "💬", accent: "rgba(6,182,212,0.15)",   label: "Sentiment Analysis" },
  detect_fraud:                  { icon: "🛡️", accent: "rgba(244,63,94,0.15)",  label: "Fraud Detection" },
  get_recommendations:           { icon: "⭐", accent: "rgba(139,92,246,0.15)",  label: "Recommendations" },
  query_documents:               { icon: "📚", accent: "rgba(59,130,246,0.15)",  label: "Document Q&A (RAG)" },
  get_trending_products:         { icon: "🔥", accent: "rgba(249,115,22,0.15)",  label: "Trending Products" },
  smart_product_recommendations: { icon: "🧠", accent: "rgba(99,102,241,0.15)",  label: "Smart Recs" },
  explain_product_complaints:    { icon: "📊", accent: "rgba(244,63,94,0.12)",   label: "Product Complaints" },
  find_visually_similar_products:{ icon: "🖼️", accent: "rgba(139,92,246,0.15)", label: "Visual Search (CLIP)" },
};

const EXAMPLE_PROMPTS = [
  { label: "Recommendations", icon: "⭐", text: "What products would you recommend for user U003 and what's trending right now?" },
  { label: "Smart Recs (Cross-Module)", icon: "🧠", text: "Give me smart product recommendations for user U001 — consider their fraud risk level and how well-reviewed each product is." },
  { label: "Product Complaints", icon: "📊", text: "Which Electronics products should we stop recommending based on recent customer complaints?" },
  { label: "Visual Search", icon: "🖼️", text: "Find products that look like a wireless charging pad from our catalog." },
  { label: "Fraud Check", icon: "🛡️", text: "Is this transaction suspicious? Amount: $12,000, merchant: luxury goods, 3 transactions in the last hour, 800km from home." },
  { label: "Sentiment Analysis", icon: "💬", text: "Analyze the sentiment of this customer review: 'The product arrived late but the quality exceeded my expectations. Support team was responsive.'" },
];

function ToolCard({ step }: { step: ToolStep }) {
  const [expanded, setExpanded] = useState(false);
  const meta = TOOL_META[step.tool] || { icon: "🔧", accent: "rgba(139,92,246,0.1)", label: step.tool };

  return (
    <div style={{
      borderRadius: 10,
      border: "1px solid rgba(139,92,246,0.2)",
      background: meta.accent,
      overflow: "hidden",
      fontSize: 12,
    }}>
      <button
        onClick={() => setExpanded(!expanded)}
        style={{ width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", background: "transparent", border: "none", cursor: "pointer", color: "var(--text-secondary)" }}
      >
        <span style={{ fontSize: 14 }}>{meta.icon}</span>
        <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{meta.label}</span>
        {step.status === "running" ? (
          <RefreshCw className="w-3 h-3 animate-spin" style={{ marginLeft: 2, color: "#a78bfa" }} />
        ) : (
          <CheckCircle className="w-3 h-3" style={{ marginLeft: 2, color: "#10b981" }} />
        )}
        <span style={{ marginLeft: "auto", color: "var(--text-muted)" }}>
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </span>
      </button>

      {expanded && (
        <div style={{ borderTop: "1px solid rgba(139,92,246,0.15)", padding: "10px 12px", display: "flex", flexDirection: "column", gap: 8 }}>
          {step.input && (
            <div>
              <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 4 }}>Input</p>
              <p style={{ fontFamily: "monospace", background: "rgba(0,0,0,0.3)", borderRadius: 6, padding: "6px 8px", color: "#a78bfa", fontSize: 11, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
                {step.input.length > 200 ? step.input.slice(0, 200) + "…" : step.input}
              </p>
            </div>
          )}
          {step.output && step.status === "done" && (
            <div>
              <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 4 }}>Result</p>
              <p style={{ fontFamily: "monospace", background: "rgba(0,0,0,0.3)", borderRadius: 6, padding: "6px 8px", color: "var(--text-secondary)", fontSize: 11, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
                {step.output.length > 500 ? step.output.slice(0, 500) + "…" : step.output}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AssistantBubble({ msg }: { msg: AgentMessage }) {
  const isStreaming = msg.status !== "done" && msg.status !== "error";

  return (
    <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
      <div style={{
        width: 32, height: 32, borderRadius: 10,
        background: "linear-gradient(135deg, #7c3aed, #6d28d9)",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0, boxShadow: "0 0 16px rgba(124,58,237,0.4)",
      }}>
        <Brain className="w-4 h-4" style={{ color: "#e9d5ff" }} />
      </div>

      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
        {isStreaming && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#a78bfa", fontWeight: 500 }}>
            <RefreshCw className="w-3 h-3 animate-spin" />
            {msg.statusText}
          </div>
        )}

        {msg.toolSteps.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {msg.toolSteps.map((step, i) => <ToolCard key={i} step={step} />)}
          </div>
        )}

        {msg.content && (
          <div className="chat-msg-ai prose-dark" style={{ maxWidth: "none" }}>
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, fontSize: 14 }}>
              {msg.content}
              {isStreaming && msg.status === "writing" && (
                <span style={{ display: "inline-block", width: 2, height: 16, background: "#a78bfa", marginLeft: 2, animation: "pulse-dot 0.8s infinite", borderRadius: 1, verticalAlign: "middle" }} />
              )}
            </div>
          </div>
        )}

        {msg.status === "error" && (
          <div style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "12px 14px", background: "rgba(244,63,94,0.1)", border: "1px solid rgba(244,63,94,0.3)", borderRadius: 12, fontSize: 13, color: "#fca5a5" }}>
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "#f43f5e" }} />
            <span>{msg.content}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "flex-start", justifyContent: "flex-end" }}>
      <div className="chat-msg-user">{content}</div>
      <div style={{
        width: 32, height: 32, borderRadius: 10,
        background: "var(--bg-elevated)", border: "1px solid var(--border)",
        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
      }}>
        <User className="w-4 h-4" style={{ color: "var(--text-muted)" }} />
      </div>
    </div>
  );
}

export default function AgentHub() {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [input, setInput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || isRunning) return;
    setIsRunning(true);
    setInput("");

    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content || m.statusText }));

    const userMsg: AgentMessage = { id: Date.now().toString(), role: "user", content: text, toolSteps: [], status: "done", statusText: "" };
    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: AgentMessage = { id: assistantId, role: "assistant", content: "", toolSteps: [], status: "thinking", statusText: "🧠 NEXUS Agent is thinking..." };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    try {
      const resp = await fetch(`${BACKEND}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
      });

      if (!resp.body) throw new Error("No response stream");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try { handleSSEEvent(JSON.parse(line.slice(6)), assistantId); } catch {}
        }
      }
    } catch (e: unknown) {
      setMessages((prev) => prev.map((m) =>
        m.id === assistantId
          ? { ...m, status: "error", content: e instanceof Error ? e.message : "Connection failed", statusText: "" }
          : m
      ));
    } finally {
      setIsRunning(false);
      inputRef.current?.focus();
    }
  };

  const handleSSEEvent = (payload: Record<string, string>, assistantId: string) => {
    setMessages((prev) => prev.map((msg) => {
      if (msg.id !== assistantId) return msg;
      switch (payload.type) {
        case "status":      return { ...msg, statusText: payload.text, status: "thinking" };
        case "tool_start":  return { ...msg, status: "using-tools", statusText: `🔧 Running ${TOOL_META[payload.tool]?.label || payload.tool}...`, toolSteps: [...msg.toolSteps, { tool: payload.tool, input: payload.input || "", output: "", status: "running" }] };
        case "tool_result": return { ...msg, toolSteps: msg.toolSteps.map((s) => s.tool === payload.tool && s.status === "running" ? { ...s, output: payload.output, status: "done" as const } : s) };
        case "text_delta":  return { ...msg, status: "writing", statusText: "✍️ Writing response...", content: msg.content + payload.text };
        case "error":       return { ...msg, status: "error", content: payload.text, statusText: "" };
        case "done":        return { ...msg, status: "done", statusText: "" };
        default:            return msg;
      }
    }));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 260px)", minHeight: 520 }}>

      {/* Tool chips */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, paddingBottom: 14 }}>
        {Object.entries(TOOL_META).map(([key, meta]) => (
          <span key={key} style={{
            fontSize: 11, fontWeight: 500,
            padding: "3px 10px", borderRadius: 99,
            background: meta.accent,
            border: "1px solid rgba(139,92,246,0.2)",
            color: "var(--text-secondary)",
          }}>
            {meta.icon} {meta.label}
          </span>
        ))}
        {messages.length > 0 && (
          <button
            onClick={() => setMessages([])}
            style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-muted)", background: "transparent", border: "1px solid var(--border)", borderRadius: 99, padding: "3px 10px", cursor: "pointer" }}
          >
            <Trash2 className="w-3 h-3" /> Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="nx-scroll" style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 20, paddingRight: 4 }}>
        {messages.length === 0 ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 24 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{
                width: 64, height: 64, borderRadius: 20,
                background: "linear-gradient(135deg, rgba(124,58,237,0.3), rgba(109,40,217,0.2))",
                border: "1px solid rgba(139,92,246,0.4)",
                display: "flex", alignItems: "center", justifyContent: "center",
                margin: "0 auto 16px",
                boxShadow: "0 0 40px rgba(124,58,237,0.25)",
              }}>
                <Brain className="w-8 h-8" style={{ color: "#a78bfa" }} />
              </div>
              <h2 style={{ fontSize: 20, fontWeight: 700, color: "var(--text-primary)" }}>NEXUS Agent</h2>
              <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 6, maxWidth: 440, lineHeight: 1.6 }}>
                Ask anything. Routes automatically across all 8 ML tools — sentiment, fraud, recommendations, documents, trending products, and visual search.
              </p>
            </div>

            <div style={{ width: "100%", maxWidth: 640, display: "flex", flexDirection: "column", gap: 8 }}>
              <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--text-muted)", textAlign: "center", marginBottom: 4 }}>Try an example</p>
              {EXAMPLE_PROMPTS.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(ex.text)}
                  style={{
                    width: "100%", textAlign: "left",
                    padding: "12px 16px", borderRadius: 12,
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                    cursor: "pointer", transition: "all 0.15s",
                    display: "flex", alignItems: "flex-start", gap: 12,
                  }}
                  onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "rgba(139,92,246,0.4)"; (e.currentTarget as HTMLElement).style.background = "rgba(124,58,237,0.08)"; }}
                  onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--border)"; (e.currentTarget as HTMLElement).style.background = "var(--bg-elevated)"; }}
                >
                  <span style={{ fontSize: 18, flexShrink: 0 }}>{ex.icon}</span>
                  <div>
                    <p style={{ fontSize: 11, fontWeight: 700, color: "#a78bfa", marginBottom: 3 }}>{ex.label}</p>
                    <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>{ex.text}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) =>
            msg.role === "user"
              ? <UserBubble key={msg.id} content={msg.content} />
              : <AssistantBubble key={msg.id} msg={msg} />
          )
        )}
      </div>

      {/* Input */}
      <div style={{ paddingTop: 16, borderTop: "1px solid var(--border)", marginTop: 16 }}>
        <div style={{
          display: "flex", alignItems: "flex-end", gap: 12,
          background: "var(--bg-elevated)",
          border: `1px solid ${isRunning ? "rgba(139,92,246,0.5)" : "var(--border)"}`,
          borderRadius: 14, padding: "10px 14px",
          boxShadow: isRunning ? "0 0 20px rgba(124,58,237,0.15)" : "none",
          transition: "all 0.2s",
        }}>
          <Sparkles className="w-4 h-4 flex-shrink-0" style={{ color: "#7c3aed", marginBottom: 2 }} />
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything — I'll route to the right tool automatically…"
            rows={1}
            disabled={isRunning}
            style={{
              flex: 1, background: "transparent", border: "none", outline: "none",
              fontSize: 14, color: "var(--text-primary)", resize: "none",
              fontFamily: "inherit", lineHeight: 1.5, maxHeight: 120, overflowY: "auto",
              minHeight: 24,
            }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={isRunning || !input.trim()}
            style={{
              flexShrink: 0, width: 34, height: 34, borderRadius: 10,
              background: isRunning || !input.trim() ? "rgba(124,58,237,0.3)" : "linear-gradient(135deg,#7c3aed,#6d28d9)",
              border: "none", cursor: isRunning || !input.trim() ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: input.trim() && !isRunning ? "0 0 16px rgba(124,58,237,0.4)" : "none",
              transition: "all 0.2s",
            }}
          >
            {isRunning ? <RefreshCw className="w-4 h-4 animate-spin" style={{ color: "#a78bfa" }} /> : <Send className="w-4 h-4" style={{ color: "#e9d5ff" }} />}
          </button>
        </div>
        <p style={{ textAlign: "center", fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
          Press <kbd style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, padding: "0 5px", fontSize: 10 }}>Enter</kbd> to send ·{" "}
          <kbd style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 4, padding: "0 5px", fontSize: 10 }}>Shift+Enter</kbd> for newline
        </p>
      </div>
    </div>
  );
}
