"use client";

import { useState, useRef, useEffect } from "react";
import {
  Zap,
  Send,
  RefreshCw,
  Bot,
  User,
  ChevronDown,
  ChevronUp,
  Wrench,
  CheckCircle,
  Sparkles,
  Brain,
  MessageSquare,
} from "lucide-react";

const BACKEND = "http://localhost:8000/api/v1";

// ── Types ──────────────────────────────────────────────────────────────────────
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

// ── Tool metadata ──────────────────────────────────────────────────────────────
const TOOL_META: Record<string, { icon: string; color: string; label: string }> = {
  analyze_sentiment:             { icon: "💬", color: "bg-teal-50 border-teal-200 text-teal-700",   label: "Sentiment Analysis" },
  detect_fraud:                  { icon: "🛡️", color: "bg-red-50 border-red-200 text-red-700",     label: "Fraud Detection" },
  get_recommendations:           { icon: "⭐", color: "bg-violet-50 border-violet-200 text-violet-700", label: "Recommendations" },
  query_documents:               { icon: "📚", color: "bg-blue-50 border-blue-200 text-blue-700",  label: "Document Q&A (RAG)" },
  get_trending_products:         { icon: "🔥", color: "bg-orange-50 border-orange-200 text-orange-700", label: "Trending Products" },
  smart_product_recommendations: { icon: "🧠", color: "bg-indigo-50 border-indigo-200 text-indigo-700", label: "Smart Recs (Cross-Module)" },
  explain_product_complaints:    { icon: "📊", color: "bg-rose-50 border-rose-200 text-rose-700",  label: "Product Complaints" },
  find_visually_similar_products:{ icon: "🖼️", color: "bg-purple-50 border-purple-200 text-purple-700", label: "Visual Search (CLIP)" },
};

// Example prompts that showcase multi-tool capabilities
const EXAMPLE_PROMPTS = [
  {
    label: "Sentiment + Fraud",
    icon: "💬🛡️",
    text: "Analyze the sentiment of this review: 'Absolutely terrible service!' and check if a $4,800 ATM withdrawal at midnight is suspicious.",
  },
  {
    label: "Recommendations",
    icon: "⭐",
    text: "What products would you recommend for user U003 and what's trending right now?",
  },
  {
    label: "Smart Recs (Cross-Module)",
    icon: "🧠",
    text: "Give me smart product recommendations for user U001 — consider their fraud risk level and how well-reviewed each product is.",
  },
  {
    label: "Product Complaints",
    icon: "📊",
    text: "Which Electronics products should we stop recommending based on recent customer complaints?",
  },
  {
    label: "Visual Search",
    icon: "🖼️",
    text: "Find products that look like a wireless charging pad from our catalog.",
  },
  {
    label: "Fraud Check",
    icon: "🛡️",
    text: "Is this transaction suspicious? Amount: $12,000, merchant: luxury goods, 3 transactions in the last hour, 800km from home.",
  },
];

// ── Sub-components ─────────────────────────────────────────────────────────────
function ToolCard({ step }: { step: ToolStep }) {
  const [expanded, setExpanded] = useState(false);
  const meta = TOOL_META[step.tool] || { icon: "🔧", color: "bg-gray-50 border-gray-200 text-gray-700", label: step.tool };

  return (
    <div className={`rounded-xl border text-xs overflow-hidden transition-all ${meta.color}`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:opacity-80 transition-opacity"
      >
        <span className="text-base">{meta.icon}</span>
        <span className="font-semibold">{meta.label}</span>
        {step.status === "running" ? (
          <RefreshCw className="w-3 h-3 ml-1 animate-spin opacity-60" />
        ) : (
          <CheckCircle className="w-3 h-3 ml-1 opacity-60" />
        )}
        <span className="ml-auto opacity-50">
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-current/10 px-3 pb-3 pt-2 space-y-2">
          {step.input && (
            <div>
              <p className="font-semibold opacity-60 mb-0.5">Input</p>
              <p className="font-mono bg-white/50 rounded p-1.5 leading-relaxed whitespace-pre-wrap">
                {step.input.length > 200 ? step.input.slice(0, 200) + "…" : step.input}
              </p>
            </div>
          )}
          {step.output && step.status === "done" && (
            <div>
              <p className="font-semibold opacity-60 mb-0.5">Result</p>
              <p className="font-mono bg-white/50 rounded p-1.5 leading-relaxed whitespace-pre-wrap">
                {step.output.length > 400 ? step.output.slice(0, 400) + "…" : step.output}
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
    <div className="flex gap-3 items-start">
      {/* Brain Avatar */}
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center flex-shrink-0 shadow-sm mt-0.5">
        <Brain className="w-4 h-4 text-white" />
      </div>

      <div className="flex-1 space-y-2.5 max-w-none">
        {/* Status badge */}
        {isStreaming && (
          <div className="flex items-center gap-1.5 text-xs text-indigo-500 font-medium">
            <RefreshCw className="w-3 h-3 animate-spin" />
            {msg.statusText}
          </div>
        )}

        {/* Tool steps */}
        {msg.toolSteps.length > 0 && (
          <div className="space-y-1.5">
            {msg.toolSteps.map((step, i) => (
              <ToolCard key={i} step={step} />
            ))}
          </div>
        )}

        {/* Final answer */}
        {msg.content && (
          <div className="bg-white border border-gray-100 rounded-xl px-4 py-3 shadow-sm">
            <div className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
              {msg.content}
              {isStreaming && msg.status === "writing" && (
                <span className="inline-block w-1.5 h-4 bg-indigo-500 ml-0.5 animate-pulse rounded-sm align-middle" />
              )}
            </div>
          </div>
        )}

        {msg.status === "error" && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
            ⚠️ {msg.content}
          </div>
        )}
      </div>
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex gap-3 items-start justify-end">
      <div className="bg-indigo-600 text-white rounded-xl px-4 py-3 text-sm max-w-[75%] shadow-sm">
        {content}
      </div>
      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0 mt-0.5">
        <User className="w-4 h-4 text-gray-600" />
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
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

    // Build history for context (last 6 turns)
    const history = messages.slice(-6).map((m) => ({
      role: m.role,
      content: m.content || m.statusText,
    }));

    // Add user message
    const userMsg: AgentMessage = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      toolSteps: [],
      status: "done",
      statusText: "",
    };

    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: AgentMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      toolSteps: [],
      status: "thinking",
      statusText: "🧠 NEXUS Agent is thinking...",
    };

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
          try {
            const payload = JSON.parse(line.slice(6));
            handleSSEEvent(payload, assistantId);
          } catch {}
        }
      }
    } catch (e: unknown) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, status: "error", content: e instanceof Error ? e.message : "Connection failed", statusText: "" }
            : m
        )
      );
    } finally {
      setIsRunning(false);
      inputRef.current?.focus();
    }
  };

  const handleSSEEvent = (payload: Record<string, string>, assistantId: string) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== assistantId) return msg;

        switch (payload.type) {
          case "status":
            return { ...msg, statusText: payload.text, status: "thinking" };

          case "tool_start": {
            const newStep: ToolStep = {
              tool: payload.tool,
              input: payload.input || "",
              output: "",
              status: "running",
            };
            return {
              ...msg,
              status: "using-tools",
              statusText: `🔧 Running ${TOOL_META[payload.tool]?.label || payload.tool}...`,
              toolSteps: [...msg.toolSteps, newStep],
            };
          }

          case "tool_result": {
            const updatedSteps = msg.toolSteps.map((s) =>
              s.tool === payload.tool && s.status === "running"
                ? { ...s, output: payload.output, status: "done" as const }
                : s
            );
            return { ...msg, toolSteps: updatedSteps };
          }

          case "text_delta":
            return {
              ...msg,
              status: "writing",
              statusText: "✍️ Writing response...",
              content: msg.content + payload.text,
            };

          case "error":
            return { ...msg, status: "error", content: payload.text, statusText: "" };

          case "done":
            return { ...msg, status: "done", statusText: "" };

          default:
            return msg;
        }
      })
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setInput("");
  };

  return (
    <div className="flex flex-col h-[calc(100vh-220px)] min-h-[520px]">

      {/* Tool Legend Bar */}
      <div className="flex flex-wrap gap-2 pb-4">
        {Object.entries(TOOL_META).map(([key, meta]) => (
          <span key={key} className={`text-xs px-2.5 py-1 rounded-full border font-medium ${meta.color}`}>
            {meta.icon} {meta.label}
          </span>
        ))}
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="ml-auto text-xs text-gray-400 hover:text-gray-600 transition-colors px-2 py-1 rounded hover:bg-gray-100"
          >
            Clear chat
          </button>
        )}
      </div>

      {/* Message area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-6 pr-1 scroll-smooth">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 py-8">
            {/* Hero */}
            <div className="text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center mx-auto mb-4 shadow-md shadow-indigo-200">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-xl font-bold text-gray-800">NEXUS Agent</h2>
              <p className="text-sm text-gray-500 mt-1 max-w-md">
                Ask anything. I automatically route across all 8 NEXUS-AI tools —
                sentiment, fraud, recommendations, documents, trending, 
                <span className="text-indigo-500 font-medium"> smart cross-module recs, product complaints, and visual search</span>.
              </p>
            </div>

            {/* Example prompts */}
            <div className="w-full max-w-2xl space-y-2">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider text-center mb-3">
                Try an example
              </p>
              {EXAMPLE_PROMPTS.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => sendMessage(ex.text)}
                  className="w-full text-left p-3.5 rounded-xl bg-white border border-gray-100 hover:border-indigo-200 hover:bg-indigo-50/40 transition-all group shadow-sm"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xl">{ex.icon}</span>
                    <div>
                      <p className="text-xs font-semibold text-indigo-600 mb-0.5 group-hover:text-indigo-700">
                        {ex.label}
                      </p>
                      <p className="text-sm text-gray-600 leading-snug">{ex.text}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) =>
            msg.role === "user" ? (
              <UserBubble key={msg.id} content={msg.content} />
            ) : (
              <AssistantBubble key={msg.id} msg={msg} />
            )
          )
        )}
      </div>

      {/* Input bar */}
      <div className="pt-4 border-t border-gray-100 mt-4">
        <div className={`flex items-end gap-3 bg-white border rounded-2xl px-4 py-3 shadow-sm transition-all duration-200
          ${isRunning ? "border-indigo-200 bg-indigo-50/20" : "border-gray-200 hover:border-indigo-300 focus-within:border-indigo-400 focus-within:shadow-md focus-within:shadow-indigo-100/50"}`}
        >
          <Sparkles className="w-4 h-4 text-indigo-400 flex-shrink-0 mb-1" />
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything — I'll route to the right tool automatically…"
            rows={1}
            disabled={isRunning}
            className="flex-1 text-sm text-gray-800 bg-transparent resize-none outline-none placeholder-gray-400 max-h-32 overflow-y-auto leading-relaxed disabled:opacity-50"
            style={{ minHeight: "24px" }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={isRunning || !input.trim()}
            className="flex-shrink-0 w-8 h-8 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors shadow-sm"
          >
            {isRunning ? (
              <RefreshCw className="w-3.5 h-3.5 text-white animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5 text-white" />
            )}
          </button>
        </div>
        <p className="text-center text-xs text-gray-400 mt-2">
          Press <kbd className="bg-gray-100 rounded px-1 text-gray-500">Enter</kbd> to send ·{" "}
          <kbd className="bg-gray-100 rounded px-1 text-gray-500">Shift+Enter</kbd> for newline
        </p>
      </div>
    </div>
  );
}
