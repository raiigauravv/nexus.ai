"use client";

import { Send, Bot, User, Sparkles, AlertTriangle, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
}

const BACKEND_URL = "http://localhost:8000/api/v1/chat";

export default function ChatUI() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: text };
    const allMessages = [...messages, userMsg];
    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: Message = { id: assistantId, role: "assistant", content: "" };

    setMessages([...allMessages, assistantMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: allMessages.map((m) => ({ role: m.role, content: m.content })),
          namespace: "default",
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (!data || data === "[DONE]") continue;
          try {
            const chunk = JSON.parse(data);
            if (chunk.type === "text-delta" && chunk.delta) {
              setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, content: m.content + chunk.delta } : m));
            } else if (chunk.type === "error") {
              setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, content: chunk.errorText, isError: true } : m));
            }
          } catch { /* ignore non-JSON lines */ }
        }
      }
    } catch (err) {
      setMessages((prev) => prev.map((m) =>
        m.id === assistantId
          ? { ...m, content: err instanceof Error ? err.message : "Connection failed. Please retry.", isError: true }
          : m
      ));
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  return (
    <div style={{
      display: "flex", flexDirection: "column", height: 600,
      background: "var(--bg-card)",
      border: "1px solid var(--border)",
      borderRadius: 14, overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        background: "var(--bg-elevated)",
        borderBottom: "1px solid var(--border)",
        padding: "14px 20px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Sparkles className="w-4 h-4" style={{ color: "#a78bfa" }} />
          <span style={{ fontWeight: 600, fontSize: 14, color: "var(--text-primary)" }}>NEXUS Agent</span>
        </div>
        <span style={{
          fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase",
          color: "#10b981", border: "1px solid rgba(16,185,129,0.3)",
          background: "rgba(16,185,129,0.1)", padding: "2px 8px", borderRadius: 99,
          display: "flex", alignItems: "center", gap: 4,
        }}>
          <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#10b981", display: "inline-block", boxShadow: "0 0 6px #10b981" }} />
          Online
        </span>
      </div>

      {/* Messages */}
      <div className="nx-scroll" style={{ flex: 1, overflowY: "auto", padding: "20px", display: "flex", flexDirection: "column", gap: 16 }}>
        {messages.length === 0 ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", textAlign: "center", gap: 12 }}>
            <div style={{
              width: 48, height: 48, borderRadius: 14,
              background: "rgba(124,58,237,0.15)", border: "1px solid rgba(139,92,246,0.3)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Bot className="w-6 h-6" style={{ color: "#a78bfa" }} />
            </div>
            <div>
              <p style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>NEXUS-AI RAG Agent</p>
              <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4, maxWidth: 240, lineHeight: 1.5 }}>
                Ask me anything about the documents in your knowledge base.
              </p>
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <div key={m.id} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
              <div style={{
                display: "flex", alignItems: "flex-end", gap: 8,
                flexDirection: m.role === "user" ? "row-reverse" : "row",
                maxWidth: "82%",
              }}>
                {/* Avatar */}
                <div style={{
                  width: 28, height: 28, borderRadius: 8, flexShrink: 0,
                  background: m.role === "user"
                    ? "linear-gradient(135deg,#7c3aed,#6d28d9)"
                    : "var(--bg-elevated)",
                  border: m.role === "user" ? "none" : "1px solid var(--border)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: m.role === "user" ? "0 0 12px rgba(124,58,237,0.3)" : "none",
                }}>
                  {m.role === "user"
                    ? <User className="w-3.5 h-3.5" style={{ color: "#e9d5ff" }} />
                    : <Bot className="w-3.5 h-3.5" style={{ color: "#a78bfa" }} />
                  }
                </div>

                {/* Bubble */}
                {m.content === "" && !m.isError ? (
                  /* Typing indicator */
                  <div style={{
                    display: "flex", alignItems: "center", gap: 5,
                    background: "var(--bg-elevated)", border: "1px solid var(--border)",
                    borderRadius: "14px 14px 14px 4px", padding: "12px 16px",
                  }}>
                    {[0, 0.15, 0.3].map((delay, i) => (
                      <div key={i} style={{
                        width: 7, height: 7, borderRadius: "50%", background: "#7c3aed",
                        animation: "pulse-dot 1.2s infinite", animationDelay: `${delay}s`,
                      }} />
                    ))}
                  </div>
                ) : m.isError ? (
                  <div style={{
                    display: "flex", alignItems: "flex-start", gap: 8,
                    background: "rgba(244,63,94,0.1)", border: "1px solid rgba(244,63,94,0.3)",
                    borderRadius: "14px 14px 14px 4px", padding: "10px 14px",
                    fontSize: 13, color: "#fca5a5", lineHeight: 1.5,
                  }}>
                    <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "#f43f5e" }} />
                    <span>{m.content}</span>
                  </div>
                ) : (
                  <div style={{
                    padding: "10px 14px",
                    borderRadius: m.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                    background: m.role === "user"
                      ? "linear-gradient(135deg,rgba(124,58,237,0.3),rgba(109,40,217,0.25))"
                      : "var(--bg-elevated)",
                    border: `1px solid ${m.role === "user" ? "rgba(139,92,246,0.4)" : "var(--border)"}`,
                    fontSize: 13, lineHeight: 1.65,
                    color: m.role === "user" ? "var(--text-primary)" : "var(--text-secondary)",
                    whiteSpace: "pre-wrap",
                  }}>
                    {m.content}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: "12px 16px",
        background: "var(--bg-elevated)",
        borderTop: "1px solid var(--border)",
      }}>
        <form onSubmit={sendMessage} style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your documents..."
            disabled={isLoading}
            style={{
              flex: 1, background: "var(--bg-card)",
              border: `1px solid ${isLoading ? "rgba(139,92,246,0.4)" : "var(--border)"}`,
              borderRadius: 10, color: "var(--text-primary)",
              fontSize: 13, padding: "9px 14px", outline: "none",
              fontFamily: "inherit",
              transition: "all 0.2s",
            }}
            onFocus={e => { (e.target as HTMLElement).style.borderColor = "rgba(139,92,246,0.5)"; (e.target as HTMLElement).style.boxShadow = "0 0 0 3px rgba(139,92,246,0.1)"; }}
            onBlur={e => { (e.target as HTMLElement).style.borderColor = "var(--border)"; (e.target as HTMLElement).style.boxShadow = "none"; }}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            style={{
              flexShrink: 0, width: 36, height: 36, borderRadius: 10,
              background: isLoading || !input.trim()
                ? "rgba(124,58,237,0.25)"
                : "linear-gradient(135deg,#7c3aed,#6d28d9)",
              border: "none", cursor: isLoading || !input.trim() ? "not-allowed" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: input.trim() && !isLoading ? "0 0 14px rgba(124,58,237,0.4)" : "none",
              transition: "all 0.2s",
            }}
          >
            {isLoading
              ? <RefreshCw className="w-4 h-4 animate-spin" style={{ color: "#a78bfa" }} />
              : <Send className="w-4 h-4" style={{ color: "#e9d5ff" }} />
            }
          </button>
        </form>
      </div>
    </div>
  );
}
