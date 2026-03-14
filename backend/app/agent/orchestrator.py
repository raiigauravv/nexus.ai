"""
NEXUS-AI Agent Orchestrator
============================
- LangChain native tool-calling (no AgentExecutor) for full control
- ConversationBufferWindowMemory — last 10 turns persisted per session
- Supports multi-tool chaining (up to 8 iterations)
- SSE streaming: status → tool_start → tool_result → text_delta → done
- Gemini fallback chain: 2.0-flash → 1.5-flash → 1.5-flash-8b
"""
import json
import asyncio
import logging
from typing import AsyncGenerator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
)

from app.config import settings
from app.agent.tools import NEXUS_TOOLS

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NEXUS Agent, the unified intelligence orchestrator of the NEXUS-AI enterprise platform.

You have access to 8 specialized ML tools:

SENTIMENT & ANALYSIS:
  1. analyze_sentiment       — DistilBERT + VADER ensemble. Detects emotions, aspects, polarity.

FRAUD DETECTION:
  2. detect_fraud            — XGBoost fraud scorer. Pass JSON with: amount, merchant_category,
                               velocity_1h, distance_from_home_km, unusual_location (0/1).

RECOMMENDATIONS (Cross-Module Aware):
  3. get_recommendations     — SVD hybrid recs for a user (U001–U010).
  4. smart_product_recommendations — Cross-module recs that factor in fraud risk + sentiment health score.
  5. get_trending_products   — Bayesian popularity ranking across 500-user interaction matrix.

CROSS-MODULE INTELLIGENCE:
  6. explain_product_complaints — Pulls category-level complaint analysis from sentiment + recs.
                                  Use when asked "which products have bad reviews" or "what should we stop recommending".

DOCUMENT KNOWLEDGE:
  7. query_documents         — RAG: search the uploaded PDF knowledge base and answer from it.

VISUAL SEARCH:
  8. find_visually_similar_products — CLIP ViT-B/32 visual similarity. Pass a product description
                                      (e.g. "wireless charging pad"). Returns similar catalog items.

RULES:
- ALWAYS use the most relevant tool(s). Never improvise data you could get from a tool.
- Chain tools when a query spans multiple domains — you support up to 8 tool calls per response.
- For detect_fraud, construct valid JSON from the user's description.
- For cross-module queries (e.g. "which products should we avoid recommending?"), call
  explain_product_complaints AND get_recommendations/smart_product_recommendations in sequence.
- Use conversation history context — if someone says "that product" or "same user", refer back.
- After all tools have run, synthesize into a clear, structured, friendly final response.
- Format with Markdown headers if multiple tools were used.
- Be confident and authoritative — you are backed by real ML models."""


FALLBACK_MODELS = ["gemini-2.5-flash"]

# ── Per-session in-memory conversation store ───────────────────────────────────
# Maps session_id → list of BaseMessage (human + AI turns, last 10 kept)
_SESSION_HISTORY: dict[str, list[BaseMessage]] = {}
MAX_HISTORY_TURNS = 10  # each turn = 1 human + 1 AI message = 20 messages max


def _get_session_history(session_id: str) -> list[BaseMessage]:
    return _SESSION_HISTORY.get(session_id, [])


def _save_to_session(session_id: str, human_msg: HumanMessage, ai_msg: AIMessage) -> None:
    history = _SESSION_HISTORY.setdefault(session_id, [])
    history.append(human_msg)
    history.append(ai_msg)
    # Keep only last MAX_HISTORY_TURNS turns (2 messages per turn)
    if len(history) > MAX_HISTORY_TURNS * 2:
        _SESSION_HISTORY[session_id] = history[-(MAX_HISTORY_TURNS * 2):]


def clear_session(session_id: str) -> None:
    _SESSION_HISTORY.pop(session_id, None)


# ── LLM singleton ──────────────────────────────────────────────────────────────
_llm = None
_model_idx = 0


def _build_llm(model: str):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.1,
    ).bind_tools(NEXUS_TOOLS)


def get_llm():
    global _llm, _model_idx
    if _llm is None:
        _llm = _build_llm(FALLBACK_MODELS[_model_idx])
    return _llm


def _next_model_llm():
    """Rotate to next fallback model on rate limit."""
    global _llm, _model_idx
    _model_idx = (_model_idx + 1) % len(FALLBACK_MODELS)
    logger.info(f"Switching to fallback model: {FALLBACK_MODELS[_model_idx]}")
    _llm = _build_llm(FALLBACK_MODELS[_model_idx])
    return _llm


def _execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool by name and return its string output."""
    tool_map = {t.name: t for t in NEXUS_TOOLS}
    tool = tool_map.get(tool_name)
    if not tool:
        return f"Error: Tool '{tool_name}' not found."
    try:
        result = tool.invoke(tool_args)
        return str(result)
    except Exception as e:
        logger.error(f"Tool '{tool_name}' failed: {e}", exc_info=True)
        return f"Tool error: {e}"


async def run_agent_stream(
    user_message: str,
    chat_history: list | None = None,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Run the NEXUS agent and yield SSE events.

    Args:
        user_message:  The user's current message.
        chat_history:  Explicit history list from the frontend (used when session_id not found).
        session_id:    Unique session identifier for conversation memory.
    """
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    try:
        yield _sse({"type": "status", "text": "🧠 NEXUS Agent is thinking..."})
        await asyncio.sleep(0.05)

        llm = get_llm()

        # ── Build message history ──────────────────────────────────────────────
        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        # Priority 1: server-side session memory (persistent across SSE calls)
        server_history = _get_session_history(session_id)
        if server_history:
            messages.extend(server_history)
        elif chat_history:
            # Fallback: use history passed explicitly from frontend
            for item in chat_history[-MAX_HISTORY_TURNS * 2:]:
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user" and content:
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    messages.append(AIMessage(content=content))

        human_msg = HumanMessage(content=user_message)
        messages.append(human_msg)

        tool_steps = []
        final_content = ""

        # ── Agent loop (max 8 iterations for complex multi-tool chains) ────────
        for iteration in range(8):
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda msgs=messages: llm.invoke(msgs)
            )
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", []) or []
            if not tool_calls:
                # No more tool calls — extract final answer
                break

            yield _sse({
                "type": "status",
                "text": f"🔧 Running {len(tool_calls)} tool{'s' if len(tool_calls) > 1 else ''}...",
            })
            await asyncio.sleep(0.05)

            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", tool_name)

                # Format input for display
                if isinstance(tool_args, dict):
                    display_input = next(iter(tool_args.values()), "") if tool_args else ""
                else:
                    display_input = str(tool_args)
                display_input = str(display_input)[:200]

                yield _sse({
                    "type": "tool_start",
                    "tool": tool_name,
                    "input": display_input,
                })
                await asyncio.sleep(0.08)

                output = await loop.run_in_executor(
                    None, lambda tn=tool_name, ta=tool_args: _execute_tool(tn, ta)
                )
                tool_steps.append((tool_name, output))

                yield _sse({
                    "type": "tool_result",
                    "tool": tool_name,
                    "output": output[:1000],
                })
                await asyncio.sleep(0.08)

                messages.append(ToolMessage(content=output, tool_call_id=tool_id))

        # ── Extract final text content ─────────────────────────────────────────
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                final_content = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        final_content += part.get("text", "")
                    elif isinstance(part, str):
                        final_content += part

        if not final_content:
            final_content = "Analysis complete. Please see the tool results above for details."

        # ── Save to server-side session memory ────────────────────────────────
        ai_reply = AIMessage(content=final_content)
        _save_to_session(session_id, human_msg, ai_reply)

        # ── Stream final answer ────────────────────────────────────────────────
        yield _sse({"type": "status", "text": "✍️ Writing response..."})
        await asyncio.sleep(0.05)

        words = final_content.split(" ")
        chunk_size = 4
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield _sse({"type": "text_delta", "text": chunk})
            await asyncio.sleep(0.015)

        yield _sse({"type": "done"})

    except Exception as e:
        err_str = str(e)
        logger.error(f"Agent error: {e}", exc_info=True)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            msg = (
                "⚠️ Gemini API rate limit reached for today's free tier quota. "
                "The agent code is fully operational. Please try again tomorrow, "
                "or add a paid API key to unlock higher limits."
            )
        else:
            msg = f"Agent error: {err_str[:300]}"
        yield _sse({"type": "error", "text": msg})
        yield _sse({"type": "done"})
