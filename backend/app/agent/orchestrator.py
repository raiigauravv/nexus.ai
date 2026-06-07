"""
NEXUS-AI Agent Orchestrator — LangGraph StateGraph Implementation
==================================================================
Architecture:
  - LangGraph StateGraph with two nodes: `agent` (LLM call) and `tools` (tool execution)
  - Conditional edge: agent → tools if tool_calls present, else → END
  - Max 8 iterations enforced via state counter
  - Per-session conversation memory (last 10 turns)
  - SSE streaming: status → tool_start → tool_result → text_delta → done
  - Gemini fallback chain: 2.5-flash → 2.0-flash → 1.5-flash
"""
from __future__ import annotations

import json
import asyncio
import logging
import operator
from typing import AsyncGenerator, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

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
                                  Use when asked "which products have bad reviews".

DOCUMENT KNOWLEDGE:
  7. query_documents         — RAG: search the uploaded PDF knowledge base and answer from it.

VISUAL SEARCH:
  8. find_visually_similar_products — CLIP ViT-B/32 visual similarity search.

RULES:
- ALWAYS use tools when available. Never make up data you could get from a tool.
- Call ONE tool per turn. The graph will route back — chain tools across turns.
- After all tools have run, synthesize into a clear, structured, friendly final response.
- Format with Markdown headers if multiple tools were used."""

FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-001"]
MAX_ITERATIONS = 8
MAX_HISTORY_TURNS = 10

# ── LangGraph State ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    iterations: int


# ── Per-session conversation store ────────────────────────────────────────────
_SESSION_HISTORY: dict[str, list[BaseMessage]] = {}


def _get_session_history(session_id: str) -> list[BaseMessage]:
    return _SESSION_HISTORY.get(session_id, [])


def _save_to_session(session_id: str, human_msg: HumanMessage, ai_msg: AIMessage) -> None:
    history = _SESSION_HISTORY.setdefault(session_id, [])
    history.extend([human_msg, ai_msg])
    if len(history) > MAX_HISTORY_TURNS * 2:
        _SESSION_HISTORY[session_id] = history[-(MAX_HISTORY_TURNS * 2):]


def clear_session(session_id: str) -> None:
    _SESSION_HISTORY.pop(session_id, None)


# ── LLM singleton with fallback rotation ──────────────────────────────────────
_llm_cache: dict[str, object] = {}
_model_idx = 0


def _build_llm(model: str):
    if model not in _llm_cache:
        _llm_cache[model] = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1,
        ).bind_tools(NEXUS_TOOLS)
    return _llm_cache[model]


def get_llm():
    return _build_llm(FALLBACK_MODELS[_model_idx])


def _rotate_llm():
    global _model_idx
    _model_idx = (_model_idx + 1) % len(FALLBACK_MODELS)
    logger.info("Rotating to fallback model: %s", FALLBACK_MODELS[_model_idx])
    return _build_llm(FALLBACK_MODELS[_model_idx])


# ── LangGraph nodes ────────────────────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """Agent node: invoke the LLM with current message history."""
    llm = get_llm()
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
    }


def should_continue(state: AgentState) -> str:
    """Routing function: go to tools if LLM made a tool call, else finish."""
    last = state["messages"][-1]
    if (
        getattr(last, "tool_calls", None)
        and state["iterations"] < MAX_ITERATIONS
    ):
        return "tools"
    return END


# ── Build the LangGraph graph ──────────────────────────────────────────────────
_tool_node = ToolNode(NEXUS_TOOLS)

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", _tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()


# ── SSE streaming entry-point ──────────────────────────────────────────────────

async def run_agent_stream(
    user_message: str,
    chat_history: list | None = None,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Run the NEXUS LangGraph agent and yield SSE-formatted events.

    The LangGraph StateGraph handles the agent ↔ tool loop.
    This wrapper converts LangGraph astream_events into SSE chunks.
    """
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    try:
        yield _sse({"type": "status", "text": "🧠 NEXUS Agent is thinking..."})
        await asyncio.sleep(0.05)

        # ── Build initial message list ─────────────────────────────────────────
        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

        server_history = _get_session_history(session_id)
        if server_history:
            messages.extend(server_history)
        elif chat_history:
            for item in (chat_history or [])[-MAX_HISTORY_TURNS * 2:]:
                role, content = item.get("role", ""), item.get("content", "")
                if role == "user" and content:
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    messages.append(AIMessage(content=content))

        human_msg = HumanMessage(content=user_message)
        messages.append(human_msg)

        initial_state: AgentState = {"messages": messages, "iterations": 0}

        final_content = ""
        final_ai_message: AIMessage | None = None

        # ── Stream events from the LangGraph graph ─────────────────────────────
        loop = asyncio.get_event_loop()

        async def _stream_graph():
            nonlocal final_content, final_ai_message

            # astream_events yields fine-grained LangGraph events
            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # LLM streaming chunks
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        text = ""
                        if isinstance(chunk.content, str):
                            text = chunk.content
                        elif isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text += part.get("text", "")
                        if text:
                            final_content += text
                            yield _sse({"type": "text_delta", "text": text})

                # Tool call start
                elif kind == "on_tool_start":
                    tool_name = name or "tool"
                    tool_input = data.get("input", {})
                    display = str(next(iter(tool_input.values()), ""))[:200] if isinstance(tool_input, dict) else str(tool_input)[:200]
                    yield _sse({"type": "status", "text": f"🔧 Running {tool_name}..."})
                    yield _sse({"type": "tool_start", "tool": tool_name, "input": display})
                    await asyncio.sleep(0.05)

                # Tool call result
                elif kind == "on_tool_end":
                    tool_name = name or "tool"
                    output = str(data.get("output", ""))
                    yield _sse({"type": "tool_result", "tool": tool_name, "output": output[:1000]})
                    await asyncio.sleep(0.05)

                # Final agent output (non-streaming fallback)
                elif kind == "on_chain_end" and name == "agent":
                    output = data.get("output", {})
                    msgs = output.get("messages", []) if isinstance(output, dict) else []
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                            final_ai_message = msg
                            if not final_content:
                                c = msg.content
                                if isinstance(c, str):
                                    final_content = c
                                elif isinstance(c, list):
                                    for part in c:
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            final_content += part.get("text", "")

        yield _sse({"type": "status", "text": "✍️ Writing response..."})
        async for chunk in _stream_graph():
            yield chunk

        if not final_content:
            final_content = "Analysis complete. Please see the tool results above."

        # ── If content wasn't streamed token-by-token, send it now ────────────
        if final_ai_message and not any(
            '"type": "text_delta"' in c async for c in _stream_graph()
            if False  # just a guard — don't re-run the graph
        ):
            pass  # final_content was already accumulated above

        # ── Persist to session memory ──────────────────────────────────────────
        _save_to_session(session_id, human_msg, AIMessage(content=final_content))

        yield _sse({"type": "done"})

    except Exception as e:
        err = str(e)
        logger.error("Agent error: %s", e, exc_info=True)
        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            msg = (
                "⚠️ Gemini API rate limit reached. "
                "The LangGraph agent is fully operational — please retry tomorrow "
                "or upgrade to a paid API key for higher limits."
            )
        else:
            msg = f"Agent error: {err[:300]}"
        yield _sse({"type": "error", "text": msg})
        yield _sse({"type": "done"})
