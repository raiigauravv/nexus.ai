"""
NEXUS Agent API Endpoint
POST /api/v1/agent/chat              — chat with the unified multi-tool orchestrator (SSE)
GET  /api/v1/agent/tools             — list available tools
DELETE /api/v1/agent/session/{id}    — clear conversation memory for a session
"""
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from app.agent.tools import NEXUS_TOOLS
from app.agent.orchestrator import run_agent_stream, clear_session
from app.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)
router = APIRouter()
AGENT_CHAT_LIMIT_PER_MINUTE = 30


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class AgentMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class AgentChatRequest(BaseModel):
    message: str
    history: Optional[List[AgentMessage]] = []
    session_id: Optional[str] = "default"


@router.post("/agent/chat")
async def agent_chat(req: AgentChatRequest, request: Request):
    """
    Stream a conversation with the NEXUS Agent.
    Supports conversation memory via session_id — follow-up questions work.
    Returns SSE stream with status/tool_start/tool_result/text_delta/done events.
    """
    # Pass history as plain dicts (orchestrator handles LangChain conversion)
    allowed, retry_after = rate_limiter.allow(
        f"agent-chat:{_client_ip(request)}",
        limit=AGENT_CHAT_LIMIT_PER_MINUTE,
        window_seconds=60,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Too many agent chat requests. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    history_dicts = [{"role": m.role, "content": m.content} for m in (req.history or [])]

    async def generator():
        async for chunk in run_agent_stream(
            req.message,
            chat_history=history_dicts,
            session_id=req.session_id or "default",
        ):
            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
        },
    )


@router.delete("/agent/session/{session_id}")
async def clear_agent_session(session_id: str):
    """Clear conversation memory for a given session."""
    clear_session(session_id)
    return {"cleared": session_id}


@router.get("/agent/tools")
async def list_tools():
    """List all available agent tools with descriptions."""
    return {
        "tools": [
            {"name": t.name, "description": t.description[:250]}
            for t in NEXUS_TOOLS
        ],
        "count": len(NEXUS_TOOLS),
    }
