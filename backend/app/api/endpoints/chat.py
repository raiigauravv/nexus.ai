import json
import asyncio
import uuid
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.rag.vectorstore import get_vectorstore
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Models confirmed available via ListModels API — primary is gemini-2.5-flash
RAG_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-001",
]


class ChatRequest(BaseModel):
    messages: List[dict]
    namespace: str = "default"


def get_llm(model: str = RAG_MODELS[0]):
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        streaming=True,
        google_api_key=settings.GEMINI_API_KEY,
        request_timeout=30,
    )


def extract_text(msg: dict) -> str:
    if "parts" in msg and isinstance(msg["parts"], list):
        text_parts = [p.get("text", "") for p in msg["parts"] if p.get("type") == "text"]
        if text_parts:
            return " ".join(text_parts).strip()
    return (msg.get("content") or msg.get("text") or "").strip()


def convert_messages(messages: List[dict]) -> List[BaseMessage]:
    converted = []
    for msg in messages:
        content = extract_text(msg)
        if not content:
            continue
        if msg.get("role") == "user":
            converted.append(HumanMessage(content=content))
        elif msg.get("role") == "assistant":
            converted.append(AIMessage(content=content))
    return converted


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _is_retryable(error: Exception) -> bool:
    """Return True for 503/429 (rate limit / high demand) errors."""
    msg = str(error).lower()
    return any(x in msg for x in ["503", "429", "resource_exhausted", "overloaded", "high demand", "quota"])


async def _stream_rag_with_retry(
    user_query: str,
    chat_history: List[BaseMessage],
    namespace: str,
):
    """
    Streams RAG response tokens. Retries with fallback models on 503/429.
    Yields raw token strings (not SSE-wrapped).
    """
    system_prompt = (
        "You are the NEXUS-AI intelligent assistant. "
        "Use the following retrieved context to answer the user's question accurately and concisely. "
        "If the answer is not in the context, say you don't know based on the loaded documents. "
        "Keep reasoning concise and professional.\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ])

    vectorstore = get_vectorstore(namespace=namespace)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    last_error = None
    for model_idx, model_name in enumerate(RAG_MODELS):
        try:
            if model_idx > 0:
                logger.info(f"RAG: retrying with fallback model {model_name}")

            llm = get_llm(model_name)
            rag_chain = (
                {
                    "context": (lambda x: x["input"]) | retriever | format_docs,
                    "chat_history": lambda x: x["chat_history"],
                    "input": lambda x: x["input"],
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            collected = []
            async for chunk in rag_chain.astream({
                "chat_history": chat_history,
                "input": user_query,
            }):
                if chunk:
                    collected.append(chunk)
                    yield chunk
            return  # success — stop retrying

        except Exception as e:
            last_error = e
            if _is_retryable(e) and model_idx < len(RAG_MODELS) - 1:
                logger.warning(f"RAG 503/429 on {model_name}, switching to {RAG_MODELS[model_idx + 1]}")
                await asyncio.sleep(1.5 * (model_idx + 1))  # backoff
                continue
            # Non-retryable or exhausted all models
            raise e

    if last_error:
        raise last_error


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Streaming RAG chat endpoint with automatic model fallback on 503/429.
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        last_msg = request.messages[-1]
        user_query = extract_text(last_msg) or "Hello"
        chat_history = convert_messages(request.messages[:-1])

        async def stream_generator():
            msg_id = str(uuid.uuid4())
            try:
                yield sse({"type": "text-start", "id": msg_id})
                async for token in _stream_rag_with_retry(user_query, chat_history, request.namespace):
                    yield sse({"type": "text-delta", "id": msg_id, "delta": token})
                yield sse({"type": "text-end", "id": msg_id})

            except Exception as e:
                logger.error(f"RAG stream error: {e}")
                # Surface a clean user-facing error
                if _is_retryable(e):
                    msg = "Gemini API is temporarily overloaded. Please wait a moment and try again."
                else:
                    msg = f"RAG error: {str(e)}"
                yield sse({"type": "error", "errorText": msg})

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-vercel-ai-ui-message-stream": "v1",
                "Access-Control-Allow-Origin": "*",
            },
        )

    except Exception as e:
        logger.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
