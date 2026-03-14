import json
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

class ChatRequest(BaseModel):
    messages: List[dict]
    namespace: str = "default"

def get_llm():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        streaming=True,
        google_api_key=settings.GEMINI_API_KEY
    )

def extract_text(msg: dict) -> str:
    """
    Supports both old and new Vercel AI SDK message formats:
    - New (v5/v6): {"role": "user", "parts": [{"type": "text", "text": "Hello"}]}
    - Old (v4):    {"role": "user", "content": "Hello"}
    """
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
    """Format a dict as a proper SSE event expected by AI SDK v6 DefaultChatTransport."""
    return f"data: {json.dumps(data)}\n\n"

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Streaming chat endpoint compatible with Vercel AI SDK v6 DefaultChatTransport.
    Emits Server-Sent Events (SSE) with uiMessageChunk JSON objects.
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided.")

        last_msg = request.messages[-1]
        user_query = extract_text(last_msg) or "Hello"
        chat_history = convert_messages(request.messages[:-1])

        vectorstore = get_vectorstore(namespace=request.namespace)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = get_llm()

        system_prompt = (
            "You are the NEXUS-AI intelligent assistant. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If you don't know the answer based on the context, say that you don't know. "
            "Keep your reasoning concise and professional.\n\n"
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("user", "{input}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": (lambda x: x["input"]) | retriever | format_docs,
                "chat_history": lambda x: x["chat_history"],
                "input": lambda x: x["input"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        async def stream_generator():
            # AI SDK v6 expects SSE with typed JSON chunks:
            # text-start → text-delta (repeated) → text-end
            msg_id = str(uuid.uuid4())
            try:
                # Signal start of assistant message
                yield sse({"type": "text-start", "id": msg_id})

                async for chunk in rag_chain.astream({
                    "chat_history": chat_history,
                    "input": user_query
                }):
                    if chunk:
                        yield sse({"type": "text-delta", "id": msg_id, "delta": chunk})

                # Signal end of assistant message
                yield sse({"type": "text-end", "id": msg_id})

            except Exception as e:
                import traceback
                with open("/tmp/chat_stream_error.log", "w") as f:
                    f.write(traceback.format_exc())
                logger.error(f"Streaming error: {e}")
                yield sse({"type": "error", "errorText": str(e)})

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-vercel-ai-ui-message-stream": "v1",
                "Access-Control-Allow-Origin": "*",
            }
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        import traceback
        with open("/tmp/chat_error.log", "w") as f:
            f.write(traceback.format_exc())
            try:
                f.write("\n\nPayload:\n")
                f.write(json.dumps(request.dict()))
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))
