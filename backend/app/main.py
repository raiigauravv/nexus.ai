import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.endpoints import chat, ingest, fraud, recommend, sentiment, vision, agent, auth
from app.database import SessionLocal
from app.init_db import init_db
from app.kafka.consumer import start_consumer, stop_consumer
from app.kafka.producer import stop_producer

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("Initializing persistent database on startup...")
    db = SessionLocal()
    try:
        init_db(db)
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        raise
    finally:
        db.close()

    # Start Kafka consumer — non-fatal; app works without it
    try:
        await start_consumer()
    except Exception as e:
        logger.warning("Kafka consumer failed to start (non-fatal): %s", e)

    # Pre-warm sentiment model
    try:
        from app.ml.sentiment import analyze
        import asyncio
        logger.info("Pre-warming sentiment model (this may take a few seconds)...")
        await asyncio.to_thread(analyze, "warmup")
        logger.info("Sentiment model pre-warmed successfully.")
    except Exception as e:
        logger.warning("Could not pre-warm sentiment model: %s", e)

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    try:
        await stop_consumer()
    except Exception:
        pass
    try:
        await stop_producer()
    except Exception:
        pass

app = FastAPI(
    title="NEXUS-AI API",
    description="Backend services for NEXUS-AI Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration (Dev Mode: Allow typical Next.js alternative ports too)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:3001",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0", "message": "NEXUS-AI API is running"}

@app.get("/")
async def root():
    return {"message": "Welcome to NEXUS-AI API"}

# Include routers
app.include_router(chat.router, prefix=settings.API_V1_STR, tags=["chat"])
app.include_router(ingest.router, prefix=settings.API_V1_STR, tags=["ingestion"])
app.include_router(fraud.router, prefix=settings.API_V1_STR, tags=["fraud"])
app.include_router(recommend.router, prefix=settings.API_V1_STR, tags=["recommend"])
app.include_router(sentiment.router, prefix=settings.API_V1_STR, tags=["sentiment"])
app.include_router(vision.router, prefix=settings.API_V1_STR, tags=["vision"])
app.include_router(agent.router, prefix=settings.API_V1_STR, tags=["agent"])
app.include_router(auth.router, prefix=settings.API_V1_STR, tags=["auth"])
