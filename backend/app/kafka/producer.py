"""
Kafka Producer — NEXUS-AI Event Streaming
==========================================
Publishes structured events to two Kafka topics:

  nexus.purchase_events  — fired after every recommendation purchase recorded
  nexus.fraud_alerts     — fired when fraud_score > 0.70

The consumer (consumer.py) listens to nexus.purchase_events and
performs a real-time ALS single-user embedding update (Option B).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_PURCHASES = "nexus.purchase_events"
TOPIC_FRAUD     = "nexus.fraud_alerts"

_producer = None
_producer_lock = asyncio.Lock()


async def get_producer():
    """Lazily initialise the AIOKafka producer (singleton)."""
    global _producer
    if _producer is not None:
        return _producer

    async with _producer_lock:
        if _producer is not None:
            return _producer
        try:
            from aiokafka import AIOKafkaProducer  # type: ignore
            p = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                compression_type="gzip",
                request_timeout_ms=5_000,
            )
            await p.start()
            _producer = p
            logger.info(f"✅  Kafka producer connected → {KAFKA_BOOTSTRAP}")
        except Exception as e:
            logger.warning(f"⚠️  Kafka producer unavailable ({e}). Events will be skipped.")
            _producer = None
    return _producer


async def stop_producer() -> None:
    """Gracefully stop the producer (called on app shutdown)."""
    global _producer
    if _producer is not None:
        try:
            await _producer.stop()
            logger.info("Kafka producer stopped.")
        except Exception:
            pass
        _producer = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def publish_purchase_event(
    user_id: str,
    product_id: str,
    product_name: str,
    category: str,
    price: float,
    recommendation_score: float | None = None,
) -> None:
    """
    Publish a purchase event to nexus.purchase_events.

    The Kafka consumer reads this and triggers a real-time ALS embedding
    update for the user so the next recommendation reflects the purchase.
    """
    event: dict[str, Any] = {
        "event_type":           "purchase",
        "user_id":              user_id,
        "product_id":           product_id,
        "product_name":         product_name,
        "category":             category,
        "price":                price,
        "recommendation_score": recommendation_score,
        "rating":               5.0,   # purchase = maximum positive signal
        "timestamp":            _now_iso(),
    }
    await _send(TOPIC_PURCHASES, event, key=user_id)


async def publish_fraud_alert(
    transaction_id: str,
    cardholder_id: str | None,
    amount: float,
    fraud_score: float,
    risk_level: str,
    merchant_category: str,
    reasons: list[str],
) -> None:
    """
    Publish a high-confidence fraud alert to nexus.fraud_alerts.
    Only fired when fraud_score > 0.70.
    """
    event: dict[str, Any] = {
        "event_type":        "fraud_alert",
        "transaction_id":    transaction_id,
        "cardholder_id":     cardholder_id,
        "amount":            amount,
        "fraud_score":       round(fraud_score, 4),
        "risk_level":        risk_level,
        "merchant_category": merchant_category,
        "reasons":           reasons,
        "timestamp":         _now_iso(),
    }
    await _send(TOPIC_FRAUD, event, key=transaction_id)


async def _send(topic: str, event: dict, key: str | None = None) -> None:
    """Internal send — swallows errors so Kafka never blocks the API response."""
    producer = await get_producer()
    if producer is None:
        logger.debug(f"Kafka unavailable — skipping event to {topic}")
        return
    try:
        key_bytes = key.encode("utf-8") if key else None
        await producer.send_and_wait(topic, value=event, key=key_bytes)
        logger.debug(f"→ Kafka [{topic}] {event.get('event_type')} key={key}")
    except Exception as e:
        logger.warning(f"Kafka send failed [{topic}]: {e}")
