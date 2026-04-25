"""
Kafka Consumer — NEXUS-AI Real-Time Embedding Updates (Option B)
=================================================================
Listens to nexus.purchase_events and performs a live ALS-style
single-user embedding update so the next recommendation call for
that user immediately reflects their new purchase.

ALS Update Math
---------------
Given the precomputed item factor matrix Vt (k × n_items) from SVD:
  user_factors = argmin ||R_u - user_factors @ Vt||^2
               = R_u @ Vt.T @ pinv(Vt @ Vt.T)      (closed-form ALS step)
  predicted[u] = clip(user_factors @ Vt, 0, 5)

This recomputes only ONE user's row — O(k²) instead of retraining SVD.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_PURCHASES = "nexus.purchase_events"
GROUP_ID        = "nexus-recommender-updater"

_consumer_task: asyncio.Task | None = None


# ── Real-Time ALS Embedding Update ────────────────────────────────────────────

def update_user_embedding(user_id: str, product_id: str, rating: float = 5.0) -> bool:
    """
    Perform an ALS-style single-user embedding update in-place.

    Returns True if the update succeeded, False if user/product not found.
    """
    try:
        from app.ml.recommender import get_recommender, USERS, PRODUCTS

        bundle = get_recommender()
        uid    = bundle["user_id_to_idx"].get(user_id)

        # Map product_id to matrix column index
        prod_idx = bundle.get("prod_id_to_idx", {}).get(product_id)
        if uid is None or prod_idx is None:
            logger.debug(f"ALS update skipped: user={user_id} prod={product_id} not in matrix")
            return False

        Vt = bundle.get("Vt")
        if Vt is None:
            logger.debug("Vt not in bundle — skipping ALS update")
            return False

        # 1. Update the raw rating
        bundle["large_matrix"][uid, prod_idx] = rating

        # 2. Closed-form ALS update for this user
        R_u = bundle["large_matrix"][uid].astype(np.float64)  # (n_items,)
        VtVt_inv = np.linalg.pinv(Vt @ Vt.T)                  # (k, k)
        user_factors = R_u @ Vt.T @ VtVt_inv                  # (k,)

        # 3. Update predicted ratings for this user only
        bundle["predicted_ratings"][uid] = np.clip(user_factors @ Vt, 0.0, 5.0)

        logger.info(
            f"✅  ALS update: user={user_id} purchased product={product_id} "
            f"(rating={rating}) → embeddings refreshed"
        )
        return True

    except Exception as e:
        logger.error(f"ALS embedding update failed: {e}", exc_info=True)
        return False


# ── Consumer Loop ──────────────────────────────────────────────────────────────

async def consume_purchase_events() -> None:
    """
    Background coroutine: consume nexus.purchase_events forever.
    Gracefully handles Kafka being unavailable (logs warning, exits).
    """
    try:
        from aiokafka import AIOKafkaConsumer  # type: ignore
    except ImportError:
        logger.warning("aiokafka not installed — Kafka consumer disabled.")
        return

    consumer = AIOKafkaConsumer(
        TOPIC_PURCHASES,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=1_000,
    )

    try:
        await consumer.start()
        logger.info(f"✅  Kafka consumer started → topic={TOPIC_PURCHASES}")
    except Exception as e:
        logger.warning(f"⚠️  Kafka consumer could not connect ({e}). Real-time updates disabled.")
        return

    try:
        async for msg in consumer:
            event = msg.value
            if not isinstance(event, dict):
                continue

            user_id    = event.get("user_id")
            product_id = event.get("product_id")
            rating     = float(event.get("rating", 5.0))

            if user_id and product_id:
                # Run the CPU-bound ALS update in a thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda u=user_id, p=product_id, r=rating: update_user_embedding(u, p, r),
                )
    except asyncio.CancelledError:
        logger.info("Kafka consumer cancelled — shutting down.")
    except Exception as e:
        logger.error(f"Kafka consumer error: {e}", exc_info=True)
    finally:
        try:
            await consumer.stop()
        except Exception:
            pass
        logger.info("Kafka consumer stopped.")


# ── Lifecycle Helpers ──────────────────────────────────────────────────────────

async def start_consumer() -> None:
    """Launch the consumer as a background asyncio task."""
    global _consumer_task
    _consumer_task = asyncio.create_task(consume_purchase_events())
    logger.info("Kafka consumer task created.")


async def stop_consumer() -> None:
    """Cancel and await the consumer task."""
    global _consumer_task
    if _consumer_task and not _consumer_task.done():
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            pass
    logger.info("Kafka consumer task stopped.")
