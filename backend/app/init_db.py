import logging
import random
import numpy as np
from sqlalchemy.orm import Session
from app.database import engine, Base
from app.models import User, Product, Interaction, Review
from app.ml.recommender import USERS, PRODUCTS, _generate_large_interaction_matrix, _EXTENDED_PERSONAS

logger = logging.getLogger(__name__)

def init_db(db: Session):
    Base.metadata.create_all(bind=engine)
    
    if db.query(User).first():
        logger.info("Database already initialized with data.")
        return

    logger.info("Initializing database with synthetic users, products, and interactions...")
    
    # 1. Insert Products
    for p in PRODUCTS:
        db.add(Product(
            id=p["id"],
            name=p["name"],
            category=p["category"],
            price=p["price"],
            rating=p["rating"],
            tags=",".join(p["tags"])
        ))
        
    # 2. Insert Base Users + Synthetic Users up to 500
    for u in USERS:
        db.add(User(id=u["id"], name=u["name"], avatar=u["avatar"], persona=u["persona"]))
    
    for i in range(11, 501):
        pid = f"U{i:03d}"
        persona = _EXTENDED_PERSONAS[i % len(_EXTENDED_PERSONAS)]
        db.add(User(id=pid, name=f"Synthetic {i}", avatar="SU", persona=persona))
        
    db.commit()

    # 3. Generate and Insert Interactions
    matrix = _generate_large_interaction_matrix()
    interactions = []
    
    for uid_idx in range(500):
        uid = f"U{uid_idx+1:03d}"
        for pid_idx, p in enumerate(PRODUCTS):
            rating = matrix[uid_idx, pid_idx]
            if rating > 0:
                interactions.append(Interaction(
                    user_id=uid,
                    product_id=p["id"],
                    rating=float(rating)
                ))
                
    db.bulk_save_objects(interactions)
    db.commit()
    # 4. Generate and Insert Reviews (for BERTopic)
    themes = [
        "The battery drains too fast on this model.",
        "Build quality is cheap and flimsy, broke after two weeks.",
        "Delivery was extremely slow, took a month to arrive.",
        "Software crashes constantly when opening multi-tabs.",
        "Excellent product, exceeded my expectations!",
        "Poor customer support, they ignored my emails for days.",
        "The screen flickering issue is very annoying.",
        "Beautiful design but the price is way too high.",
        "Setup was difficult and the instructions were unclear.",
        "Highly recommend for anyone in the tech industry."
    ]
    reviews = []
    for i in range(200):
        uid = f"U{random.randint(1, 500):03d}"
        pid = random.choice(PRODUCTS)["id"]
        reviews.append(Review(
            user_id=uid,
            product_id=pid,
            comment=random.choice(themes),
            rating=random.randint(1, 5)
        ))
    db.bulk_save_objects(reviews)
    db.commit()

    logger.info("Database initialization complete.")
