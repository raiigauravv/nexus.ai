import logging
from sqlalchemy.orm import Session
from app.database import engine, Base
from app.models import User, Product, Interaction, Review
from app.ml.recommender import USERS

logger = logging.getLogger(__name__)

def init_db(db: Session):
    Base.metadata.create_all(bind=engine)
    
    if db.query(User).first():
        logger.info("Database already initialized with data.")
        return

    logger.info("Initializing database with mock users...")
    
    # Insert Mock Users
    for u in USERS:
        # We don't have a hashed_password for these mock users, they are just for display.
        # But for the new Auth system, maybe we don't even need to insert them if they can't log in.
        # However, to prevent Foreign Key errors if we log interactions later, let's insert them.
        db.add(User(id=u["id"], name=u["name"], avatar=u["avatar"], persona=u["persona"], hashed_password=None))
        
    db.commit()

    logger.info("Database initialization complete.")

