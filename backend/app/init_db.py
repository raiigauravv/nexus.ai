import logging
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from app.models import User
from app.ml.recommender import USERS

logger = logging.getLogger(__name__)

def init_db(db: Session):
    try:
        has_existing_user = db.execute(select(User.id).limit(1)).scalar_one_or_none()
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Database schema is not initialized. Run migrations with 'alembic upgrade head'."
        ) from exc

    if has_existing_user:
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

