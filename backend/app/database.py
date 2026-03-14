import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)

# Construct PostgreSQL URI
POSTGRES_URI = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"

try:
    engine = create_engine(POSTGRES_URI, pool_pre_ping=True, connect_args={"connect_timeout": 3})
    # Force a connection check so that if Postgres is down, we immediately fallback to SQLite
    with engine.connect() as conn:
        pass
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Initialized database connection to {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}")
except Exception as e:
    logger.warning(f"PostgreSQL unreachable, falling back to SQLite: {e}")
    engine = create_engine("sqlite:///./nexus_fallback.db", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency to yield a generic DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
