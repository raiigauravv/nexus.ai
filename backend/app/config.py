from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "NEXUS-AI"
    API_V1_STR: str = "/api/v1"
    
    # Frontend definition for CORS
    FRONTEND_URL: str = "http://localhost:3000"
    
    # Database
    POSTGRES_USER: str = "nexus_user"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "nexus_db"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    
    # Storage (MinIO)
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = ""
    MINIO_SERVER: str = "localhost:9000"
    
    # Redis Cache/Broker
    REDIS_URL: str = "redis://localhost:6379/0"

    # External APIs
    GEMINI_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "nexus-ai-rag"
    PINECONE_VISION_INDEX: str = "nexus-ai-vision"

    # Auth / JWT
    JWT_SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # MLflow
    MLFLOW_TRACKING_URI: str = "sqlite:///mlruns.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",          # silently drop any env vars not declared above
    )

settings = Settings()
