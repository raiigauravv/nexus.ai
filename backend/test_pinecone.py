import os
import asyncio
from app.config import settings
from pinecone import Pinecone

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
idx = pc.Index(settings.PINECONE_INDEX_NAME)

print("Deleting all vectors in namespace 'default'...")
try:
    idx.delete(delete_all=True, namespace="default")
    print("Success: delete_all=True works!")
except Exception as e:
    print("Error:", e)
