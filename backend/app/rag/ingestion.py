import os
import uuid
import asyncio
from typing import List, Optional
from fastapi import UploadFile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.vectorstore import get_vectorstore

async def process_and_ingest_document(
    file: UploadFile, 
    namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> int:
    """
    Process an uploaded file (currently PDFs), chunk it, and ingest into Pinecone.
    Returns the number of chunks ingested.
    """
    # 1. Save file temporarily
    temp_file_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # 2. Load Document
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
        else:
            raise ValueError("Unsupported file type. Only PDF is supported currently.")
            
        # Add metadata to each page
        for doc in docs:
            doc.metadata["source_filename"] = file.filename
            
        # 3. Chunk Document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(docs)
        
        # 4. Ingest into Vector Store
        if chunks:
            # First, try to clear the existing namespace so new documents don't clash with old ones
            try:
                from pinecone import Pinecone
                from app.config import settings
                pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                idx = pc.Index(settings.PINECONE_INDEX_NAME)
                idx.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                print(f"Failed to clear Pinecone namespace (it might be empty): {e}")

            vectorstore = get_vectorstore(namespace=namespace)
            
            # Gemini Free Tier allows 100 requests per minute.
            # For this portfolio demo, we cap the processed chunks to avoid hitting limits.
            max_chunks = 80
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
                
            # Process in small batches with brief delays to smooth out spikes
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                vectorstore.add_documents(batch)
                if i + batch_size < len(chunks):
                    await asyncio.sleep(2)
            
        return len(chunks)
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
