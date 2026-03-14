from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
from app.rag.ingestion import process_and_ingest_document

router = APIRouter()

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    namespace: str = "default",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict[str, Any]:
    """
    Endpoint to upload a PDF document and ingest it into the vector store.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        chunks_ingested = await process_and_ingest_document(
            file=file,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return {
            "status": "success",
            "message": f"Successfully ingested document '{file.filename}'",
            "chunks": chunks_ingested
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
