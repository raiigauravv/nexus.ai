import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.config import settings

def get_embeddings():
    """Returns the embedding model initialized with the API key."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Cannot initialize embeddings.")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GEMINI_API_KEY
    )

def get_vectorstore(namespace: str = "default"):
    """
    Returns a connected LangChain PineconeVectorStore instance.
    Make sure the index 'nexus-ai-rag' exists in your Pinecone project with dimension=3072
    """
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set. Cannot initialize vector store.")
    
    # Initialize pinecone client
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    # Check if index exists, and warn if not
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if settings.PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Warning: Index '{settings.PINECONE_INDEX_NAME}' does not exist in Pinecone. Please create it first.")
    
    embeddings = get_embeddings()
    
    # langchain-pinecone expects the API key in the environment variables
    os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
    
    # Initialize the Pinecone Vector Store wrapper
    vectorstore = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )
    
    return vectorstore
