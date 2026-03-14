import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def test_gemini():
    print("Testing Gemini API...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        
        # Test Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vector = embeddings.embed_query("Test query")
        print(f"✅ Gemini Embeddings successful. Dimension: {len(vector)}")
        
        # Test Chat
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        response = llm.invoke("Say 'Hello'")
        print(f"✅ Gemini Chat successful. Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return False

def test_pinecone():
    print("\nTesting Pinecone API...")
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes()
        index_names = [i["name"] for i in indexes]
        
        print(f"✅ Pinecone connection successful.")
        print(f"Found indexes: {index_names}")
        
        expected_index = os.getenv("PINECONE_INDEX_NAME", "nexus-ai-rag")
        if expected_index in index_names:
            print(f"✅ Index '{expected_index}' exists.")
        else:
            print(f"❌ Warning: Index '{expected_index}' does not exist. You need to create it with dimension=768.")
        return True
    except Exception as e:
        print(f"❌ Pinecone Error: {e}")
        return False

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("Error: Please set GEMINI_API_KEY and PINECONE_API_KEY in .env")
        sys.exit(1)
        
    gemini_ok = test_gemini()
    pinecone_ok = test_pinecone()
    
    if gemini_ok and pinecone_ok:
        print("\n🎉 All APIs are working perfectly!")
        sys.exit(0)
    else:
        print("\n❌ API verification failed.")
        sys.exit(1)
