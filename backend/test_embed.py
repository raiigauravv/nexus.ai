from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import os
from app.config import settings

os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

async def test():
    try:
        e1 = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=settings.GEMINI_API_KEY)
        await e1.aembed_query("Hello")
        print("Success without task_type")
    except Exception as e:
        print(f"Error 1: {e}")
        
    try:
        e2 = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=settings.GEMINI_API_KEY, task_type=None)
        await e2.aembed_query("Hello")
        print("Success with task_type=None")
    except Exception as e:
        print(f"Error 2: {e}")

asyncio.run(test())
