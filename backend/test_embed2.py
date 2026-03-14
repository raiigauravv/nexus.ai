import asyncio
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

async def test():
    try:
        e1 = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=settings.GEMINI_API_KEY)
        await e1.aembed_query("Hello")
        print("Success without task_type")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
