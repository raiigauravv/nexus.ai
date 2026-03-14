from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import os
from app.config import settings

os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY

async def test():
    try:
        e1 = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=settings.GEMINI_API_KEY)
        await e1.aembed_query("")
        print("Success")
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
