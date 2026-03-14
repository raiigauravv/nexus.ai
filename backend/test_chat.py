import asyncio
from app.rag.vectorstore import get_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from app.config import settings

async def main():
    os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        streaming=True,
        google_api_key=settings.GEMINI_API_KEY
    )
    
    system_prompt = (
        "You are the NEXUS-AI intelligent assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer based on the context, say that you don't know. "
        "Keep your reasoning concise and professional.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "chat_history": lambda x: x["chat_history"], 
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Streaming...")
    try:
        async for chunk in rag_chain.astream({
            "chat_history": [],
            "input": "What is in the document?"
        }):
            print(chunk, end="")
        print()
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
