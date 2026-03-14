import asyncio
import sys
from app.rag.vectorstore import get_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import os
from app.config import settings

async def main():
    try:
        os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
        
        # Simulate exact objects
        messages = [
            {"parts": [{"type": "text", "text": "Hello"}], "id": "1", "role": "user"},
            {"parts": [{"type": "text", "text": "What is in the document?"}], "id": "2", "role": "user"}
        ]
        
        converted = []
        for msg in messages[:-1]:
            content = msg.get("content") or msg.get("text") or ""
            if msg.get("role") == "user":
                converted.append(HumanMessage(content=content))
                
        user_query = messages[-1].get("content") or messages[-1].get("text") or "Hello"

        vectorstore = get_vectorstore(namespace="default")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            streaming=True,
            google_api_key=settings.GEMINI_API_KEY
        )
        
        system_prompt = "Context: {context}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("user", "{input}")
        ])
        
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
            
        rag_chain = (
            {
                "context": (lambda x: x["input"]) | retriever | format_docs, 
                "chat_history": lambda x: x["chat_history"], 
                "input": lambda x: x["input"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print(f"Query: {user_query}")
        print(f"History: {converted}")
        
        async for chunk in rag_chain.astream({
            "chat_history": converted,
            "input": user_query
        }):
            print(chunk, end="")
            sys.stdout.flush()
        print("\nDone")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
