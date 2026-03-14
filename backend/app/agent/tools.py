"""
NEXUS-AI Agent Tools
Each tool wraps one of the 5 ML modules so the LangChain agent can call them.
"""
import json
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Tool 1: Sentiment Analysis ─────────────────────────────────────────────────
@tool
def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of any text, review, comment, or feedback.
    Returns overall sentiment (positive/neutral/negative), confidence score,
    emotion breakdown (joy, anger, sadness, etc.), and aspect-level opinions.
    Use this when the user wants to understand the tone, emotion, or opinion
    expressed in a piece of text.

    Args:
        text: The text to analyze (up to 5000 characters)
    """
    try:
        from app.ml.sentiment import analyze
        result = analyze(text)
        overall = result["overall"]
        emotions = result.get("emotions", {})
        aspects = result.get("aspects", [])
        top_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "N/A"

        summary = (
            f"Sentiment: {overall['label'].upper()} {overall['emoji']} "
            f"(score: {overall['score']:+.3f}, confidence: {overall['confidence']*100:.0f}%)\n"
            f"Dominant emotion: {top_emotion}\n"
            f"Subjectivity: {result['metadata']['subjectivity']*100:.0f}%\n"
        )
        if aspects:
            summary += "Aspect breakdown:\n"
            for a in aspects[:4]:
                summary += f"  • {a['aspect']}: {a['sentiment']} ({a['score']:+.2f})\n"
        return summary
    except Exception as e:
        logger.error(f"Sentiment tool error: {e}")
        return f"Error running sentiment analysis: {e}"


# ── Tool 2: Fraud Detection ────────────────────────────────────────────────────
@tool
def detect_fraud(transaction_json: str) -> str:
    """
    Analyze a financial transaction for fraud risk using an ML ensemble model.
    Returns a fraud score (0-1), risk level (LOW/MEDIUM/HIGH), confidence,
    and human-readable reasons explaining the prediction.
    Use this when the user describes a transaction or asks if something looks suspicious.

    Args:
        transaction_json: JSON string with transaction fields:
            - amount (float, required): transaction amount in USD
            - merchant_category (str): e.g. "atm", "grocery", "electronics", "luxury"
            - velocity_1h (int): number of transactions in the last hour (default 1)
            - distance_from_home_km (float): distance from cardholder home (default 10)
            - unusual_location (int): 1 if location is unusual, 0 otherwise (default 0)
            - timestamp (str, optional): ISO datetime string
    """
    try:
        from app.ml.fraud_model import predict_fraud
        tx = json.loads(transaction_json)
        result = predict_fraud(tx)
        summary = (
            f"Fraud Score: {result['fraud_score']*100:.1f}% — Risk Level: {result['risk_level']}\n"
            f"Confidence: {result['confidence']*100:.0f}%\n"
            f"Verdict: {'⚠️ FLAGGED AS FRAUD' if result['is_fraud'] else '✅ Appears Legitimate'}\n"
            f"Reasons:\n"
        )
        for r in result["reasons"]:
            summary += f"  • {r}\n"
        return summary
    except json.JSONDecodeError:
        return "Error: transaction_json must be valid JSON. Example: {\"amount\": 500, \"merchant_category\": \"atm\"}"
    except Exception as e:
        logger.error(f"Fraud tool error: {e}")
        return f"Error running fraud detection: {e}"


# ── Tool 3: Recommendations ────────────────────────────────────────────────────
@tool
def get_recommendations(user_id: str) -> str:
    """
    Get personalized product recommendations for a user using a hybrid
    SVD collaborative filtering + content-based ML model.
    Available user IDs: U001 (Alex Chen, tech), U002 (Maria Garcia, fitness),
    U003 (James Wilson, gaming), U004 (Emma Davis, books), U005 (Liam Brown, cooking),
    U006 (Sophia Lee, fashion), U007 (Noah Martinez, outdoor), U008 (Olivia Taylor, beauty).
    Use this when the user asks about recommendations, products, or suggestions for a specific user.

    Args:
        user_id: The user ID string (e.g. "U001", "U002", etc.)
    """
    try:
        from app.ml.recommender import get_recommendations as _get_recs, USERS
        user = next((u for u in USERS if u["id"] == user_id.strip().upper()), None)
        if not user:
            return f"User {user_id} not found. Valid IDs: U001–U008."
        recs = _get_recs(user_id.strip().upper(), top_n=4)
        if not recs:
            return f"No recommendations found for {user_id}."
        summary = f"Top recommendations for {user['name']} ({user['persona'].replace('_', ' ').title()}):\n"
        for i, r in enumerate(recs, 1):
            summary += (
                f"  {i}. {r['name']} — ${r['price']:.2f} "
                f"(⭐{r['rating']}, {r['recommendation_score']*100:.0f}% match)\n"
                f"     Reason: {r['match_reason']}\n"
            )
        return summary
    except Exception as e:
        logger.error(f"Recommend tool error: {e}")
        return f"Error fetching recommendations: {e}"


# ── Tool 4: Document Q&A (RAG) ─────────────────────────────────────────────────
@tool
def query_documents(question: str) -> str:
    """
    Query the knowledge base using RAG (Retrieval-Augmented Generation).
    Searches through all uploaded PDF documents and returns a grounded answer
    based on the actual document content. Use this when the user asks questions
    about documents, uploaded files, or wants information from the knowledge base.

    Args:
        question: The question to ask about the documents
    """
    try:
        from app.rag.vectorstore import get_vectorstore
        from app.config import settings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)

        if not docs:
            return "No relevant documents found in the knowledge base. Please upload a PDF first."

        context = "\n\n".join(d.page_content for d in docs)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer based only on the provided context. Be concise."),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return f"Based on your documents:\n{answer}"
    except Exception as e:
        logger.error(f"RAG tool error: {e}")
        return f"Could not query documents: {e}. Make sure you've uploaded and ingested a PDF first."


# ── Tool 5: Trending Products ──────────────────────────────────────────────────
@tool
def get_trending_products(count: int = 5) -> str:
    """Get the currently trending products across all users. Use this when the user
    asks what's popular or wants general (non-personalized) product picks.

    Args:
        count: Number of trending products to return (1-10, default 5)
    """
    try:
        from app.ml.recommender import get_trending
        count = max(1, min(10, count))
        items = get_trending(top_n=count)
        summary = f"Top {len(items)} trending products:\n"
        for i, p in enumerate(items, 1):
            summary += f"  {i}. {p['name']} ({p['category']}) — ${p['price']:.2f} ⭐{p['rating']}\n"
        return summary
    except Exception as e:
        return f"Error fetching trending products: {e}"


# ── Tool 6: Smart Recommendations (Cross-Module) ───────────────────────────────
@tool
def smart_product_recommendations(user_id: str, fraud_risk: str = "NONE") -> str:
    """
    Get cross-module intelligent recommendations that combine:
    - SVD collaborative filtering + content-based ML scores
    - Product sentiment health (from DistilBERT review analysis)
    - Fraud risk adjustment (de-ranks luxury items for high-risk accounts)
    Use this instead of get_recommendations when you want the full cross-module picture.

    Args:
        user_id: User ID (U001-U008)
        fraud_risk: Account risk level — NONE, LOW, MEDIUM, or HIGH.
                    HIGH: de-ranks products >$300 and adds fraud flags.
    """
    try:
        from app.ml.cross_module import get_sentiment_adjusted_recommendations
        from app.ml.recommender import USERS
        user = next((u for u in USERS if u["id"] == user_id.strip().upper()), None)
        if not user:
            return f"User {user_id} not found. Valid IDs: U001–U008."
        fraud_risk_clean = fraud_risk.upper() if fraud_risk.upper() in ("LOW", "MEDIUM", "HIGH") else None
        recs = get_sentiment_adjusted_recommendations(
            user_id.strip().upper(), top_n=4, fraud_risk=fraud_risk_clean
        )
        summary = f"Smart recommendations for {user['name']} (fraud_risk={fraud_risk.upper()}):\n"
        for i, r in enumerate(recs, 1):
            flag = f" | {r['fraud_flag']}" if r.get('fraud_flag') else ""
            summary += (
                f"  {i}. {r['name']} — ${r['price']:.2f}\n"
                f"     Score: {r['recommendation_score']*100:.0f}% "
                f"| Sentiment: {r.get('sentiment_health_label','N/A')}{flag}\n"
                f"     Reason: {r['match_reason']}\n"
            )
        return summary
    except Exception as e:
        return f"Error: {e}"


# ── Tool 7: Product Complaint Analysis (Cross-Module) ─────────────────────────
@tool
def explain_product_complaints(category: str) -> str:
    """
    Cross-module tool: analyzes review sentiment for all products in a category
    using DistilBERT and reports which products have poor sentiment, what complaints
    customers have, and which products should be removed from recommendations.
    Use this when asked: 'Which products should we stop recommending?'
    or 'What are customers complaining about in [category]?'

    Args:
        category: Product category name, e.g. 'Electronics', 'Gaming', 'Sports',
                  'Books', 'Home & Kitchen', 'Clothing', 'Beauty', 'Automotive'
    """
    try:
        from app.ml.cross_module import explain_complaints_for_category
        return explain_complaints_for_category(category)
    except Exception as e:
        return f"Error analyzing complaints: {e}"


# ── Tool 8: Visual Product Search (CLIP) ──────────────────────────────────────
@tool
def find_visually_similar_products(image_description: str) -> str:
    """
    Conceptually searches the product catalog for items visually similar to
    a described image. In the full platform, this uses CLIP embeddings.
    For agent text-based queries, this maps the description to relevant catalog categories.
    Use this when the user describes a visual product or says 'find products that look like X'.

    Args:
        image_description: Text description of what the user is looking for visually,
                           e.g. 'wireless headphones', 'fitness tracker watch', 'gaming mouse'
    """
    try:
        from app.ml.visual_search import search_by_description
        
        # Call the CLIP text-to-image semantic search
        results = search_by_description(image_description, top_k=5)
        
        if not results:
            return f"No products visually matched '{image_description}'."
            
        if "error" in results[0]:
            return f"Error using visual search: {results[0]['error']}"
            
        summary = f"Visually similar products for '{image_description}':\n"
        for rank, prod in enumerate(results, 1):
            summary += f"  {rank}. {prod['name']} ({prod['category']}) — ${prod['price']:.2f} ⭐{prod['rating']} (Visual match: {prod.get('similarity_pct', 0)}%)\n"
        return summary
    except Exception as e:
        return f"Error: {e}"


# ── Exported tool list ─────────────────────────────────────────────────────────
NEXUS_TOOLS = [
    analyze_sentiment,
    detect_fraud,
    get_recommendations,
    query_documents,
    get_trending_products,
    smart_product_recommendations,
    explain_product_complaints,
    find_visually_similar_products,
]
