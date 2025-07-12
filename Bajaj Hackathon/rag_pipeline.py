import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms.base import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt templates for different query types
PROMPT_TEMPLATES = {
    "stock_price": "You are a financial analyst. Using the following stock price data, answer the user's question.\nContext:\n{context}\nQuestion: {question}\nAnswer:",
    "business_insight": "You are an expert in business analysis. Using the following transcript, extract key business insights.\nContext:\n{context}\nQuestion: {question}\nAnswer:",
    "comparative": "You are a financial analyst. Compare the following quarterly data and answer the user's question.\nContext:\n{context}\nQuestion: {question}\nAnswer:",
    "cfo_commentary": "You are the CFO. Draft a commentary for the upcoming investor call using the following transcript.\nContext:\n{context}\nQuestion: {question}\nCommentary:",
    "strategy": "You are a strategic advisor. Extract and summarize strategic discussions from the following transcript.\nContext:\n{context}\nQuestion: {question}\nSummary:",
}

# Query routing heuristics
QUERY_TYPE_KEYWORDS = {
    "stock_price": ["stock price", "highest", "lowest", "average", "price", "across"],
    "business_insight": ["insight", "business", "organic traffic", "headwinds", "rationale"],
    "comparative": ["compare", "comparison", "between", "across quarters"],
    "cfo_commentary": ["CFO", "commentary", "draft", "investor call"],
    "strategy": ["strategy", "strategic", "discussion", "partnership", "stake sale"],
}

class SimpleRetriever:
    """A simple retriever interface for vector store integration."""
    def __init__(self, vectorstore, doc_type: str = 'transcript'):
        self.vectorstore = vectorstore
        self.doc_type = doc_type

    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.vectorstore.get_relevant_context(query, doc_type=self.doc_type, top_k=top_k)

class RAGPipeline:
    def __init__(self, vectorstore, llm: LLM, top_k: int = 5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.retrievers = {
            "stock_price": SimpleRetriever(vectorstore, doc_type="stock_data"),
            "business_insight": SimpleRetriever(vectorstore, doc_type="transcript"),
            "comparative": SimpleRetriever(vectorstore, doc_type="transcript"),
            "cfo_commentary": SimpleRetriever(vectorstore, doc_type="transcript"),
            "strategy": SimpleRetriever(vectorstore, doc_type="transcript"),
        }

    def route_query(self, question: str) -> str:
        q = question.lower()
        for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return qtype
        return "business_insight"  # default

    def build_prompt(self, query_type: str, context: str, question: str) -> str:
        template = PROMPT_TEMPLATES.get(query_type, PROMPT_TEMPLATES["business_insight"])
        return template.format(context=context, question=question)

    def get_sources(self, docs: List[Tuple[str, float, Dict[str, Any]]]) -> List[str]:
        # Return file paths or metadata for citation
        return [doc[2].get('file_path', 'unknown') for doc in docs]

    def get_confidence(self, docs: List[Tuple[str, float, Dict[str, Any]]]) -> float:
        # Simple confidence: average similarity (lower is better)
        if not docs:
            return 0.0
        scores = [1 - min(1, d[1]) for d in docs]  # Convert distance to confidence
        return sum(scores) / len(scores)

    def ask(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        query_type = self.route_query(question)
        retriever = self.retrievers[query_type]
        docs = retriever.get_relevant_documents(question, top_k=self.top_k)
        context = "\n".join([doc[0] for doc in docs])
        prompt = self.build_prompt(query_type, context, question)
        # Multi-turn: Optionally add chat history to prompt
        if chat_history:
            history_str = "\n".join([f"{h['role']}: {h['content']}" for h in chat_history])
            prompt = f"{history_str}\n{prompt}"
        # LLM generation
        answer = self.llm.generate_response(query_type, context, question)
        sources = self.get_sources(docs)
        confidence = self.get_confidence(docs)
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "query_type": query_type,
        } 