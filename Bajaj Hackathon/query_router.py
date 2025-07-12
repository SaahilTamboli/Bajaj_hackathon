import re
import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import dateutil.parser
from collections import defaultdict

# For NER, use a simple rule-based approach (can be replaced with spaCy or transformers for more power)
FINANCIAL_TERMS = [
    "stock price", "price", "average", "highest", "lowest", "quarter", "Q1", "Q2", "Q3", "Q4",
    "organic traffic", "headwinds", "partnership", "stake sale", "CFO", "commentary", "investor call",
    "performance", "comparison", "compare", "business", "insight", "discussion", "Hero", "Allianz", "BAGIC", "Bajaj"
]

QUERY_CATEGORIES = {
    "stock_analysis": ["stock price", "highest", "lowest", "average", "price", "across"],
    "business_insights": ["insight", "business", "organic traffic", "headwinds", "rationale"],
    "comparative_analysis": ["compare", "comparison", "between", "across quarters", "performance"],
    "strategic_discussion": ["strategy", "strategic", "discussion", "partnership", "stake sale", "Hero", "Allianz"],
    "cfo_commentary": ["CFO", "commentary", "draft", "investor call"],
}

logger = logging.getLogger(__name__)

class QueryRouter:
    def __init__(self):
        self.last_entities = defaultdict(str)

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        for cat, keywords in QUERY_CATEGORIES.items():
            if any(kw.lower() in q for kw in keywords):
                return cat
        return "business_insights"  # default

    def extract_dates(self, query: str) -> List[str]:
        # Extract date-like strings (e.g., Q1 2024, Jan-24, March 2023)
        date_patterns = [
            r"Q[1-4]\s*\d{2,4}",
            r"[A-Za-z]{3,9}[- ]?\d{2,4}",
            r"\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}",
            r"\d{4}"
        ]
        dates = []
        for pat in date_patterns:
            matches = re.findall(pat, query)
            dates.extend(matches)
        # Normalize dates (try to parse)
        norm_dates = []
        for d in dates:
            try:
                norm = dateutil.parser.parse(d, fuzzy=True)
                norm_dates.append(norm.strftime("%Y-%m-%d"))
            except Exception:
                norm_dates.append(d)
        return norm_dates

    def extract_entities(self, query: str) -> Dict[str, Any]:
        entities = {"dates": self.extract_dates(query), "terms": [], "companies": []}
        for term in FINANCIAL_TERMS:
            if term.lower() in query.lower():
                if term in ["BAGIC", "Bajaj", "Allianz", "Hero"]:
                    entities["companies"].append(term)
                else:
                    entities["terms"].append(term)
        return entities

    def build_context(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        intent = self.classify_intent(query)
        entities = self.extract_entities(query)
        context = {
            "intent": intent,
            "entities": entities,
            "chat_history": chat_history or [],
        }
        self.last_entities = entities
        return context

    def generate_prompt(self, query: str, context: Dict[str, Any]) -> str:
        # Build a contextual prompt based on intent and extracted entities
        intent = context["intent"]
        entities = context["entities"]
        prompt = f"Query Type: {intent}\n"
        if entities["dates"]:
            prompt += f"Relevant Dates: {', '.join(entities['dates'])}\n"
        if entities["companies"]:
            prompt += f"Companies: {', '.join(entities['companies'])}\n"
        if entities["terms"]:
            prompt += f"Financial Terms: {', '.join(entities['terms'])}\n"
        prompt += f"User Question: {query}\n"
        return prompt

    def handle_followup(self, query: str, chat_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        # Use last entities if follow-up is detected (e.g., "What about Q3?")
        context = self.build_context(query, chat_history)
        if not context["entities"]["dates"] and self.last_entities.get("dates"):
            context["entities"]["dates"] = self.last_entities["dates"]
        return query, context

    def route(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        # Main entry: classify, extract, build context, generate prompt
        if chat_history is None:
            chat_history = []
        if self.is_followup(query):
            query, context = self.handle_followup(query, chat_history)
        else:
            context = self.build_context(query, chat_history)
        prompt = self.generate_prompt(query, context)
        return {
            "intent": context["intent"],
            "entities": context["entities"],
            "prompt": prompt,
            "context": context,
        }

    def is_followup(self, query: str) -> bool:
        # Simple heuristic: short queries or those starting with "What about", "And", etc.
        followup_starts = ["what about", "and ", "also ", "how about"]
        q = query.strip().lower()
        return len(q.split()) < 5 or any(q.startswith(fu) for fu in followup_starts) 