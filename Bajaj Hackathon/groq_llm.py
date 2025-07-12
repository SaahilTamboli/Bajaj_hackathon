import os
import time
import logging
from typing import List, Dict, Optional, Generator
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLAMA3_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

PROMPT_TEMPLATES = {
    "financial_analysis": "Analyze the following financial data and provide insights: {context}\nQuestion: {question}",
    "cfo_commentary": "Generate a summary of the CFO's commentary from the following transcript: {context}",
    "stock_price_analysis": "Analyze the stock price data and answer: {question}\nData: {context}",
    "business_insights": "Extract key business insights from the following transcript: {context}",
    "comparative_analysis": "Compare the following quarterly data and provide a detailed analysis: {context}\nQuestion: {question}",
}

SYSTEM_PROMPT = "You are a financial analysis assistant specialized in Bajaj Finserv quarterly reports and stock data. Provide clear, concise, and accurate financial insights."

class GroqLLM:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2, max_tokens: int = 1024, top_p: float = 0.9, frequency_penalty: float = 0.0, presence_penalty: float = 0.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model if model in LLAMA3_MODELS else LLAMA3_MODELS[0]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.client = Groq(api_key=self.api_key)
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cost_per_1k = {  # Example costs, update as per Groq pricing
            "llama-3.1-8b-instant": 0.5,  # $0.50 per 1K tokens
            "llama-3.3-70b-versatile": 1.0, # $1.00 per 1K tokens
        }

    def _count_tokens(self, text: str) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except KeyError:
            # Fallback: estimate by word count if model is not supported by tiktoken
            return len(text.split())

    def _track_cost(self, prompt_tokens: int, completion_tokens: int):
        total = prompt_tokens + completion_tokens
        self.total_tokens += total
        cost = (total / 1000) * self.cost_per_1k.get(self.model, 1.0)
        self.total_cost += cost
        logger.info(f"Tokens used: {total}, Total tokens: {self.total_tokens}, Cost: ${cost:.4f}, Total cost: ${self.total_cost:.4f}")

    def _build_prompt(self, template_name: str, context: str, question: Optional[str] = None) -> str:
        template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["financial_analysis"])
        return template.format(context=context, question=question or "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
    def generate_response(self, template_name: str, context: str, question: Optional[str] = None, stream: bool = False) -> str:
        prompt = self._build_prompt(template_name, context, question)
        system_prompt = SYSTEM_PROMPT
        prompt_tokens = self._count_tokens(prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=stream
            )
            if stream:
                return self._stream_response(response, prompt_tokens)
            else:
                completion = response.choices[0].message.content
                completion_tokens = self._count_tokens(completion)
                self._track_cost(prompt_tokens, completion_tokens)
                return completion
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def _stream_response(self, response, prompt_tokens: int) -> Generator[str, None, None]:
        completion = ""
        completion_tokens = 0
        try:
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta.content or ""
                    completion += delta
                    completion_tokens += self._count_tokens(delta)
                    yield delta
            self._track_cost(prompt_tokens, completion_tokens)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def set_model(self, model: str):
        if model in LLAMA3_MODELS:
            self.model = model
            logger.info(f"Model set to {model}")
        else:
            logger.warning(f"Model {model} not supported. Using default {self.model}.")

    def set_temperature(self, temperature: float):
        self.temperature = temperature
        logger.info(f"Temperature set to {temperature}")

    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        logger.info(f"Max tokens set to {max_tokens}")

    def get_usage(self):
        return {"total_tokens": self.total_tokens, "total_cost": self.total_cost} 