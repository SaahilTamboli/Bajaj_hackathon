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
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", 
                 temperature: float = 0.2, max_tokens: int = 1024, top_p: float = 0.9, 
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required. Set it as an environment variable or pass it directly.")
        
        self.model = model if model in LLAMA3_MODELS else LLAMA3_MODELS[0]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.client = Groq(api_key=self.api_key)
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Updated cost structure - approximate values (check Groq's current pricing)
        self.cost_per_1k = {
            "llama-3.1-8b-instant": 0.05,    # Input: $0.05/1K tokens, Output: $0.08/1K tokens
            "llama-3.3-70b-versatile": 0.27,  # Input: $0.27/1K tokens, Output: $0.27/1K tokens
        }
        
        # Initialize tokenizer - use a generic one since Groq models may not be in tiktoken
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer as fallback
        except Exception as e:
            logger.warning(f"Could not load tiktoken encoder: {e}. Using word-based estimation.")
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback to word count estimation"""
        if not text:
            return 0
            
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tokenizer error: {e}. Using word count estimation.")
        
        # Fallback: rough estimation (1 token ≈ 0.75 words for most models)
        return int(len(text.split()) * 1.33)

    def _track_cost(self, prompt_tokens: int, completion_tokens: int):
        """Track token usage and cost"""
        total = prompt_tokens + completion_tokens
        self.total_tokens += total
        cost = (total / 1000) * self.cost_per_1k.get(self.model, 0.27)
        self.total_cost += cost
        logger.info(f"Tokens used: {total} (prompt: {prompt_tokens}, completion: {completion_tokens}), "
                   f"Total tokens: {self.total_tokens}, Cost: ${cost:.4f}, Total cost: ${self.total_cost:.4f}")

    def _build_prompt(self, template_name: str, context: str, question: Optional[str] = None) -> str:
        """Build prompt from template"""
        template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["financial_analysis"])
        return template.format(context=context, question=question or "")

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10), 
        retry=retry_if_exception_type((Exception,))
    )
    def generate_response(self, template_name: str, context: str, question: Optional[str] = None, 
                         stream: bool = False) -> str:
        """Generate response using Groq API"""
        prompt = self._build_prompt(template_name, context, question)
        system_prompt = SYSTEM_PROMPT
        prompt_tokens = self._count_tokens(system_prompt + prompt)
        
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
        """Handle streaming response"""
        completion = ""
        
        try:
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        completion += content
                        yield content
            
            # Track cost after streaming is complete
            completion_tokens = self._count_tokens(completion)
            self._track_cost(prompt_tokens, completion_tokens)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def generate_streaming_response(self, template_name: str, context: str, 
                                  question: Optional[str] = None) -> Generator[str, None, None]:
        """Generate streaming response (returns generator)"""
        prompt = self._build_prompt(template_name, context, question)
        system_prompt = SYSTEM_PROMPT
        prompt_tokens = self._count_tokens(system_prompt + prompt)
        
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
                stream=True
            )
            
            completion = ""
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        completion += content
                        yield content
            
            # Track cost after streaming is complete
            completion_tokens = self._count_tokens(completion)
            self._track_cost(prompt_tokens, completion_tokens)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def set_model(self, model: str):
        """Set the model to use"""
        if model in LLAMA3_MODELS:
            self.model = model
            logger.info(f"Model set to {model}")
        else:
            logger.warning(f"Model {model} not supported. Available models: {LLAMA3_MODELS}. Using default {self.model}.")

    def set_temperature(self, temperature: float):
        """Set temperature for response generation"""
        if 0.0 <= temperature <= 2.0:
            self.temperature = temperature
            logger.info(f"Temperature set to {temperature}")
        else:
            logger.warning(f"Temperature {temperature} out of range [0.0, 2.0]. Keeping current value {self.temperature}.")

    def set_max_tokens(self, max_tokens: int):
        """Set maximum tokens for response"""
        if max_tokens > 0:
            self.max_tokens = max_tokens
            logger.info(f"Max tokens set to {max_tokens}")
        else:
            logger.warning(f"Max tokens must be positive. Keeping current value {self.max_tokens}.")

    def get_usage(self) -> Dict[str, float]:
        """Get current usage statistics"""
        return {
            "total_tokens": self.total_tokens, 
            "total_cost": self.total_cost,
            "model": self.model
        }
    
    def reset_usage(self):
        """Reset usage statistics"""
        self.total_tokens = 0
        self.total_cost = 0.0
        logger.info("Usage statistics reset")

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return LLAMA3_MODELS.copy()

    def get_model_info(self) -> Dict[str, any]:
        """Get information about current model configuration"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "cost_per_1k_tokens": self.cost_per_1k.get(self.model, 0.27)
        }

    def validate_api_key(self) -> bool:
        """Validate API key by making a simple request"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize the LLM
    llm = GroqLLM(model="llama-3.3-70b-versatile", temperature=0.3)
    
    # Test API key validation
    if not llm.validate_api_key():
        print("API key validation failed!")
        exit(1)
    
    # Example financial analysis
    context = """
    Q3 2024 Results:
    - Revenue: $1.2B (up 15% YoY)
    - Net Income: $200M (up 25% YoY)
    - Assets Under Management: $50B (up 12% YoY)
    - Loan Portfolio: $30B (up 8% YoY)
    """
    
    question = "What are the key performance indicators and growth trends?"
    
    print("=== Standard Response ===")
    response = llm.generate_response("financial_analysis", context, question)
    print(response)
    
    print("\n=== Streaming Response ===")
    for chunk in llm.generate_streaming_response("financial_analysis", context, question):
        print(chunk, end="", flush=True)
    
    print(f"\n\n=== Usage Statistics ===")
    usage = llm.get_usage()
    print(f"Total tokens: {usage['total_tokens']}")
    print(f"Total cost: ${usage['total_cost']:.4f}")
    
    print(f"\n=== Model Info ===")
    model_info = llm.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")