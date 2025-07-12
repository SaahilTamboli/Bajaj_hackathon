import os
import logging
from dotenv import load_dotenv
from typing import Optional

# Load .env file if present
load_dotenv()

class ConfigError(Exception):
    pass

class AppConfig:
    # Default prompt templates
    DEFAULT_PROMPT_TEMPLATES = {
        "financial_analysis": "Analyze the following financial data and provide insights: {context}\nQuestion: {question}",
        "cfo_commentary": "Generate a summary of the CFO's commentary from the following transcript: {context}",
        "stock_price_analysis": "Analyze the stock price data and answer: {question}\nData: {context}",
        "business_insights": "Extract key business insights from the following transcript: {context}",
        "comparative_analysis": "Compare the following quarterly data and provide a detailed analysis: {context}\nQuestion: {question}",
    }

    def __init__(self, env: Optional[str] = None):
        self.env = env or os.getenv("APP_ENV", "dev")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db")
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.DATA_DIR = os.getenv("DATA_DIR", "documents")
        self._configure_logging()
        self._validate()

    def _configure_logging(self):
        logging.basicConfig(level=getattr(logging, self.LOG_LEVEL.upper(), logging.INFO),
                            format='%(asctime)s %(levelname)s %(message)s')

    def _validate(self):
        if not self.GROQ_API_KEY:
            raise ConfigError("GROQ_API_KEY is not set in environment variables or .env file.")
        if self.MODEL_NAME not in ["llama3-8b-8192", "llama3-70b-8192"]:
            raise ConfigError(f"MODEL_NAME must be one of 'llama3-8b-8192' or 'llama3-70b-8192', got {self.MODEL_NAME}")
        if self.CHUNK_SIZE <= 0 or self.CHUNK_OVERLAP < 0:
            raise ConfigError("CHUNK_SIZE must be > 0 and CHUNK_OVERLAP >= 0.")
        if self.MAX_TOKENS <= 0:
            raise ConfigError("MAX_TOKENS must be > 0.")
        if not os.path.isdir(self.DATA_DIR):
            os.makedirs(self.DATA_DIR, exist_ok=True)
        if not os.path.isdir(self.VECTOR_DB_PATH):
            os.makedirs(self.VECTOR_DB_PATH, exist_ok=True)

    @classmethod
    def from_env(cls, env: Optional[str] = None):
        return cls(env=env)

    def as_dict(self):
        return self.__dict__

# Usage:
# config = AppConfig.from_env()
# print(config.GROQ_API_KEY) 