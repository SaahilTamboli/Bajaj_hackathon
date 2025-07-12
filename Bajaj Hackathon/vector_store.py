import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.transcript_collection = self._get_or_create_collection('transcripts')
        self.stock_collection = self._get_or_create_collection('stock_data')

    def _get_or_create_collection(self, name: str):
        # Use get_or_create logic to avoid duplicate collection errors
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name)

    def create_vectorstore(self):
        # Only clear and re-create collections if they exist
        self.clear_database()
        self.transcript_collection = self._get_or_create_collection('transcripts')
        self.stock_collection = self._get_or_create_collection('stock_data')
        logger.info("Vectorstore created with separate collections.")

    def is_populated(self) -> bool:
        # Check if collections have any documents
        try:
            transcript_count = len(self.transcript_collection.get(ids=None)["ids"])
            stock_count = len(self.stock_collection.get(ids=None)["ids"])
            return transcript_count > 0 or stock_count > 0
        except Exception:
            return False

    def load_existing_vectorstore(self):
        self.transcript_collection = self._get_or_create_collection('transcripts')
        self.stock_collection = self._get_or_create_collection('stock_data')
        logger.info("Loaded existing vectorstore from disk.")

    def add_documents(self, docs: List[Dict[str, Any]], doc_type: str):
        collection = self.transcript_collection if doc_type == 'transcript' else self.stock_collection
        texts = [doc['text'] for doc in docs]
        metadatas = [doc['metadata'] for doc in docs]
        ids = [f"{doc_type}_{i}_{os.urandom(4).hex()}" for i in range(len(docs))]
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
        logger.info(f"Added {len(docs)} documents to {doc_type} collection.")

    def search_similar(self, query: str, doc_type: str = 'transcript', top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
        collection = self.transcript_collection if doc_type == 'transcript' else self.stock_collection
        from sentence_transformers import SentenceTransformer
        query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([query], convert_to_numpy=True)[0]
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if filters is not None and len(filters) > 0:
            query_args["where"] = filters
        results = collection.query(**query_args)
        return results

    def get_relevant_context(self, query: str, doc_type: str = 'transcript', top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
        results = self.search_similar(query, doc_type, top_k, filters)
        # Return texts and scores
        return list(zip(results['documents'][0], results['distances'][0], results['metadatas'][0]))

    def clear_database(self):
        for name in ['transcripts', 'stock_data']:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        logger.info("Cleared all collections in vectorstore.")

    def search_by_date_range(self, start_date: str, end_date: str, doc_type: str = 'transcript', top_k: int = 5):
        # Assumes metadata contains a 'date' field in 'YYYY-MM-DD' or similar format
        filters = {"date": {"$gte": start_date, "$lte": end_date}}
        return self.search_similar("", doc_type, top_k, filters)

    def filter_by_document_type(self, doc_type: str, top_k: int = 5):
        filters = {"document_type": doc_type}
        return self.search_similar("", doc_type, top_k, filters)

    def retrieve_top_k(self, query: str, doc_type: str = 'transcript', k: int = 5):
        return self.get_relevant_context(query, doc_type, k) 