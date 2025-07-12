import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use pdfplumber for robust PDF parsing
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def validate_file_format(self, file_path: str, expected_ext: List[str]) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in expected_ext:
            logger.error(f"Invalid file format: {file_path}")
            return False
        return True

    def process_quarterly_transcripts(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        documents = []
        for pdf_path in pdf_paths:
            if not self.validate_file_format(pdf_path, ['.pdf']):
                continue
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                metadata = self.extract_metadata(pdf_path, text)
                documents.append({
                    'text': text,
                    'metadata': metadata,
                    'file_path': pdf_path
                })
                logger.info(f"Processed PDF: {pdf_path}")
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}")
        return documents

    def process_stock_data(self, csv_path: str) -> Optional[pd.DataFrame]:
        if not self.validate_file_format(csv_path, ['.csv']):
            return None
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Processed CSV: {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error processing CSV {csv_path}: {e}")
            return None

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked = []
        for doc in documents:
            try:
                chunks = self.text_splitter.split_text(doc['text'])
                for i, chunk in enumerate(chunks):
                    chunked.append({
                        'text': chunk,
                        'metadata': {**doc['metadata'], 'chunk_index': i},
                        'file_path': doc['file_path']
                    })
                logger.info(f"Chunked document: {doc['file_path']} into {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error chunking document {doc['file_path']}: {e}")
        return chunked

    def extract_metadata(self, file_path: str, text: str) -> Dict[str, Any]:
        # Simple heuristics for metadata extraction
        metadata = {}
        basename = os.path.basename(file_path)
        # Example: Bajaj_Q1_2024.pdf
        if '_' in basename:
            parts = basename.replace('.pdf', '').split('_')
            if len(parts) >= 3:
                metadata['company'] = parts[0]
                metadata['quarter'] = parts[1]
                metadata['year'] = parts[2]
        # Try to extract date from text (very basic)
        import re
        date_match = re.search(r'(\d{1,2} \w+ \d{4})', text)
        if date_match:
            metadata['date'] = date_match.group(1)
        metadata['document_type'] = 'quarterly_transcript'
        return metadata 