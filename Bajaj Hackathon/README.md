# Bajaj Finserv RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for analyzing Bajaj Finserv quarterly transcripts and stock data using Llama 3 (Groq API), ChromaDB, LangChain, and Streamlit.

## Features
- Analyze quarterly PDF transcripts
- Analyze stock price CSV data
- Financial analysis prompts (stock price trends, business insights, CFO commentary, comparative analysis)
- Llama 3 via Groq API
- ChromaDB for vector storage
- LangChain for RAG pipeline
- Streamlit web interface

## Setup Instructions

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Copy `.env_sample` to `.env` and add your Groq API key:
     ```bash
     cp .env_sample .env
     ```
   - Edit `.env` and set `GROQ_API_KEY`.

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## Example Questions
- What was the highest stock price in Q1 2024?
- Summarize the CFO's commentary for the last quarter.
- Compare business performance between Q2 and Q3.
- What are the key business insights from the latest transcript? 