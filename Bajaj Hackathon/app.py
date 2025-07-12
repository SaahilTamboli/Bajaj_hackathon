import os
import sys
import streamlit as st
import pandas as pd
import logging
import time
import traceback
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from groq_llm import GroqLLM
from rag_pipeline import RAGPipeline
from query_router import QueryRouter
from config import AppConfig, ConfigError
import datetime
import io
import atexit
import shutil

# --- Load environment variables and configure logging early ---
try:
    config = AppConfig.from_env()
except ConfigError as ce:
    st.error(f"Configuration Error: {ce}")
    st.stop()

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Health Checks ---
def health_check():
    health = {}
    # Check Groq API
    try:
        llm = GroqLLM(api_key=config.GROQ_API_KEY, model=config.MODEL_NAME)
        _ = llm.get_usage()
        health['llm'] = True
    except Exception as e:
        logger.error(f"LLM Health Check Failed: {e}")
        health['llm'] = False
    # Check vector DB
    try:
        vectorstore = VectorStoreManager(persist_directory=config.VECTOR_DB_PATH)
        health['vector_db'] = True
    except Exception as e:
        logger.error(f"Vector DB Health Check Failed: {e}")
        health['vector_db'] = False
    # Check document processor
    try:
        _ = DocumentProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        health['doc_processor'] = True
    except Exception as e:
        logger.error(f"Document Processor Health Check Failed: {e}")
        health['doc_processor'] = False
    return health

# --- Performance Monitoring ---
def log_performance(start_time, label="Operation"):
    elapsed = time.time() - start_time
    logger.info(f"{label} took {elapsed:.2f} seconds.")

# --- Resource Cleanup ---#
def cleanup():
    shutil.rmtree(".tmp", ignore_errors=True)
    logger.info("Cleaned up temporary files.")
atexit.register(cleanup)

# --- Streamlit App Config ---#
st.set_page_config(
    page_title="Bajaj Finserv Financial Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---#
def init_session_state():
    defaults = {
        "processing_status": "Not started",
        "vectorstore_ready": False,
        "chat_history": [],
        "analysis_results": [],
        "api_usage": {"total_tokens": 0, "total_cost": 0.0},
        "user_input": "",
        "router": QueryRouter(),
        "llm": None,
        "rag": None,
        "vectorstore": None,
        "processor": None,
        "health": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session_state()

# --- Health Check on Startup ---#
if not st.session_state["health"]:
    st.session_state["health"] = health_check()
    if not all(st.session_state["health"].values()):
        st.error(f"Startup health check failed: {st.session_state['health']}")
        st.stop()

# --- Sidebar: Settings Only ---#
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
model_choice = st.sidebar.selectbox(
    "LLM Model", ["llama3-8b-8192", "llama3-70b-8192"], index=0
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, config.TEMPERATURE, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 256, 4096, config.MAX_TOKENS, 128)

# --- Main: Dashboard Layout ---#
st.title("💼 Bajaj Finserv Financial Analysis Chatbot")
st.markdown(
    """
    <style>
    .stChatMessage { background: #f8f9fa; border-radius: 8px; padding: 8px; margin-bottom: 8px; }
    .stChatUser { color: #1a237e; font-weight: bold; }
    .stChatBot { color: #388e3c; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Query Suggestions ---#
st.markdown("#### 💡 Example Questions:")
col1, col2, col3 = st.columns(3)
with col1:
    st.button("What was the highest stock price in Q2 2024?", key="ex1")
with col2:
    st.button("Why is BAGIC facing headwinds in Motor insurance?", key="ex2")
with col3:
    st.button("Compare performance from Jan-24 to Mar-24", key="ex3")

# --- Document Processing ---#
def process_documents_from_folder(folder_path):
    st.session_state.processing_status = "Processing..."
    st.info(f"Processing documents from '{folder_path}'. Please wait...")
    start_time = time.time()
    processor = DocumentProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    pdf_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    csv_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    docs = processor.process_quarterly_transcripts(pdf_paths)
    chunks = processor.chunk_documents(docs)
    stock_data = []
    for csv_path in csv_paths:
        df = processor.process_stock_data(csv_path)
        if df is not None:
            stock_data.append({"text": df.to_csv(index=False), "metadata": {"file_path": csv_path, "document_type": "stock_data"}})
    st.session_state.processing_status = "Processed"
    log_performance(start_time, label="Document Processing")
    return chunks, stock_data

# --- Vector Store Setup ---#
def setup_vectorstore(chunks, stock_data):
    start_time = time.time()
    vectorstore = VectorStoreManager(persist_directory=config.VECTOR_DB_PATH)
    vectorstore.create_vectorstore()
    if chunks:
        vectorstore.add_documents(chunks, doc_type="transcript")
    if stock_data:
        vectorstore.add_documents(stock_data, doc_type="stock_data")
    st.session_state.vectorstore_ready = True
    log_performance(start_time, label="Vector Store Setup")
    return vectorstore

# --- LLM and RAG Pipeline Setup ---#
def setup_llm_and_rag(vectorstore):
    llm = GroqLLM(api_key=config.GROQ_API_KEY, model=model_choice, temperature=temperature, max_tokens=max_tokens)
    rag = RAGPipeline(vectorstore, llm)
    return llm, rag

# --- Process and Index Documents from 'documents' folder if Needed ---#
documents_folder = config.DATA_DIR
vectorstore = VectorStoreManager(persist_directory=config.VECTOR_DB_PATH)
if not vectorstore.is_populated():
    with st.spinner(f"Processing and indexing documents from '{documents_folder}'..."):
        os.makedirs(".tmp", exist_ok=True)
        try:
            chunks, stock_data = process_documents_from_folder(documents_folder)
            vectorstore.create_vectorstore()
            if chunks:
                vectorstore.add_documents(chunks, doc_type="transcript")
            if stock_data:
                vectorstore.add_documents(stock_data, doc_type="stock_data")
            llm, rag = setup_llm_and_rag(vectorstore)
            st.session_state["llm"] = llm
            st.session_state["rag"] = rag
            st.session_state["vectorstore"] = vectorstore
            st.session_state["processor"] = DocumentProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            st.session_state.vectorstore_ready = True
        except Exception as e:
            logger.error(f"Error during document processing/indexing: {e}\n{traceback.format_exc()}")
            st.error(f"Error during document processing/indexing: {e}")
            st.stop()
else:
    try:
        llm, rag = setup_llm_and_rag(vectorstore)
        st.session_state["llm"] = llm
        st.session_state["rag"] = rag
        st.session_state["vectorstore"] = vectorstore
        st.session_state["processor"] = DocumentProcessor(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        st.session_state.vectorstore_ready = True
    except Exception as e:
        logger.error(f"Error during vectorstore/LLM setup: {e}\n{traceback.format_exc()}")
        st.error(f"Error during vectorstore/LLM setup: {e}")
        st.stop()

# --- Chat Interface ---#
st.markdown("---")
st.header("💬 Chat with the Financial Analyst Bot")

user_input = st.text_input("Ask a question about Bajaj Finserv...", value=st.session_state.user_input, key="chat_input")

if st.button("Send", key="send_btn") or st.session_state.get("ex1") or st.session_state.get("ex2") or st.session_state.get("ex3"):
    if st.session_state.get("ex1"):
        user_input = "What was the highest stock price in Q2 2024?"
    elif st.session_state.get("ex2"):
        user_input = "Why is BAGIC facing headwinds in Motor insurance?"
    elif st.session_state.get("ex3"):
        user_input = "Compare performance from Jan-24 to Mar-24"
    st.session_state.user_input = user_input
    if not st.session_state["rag"]:
        st.error("Please process documents first.")
    elif not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Use QueryRouter for intent/entity extraction and prompt building
                router = st.session_state["router"]
                route_info = router.route(user_input, st.session_state.chat_history)
                intent = route_info["intent"]
                entities = route_info["entities"]
                prompt = route_info["prompt"]
                # Use RAG pipeline for answer
                result = st.session_state["rag"].ask(user_input, chat_history=st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "bot", "content": result["answer"]})
                st.session_state.analysis_results.append(result)
                st.session_state.api_usage = st.session_state["llm"].get_usage()
                logger.info(f"User Query: {user_input} | Intent: {intent} | Entities: {entities}")
            except Exception as e:
                logger.error(f"Error during chat: {e}\n{traceback.format_exc()}")
                st.error(f"Error during chat: {e}")

# --- Display Chat History ---#
st.markdown("---")
st.subheader("Conversation History")
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f"<div class='stChatMessage stChatUser'>🧑‍💼 <b>User:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage stChatBot'>🤖 <b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)

# --- Results Visualization ---#
if st.session_state.analysis_results:
    st.markdown("---")
    st.subheader("Analysis Results & Visualizations")
    last_result = st.session_state.analysis_results[-1]
    st.markdown(f"**Query Type:** {last_result['query_type'].replace('_', ' ').title()}")
    st.markdown(f"**Confidence Score:** {last_result['confidence']:.2f}")
    st.markdown("**Source Documents:**")
    for src in last_result["sources"]:
        st.code(src)
    # Try to render tables if present in answer
    if "|" in last_result["answer"] and "---" in last_result["answer"]:
        st.markdown(last_result["answer"])

# --- API Usage Statistics ---#
st.sidebar.markdown("---")
st.sidebar.subheader("API Usage Statistics")
st.sidebar.markdown(f"**Total Tokens Used:** {st.session_state.api_usage['total_tokens']}")
st.sidebar.markdown(f"**Total Cost:** ${st.session_state.api_usage['total_cost']:.4f}")

# --- Download/Export Functionality ---#
if st.session_state.analysis_results:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Analysis Results")
    export_df = pd.DataFrame(st.session_state.analysis_results)
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"bajaj_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# --- Document Processing Status ---#
st.sidebar.markdown("---")
st.sidebar.subheader("Document Processing Status")
st.sidebar.markdown(f"**Status:** {st.session_state.processing_status}") 