# Local RAG System for Kaggle Documentation

A production-grade Retrieval-Augmented Generation (RAG) pipeline designed to run entirely on local hardware. This project processes technical Kaggle CLI documentation and provides a conversational interface for technical queries without external API dependencies.

## 🚀 Key Features
- **Hybrid Retrieval**: Implements a manual hybrid search combining Semantic (Vector) and Keyword (BM25) search for high-precision retrieval of technical commands.
- **100% Local Processing**: Utilizes Llama 3.2 via Ollama and HuggingFace embeddings (`all-MiniLM-L6-v2`) to eliminate API costs and ensure data privacy.
- **Persistent Vector Store**: Uses ChromaDB for efficient storage and retrieval of embedded document chunks.
- **Optimized for Apple Silicon**: Configured for high-performance execution on Mac mini hardware.

## 🛠️ Tech Stack
- **Framework**: LangChain 0.3
- **LLM**: Llama 3.2 (3B) via Ollama
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace (Local)
- **Language**: Python 3.12

## 🔍 Engineering Challenges & Solutions
- **Dependency Resolution**: Resolved critical version conflicts between LangChain 0.3 and legacy packages by implementing a manual retrieval chain.
- **Retrieval Precision**: Tuned the system to overcome "Information not found" errors by broadening search parameters (k=6) and implementing keyword-based boosting.
- **Environment Stability**: Configured a specialized Python 3.12 virtual environment to manage complex AI library requirements on macOS.

## 📋 Quick Start
1. **Setup**: `pip install -r requirements.txt`
2. **Ingest**: `python3 ingest.py`
3. **Query**: `python3 rag_local.py`
