# Local RAG System for Kaggle Documentation

A production-grade Retrieval-Augmented Generation (RAG) pipeline designed to run entirely on local hardware. This project processes technical Kaggle CLI documentation and provides a conversational interface for technical queries without external API dependencies.

## 🚀 Key Features
- **Hybrid Retrieval**: Implements a manual hybrid search combining Semantic (Vector) and Keyword (BM25) search for high-precision retrieval of technical commands.
- **Model Benchmarking**: Includes a custom telemetry suite to compare performance across different Small Language Models (SLMs).
- **100% Local Processing**: Utilizes Llama 3.2 and Mistral via Ollama to eliminate API costs and ensure data privacy.
- **Persistent Vector Store**: Uses ChromaDB for efficient storage and retrieval of embedded document chunks.

## 🛠️ Tech Stack
- **Framework**: LangChain 0.3
- **LLMs**: Llama 3.2 (3B), Mistral (7B) via **Ollama**
- **Vector DB**: ChromaDB
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Environment**: Python 3.12 (Mac mini / Apple Silicon)

## 📊 Performance Benchmarks (Project 2)
Tested on Mac mini (Apple Silicon) using a standardized technical query:

| Model | Response Time | Tokens Per Second (TPS) | Accuracy (1-10) |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 (3B)** | ~[Your Time]s | ~[Your TPS] | [Your Score] |
| **Mistral (7B)** | ~[Your Time]s | ~[Your TPS] | [Your Score] |

> *Note: Metrics were calculated using `benchmark.py` to measure 'Time to First Token' and generation speed.*

## 🔍 Engineering Challenges & Solutions
- **Dependency Resolution**: Resolved critical version conflicts between LangChain 0.3 and legacy packages by implementing a manual retrieval chain.
- **Retrieval Precision**: Tuned the system to overcome "Information not found" errors by broadening search parameters (k=6) and implementing keyword-based boosting.
- **Evaluation Framework**: Developed a "Judge" script using one LLM to grade the faithfulness of another, ensuring reliable RAG outputs.

## 📋 Quick Start
1. **Setup**: `pip install -r requirements.txt`
2. **Ingest**: `python3 ingest.py`
3. **Benchmark**: `python3 benchmark.py`
4. **Query**: `python3 rag_local.py`

