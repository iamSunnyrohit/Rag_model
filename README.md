# Local RAG System for Kaggle Documentation

A production-grade Retrieval-Augmented Generation (RAG) pipeline designed to run entirely on local hardware. This project processes technical Kaggle CLI documentation and provides a conversational interface for technical queries without external API dependencies or data privacy concerns.

## 🚀 Key Features
- **Hybrid Retrieval**: Implements a manual hybrid search combining Semantic (Vector) and Keyword (BM25) search for high-precision retrieval of technical commands.
- **Model Benchmarking**: Custom telemetry suite to compare performance across different Small Language Models (SLMs).
- **Observability**: Integrated **Arize Phoenix** for real-time tracing of the RAG lifecycle (Retrieval -> Generation).
- **100% Local Processing**: Utilizes Llama 3.2 and Mistral via Ollama to eliminate API costs.

## 🛠️ Tech Stack
- **Framework**: LangChain 0.3
- **LLMs**: Llama 3.2 (3B), Mistral (7B) via **Ollama**
- **Vector DB**: ChromaDB (Persistent)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Monitoring**: Arize Phoenix & OpenInference
- **Environment**: Python 3.12 (Apple Silicon / Mac mini)

## 📊 Performance Benchmarks
Tested on local hardware (Mac mini) using standardized technical queries:

| Model | Response Time | Tokens Per Second (TPS) | Accuracy (1-10) |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 (3B)** | 20.88s | **25.71** | 8 |
| **Mistral (7B)** | 26.86s | 12.73 | **9** |

## 🕵️ Observability & Traceability
To ensure system transparency, I instrumented the pipeline with **Arize Phoenix**:
- **Operation Tracing**: Visualized the sequence of events from user query to final LLM response.
- **Span Analysis**: Measured individual latencies for the BM25 and Vector retrieval steps.
- **Local MLOps**: Used OpenInference instrumentation to capture system telemetry without external cloud dependencies.

## 🔍 Engineering Challenges & Solutions
- **Dependency Resolution**: Resolved critical version conflicts between LangChain 0.3 and legacy packages by implementing a manual retrieval chain.
- **Retrieval Precision**: Overcame "Information not found" errors by broadening search parameters (k=6) and implementing keyword-based boosting.
- **Evaluation Framework**: Developed a "Judge" script using one LLM to grade the faithfulness of another to ensure grounded responses.

## 📋 Quick Start
1. **Clone & Setup**: `pip install -r requirements.txt`
2. **Data Ingestion**: `python3 ingest.py`
3. **Run Monitoring**: `python3 monitoring.py` (Access dashboard at `localhost:6006`)
4. **Query System**: `python3 rag_local.py`
