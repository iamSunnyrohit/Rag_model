# Local RAG System for Kaggle Documentation

A production-grade Retrieval-Augmented Generation (RAG) pipeline built to run entirely on local hardware. This project processes Kaggle CLI documentation and provides a conversational interface for technical queries without external API dependencies.

## 🚀 Key Features
- **Hybrid Search**: Combines Semantic Search (Vector) and Keyword Search (BM25) for high-precision retrieval of technical commands.
- **100% Local**: Uses HuggingFace embeddings and Llama 3.2 (via Ollama) to ensure data privacy and zero API costs.
- **Optimized Chunking**: Implements recursive character splitting to maintain context for complex technical documentation.
- **Hardware Optimized**: Specifically configured for Apple Silicon (Mac mini) performance.

## 🛠️ Tech Stack
- **Orchestration**: LangChain 0.3
- **LLM**: Llama 3.2 (3B) via Ollama
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Environment**: Python 3.12

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   
