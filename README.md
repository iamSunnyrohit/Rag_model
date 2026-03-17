# Local RAG System for Kaggle Documentation

A production-grade, full-stack Retrieval-Augmented Generation (RAG) pipeline designed to run entirely on local hardware. This project processes technical Kaggle CLI documentation and provides a professional web-based chat interface for technical queries without external API dependencies.

## 🚀 Key Features
- **Interactive Web UI**: Built with **Streamlit** to provide a seamless, user-friendly chat experience.
- **Hybrid Retrieval**: Combines Semantic (Vector) and Keyword (BM25) search for high-precision retrieval of technical commands.
- **Model Benchmarking**: Custom telemetry suite comparing Llama 3.2 (3B) and Mistral (7B) performance on local hardware.
- **Observability**: Integrated **Arize Phoenix** for real-time tracing of the RAG lifecycle (Retrieval -> Generation).
- **100% Local**: No data leaves the machine; powered by Ollama and local HuggingFace embeddings.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Framework**: LangChain 0.3
- **LLMs**: Llama 3.2 (3B), Mistral (7B) via **Ollama**
- **Vector DB**: ChromaDB (Persistent)
- **Monitoring**: Arize Phoenix & OpenInference
- **Environment**: Python 3.12 (Apple Silicon / Mac mini)

## 📊 Performance & Monitoring
### Benchmarks (Mac mini)
| Model | Response Time | Tokens Per Second (TPS) | Accuracy |
| :--- | :--- | :--- | :--- |
| **Llama 3.2 (3B)** | 20.88s | 25.71 | 8/10 |
| **Mistral (7B)** | 26.86s | 12.73 | 9/10 |

### Tracing
The system uses **Arize Phoenix** to visualize the internal logic flow. You can inspect exactly which chunks from the documentation were retrieved to ground the LLM's response.

## 🔍 Engineering Challenges
- **Environment Management**: Resolved conflicts between global Anaconda installs and project-specific virtual environments using targeted module execution.
- **Retrieval Accuracy**: Implemented manual hybrid search to bypass library version conflicts while maintaining high-precision CLI command retrieval.

## 📋 Quick Start
1. **Setup**: `pip install -r requirements.txt`
2. **Ingest Data**: `python3 ingest.py`
3. **Launch UI**: `python3 -m streamlit run app.py`
4. **Monitoring**: View traces at `http://localhost:6006`
