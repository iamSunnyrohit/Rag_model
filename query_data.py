import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
# Robust import for EnsembleRetriever
from langchain.retrievers.ensemble import EnsembleRetriever

load_dotenv()

# 1. Setup local embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the Vector DB we built in Step 1
vector_db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# 3. Create Hybrid Retriever
print("🔍 Setting up Hybrid Search (BM25 + Vector)...")
all_docs = vector_db.get()['documents']

if not all_docs:
    print("❌ Error: Your ChromaDB is empty. Run ingest.py first!")
else:
    bm25_retriever = BM25Retriever.from_texts(all_docs)
    bm25_retriever.k = 3

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Combine them
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

    # 4. Run the Query
    query = "How do I download a dataset using the Kaggle API?"
    docs = hybrid_retriever.invoke(query)

    print(f"\n--- Top Results from Kaggle Docs ---\n")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] {doc.page_content[:250]}...")
        print("-" * 30)