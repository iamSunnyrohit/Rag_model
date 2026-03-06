from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

query = "download dataset kaggle api"
results = vector_db.similarity_search(query, k=3)

print(f"--- Top 3 Results for '{query}' ---")
for i, res in enumerate(results):
    print(f"\nResult {i+1}:")
    print(res.page_content[:400])