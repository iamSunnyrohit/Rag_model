from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

query = "How do I download a dataset using the Kaggle API?"
docs = vector_db.similarity_search(query, k=2)

for doc in docs:
    print(f"\n--- Found Chunk ---\n{doc.page_content[:300]}...")