import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama

load_dotenv()

# 1. Setup Retrieval
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("🔍 Loading Kaggle documents into memory...")
all_docs = vector_db.get()['documents']
bm25_retriever = BM25Retriever.from_texts(all_docs)

# 2. Local LLM (Generator)
llm = ChatOllama(model="llama3.2", temperature=0)

# 3. Define the Query
query = "Give me the step by step commands to download a dataset via the Kaggle API."
print(f"🤖 Researching: {query}...")

# 4. Get Hybrid Context (Manual)
# We pull more chunks (k=6) to ensure we don't miss the specific commands
v_results = vector_db.similarity_search(query, k=6)
b_results = bm25_retriever.invoke(query)[:6]
context_text = "\n\n".join([doc.page_content for doc in v_results + b_results])

# 5. Build the Final Prompt with the Context we just created
final_prompt = f"""You are a Kaggle assistant. Using the context below, 
provide the step-by-step commands to download a dataset. 
Include the 'kaggle datasets download' command if found.

Context:
{context_text}

Question: {query}
"""

# 6. Run the LLM
try:
    response = llm.invoke(final_prompt)
    print("\n--- FINAL LOCAL AI RESPONSE ---\n")
    print(response.content)
except Exception as e:
    print(f"❌ Error during generation: {e}")