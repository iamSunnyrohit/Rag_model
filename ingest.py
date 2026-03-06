import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

load_dotenv()

def main():
    print("📂 Loading documents...")
    loader = DirectoryLoader(
        "./kaggle_docs", 
        glob="**/*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True}
    )
    docs = loader.load()

    print("✂️  Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    # Using a free, local model instead of OpenAI
    print("🧠 Generating local embeddings (Free)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Saving to ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings, # Using local embeddings
        persist_directory="./chroma_db"
    )
    print("🎉 Success! Your local vector database is ready.")

if __name__ == "__main__":
    main()