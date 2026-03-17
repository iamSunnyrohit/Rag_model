import streamlit as st
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. UI Configuration
st.set_page_config(page_title="Kaggle RAG Assistant", layout="wide")
st.title("🤖 Kaggle Local RAG Assistant")

with st.sidebar:
    st.sidebar.info("Running locally on Mac mini (Apple Silicon)")
    st.header("🖼️ Multimodal Input")
    uploaded_file = st.file_uploader("Upload a Kaggle screenshot...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.success("Image loaded! You can now ask questions about it.")

# 2. Load Local RAG Components
@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = ChatOllama(model="llama3.2", temperature=0)
    return vector_db, llm

vector_db, llm = load_rag()

# 3. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Kaggle CLI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic
    with st.chat_message("assistant"):
        context = vector_db.similarity_search(prompt, k=3)
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        full_prompt = f"Context: {context_text}\n\nQuestion: {prompt}"
        response = llm.invoke(full_prompt)
        
        st.markdown(response.content)
        with st.expander("View Retrieved Sources"):
            st.write(context_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response.content})