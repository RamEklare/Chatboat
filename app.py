import streamlit as st
from document_processor import extract_pdf_chunks
from faiss_indexer import FAISSIndexer
from llama_local import LocalLLaMAModel
from rag_chatbot import RAGChatbot

@st.cache_resource
def initialize_chatbot(pdf_path):
    chunks = extract_pdf_chunks(r"C:\Users\HP\bajaj\bajaj_finserv_factsheet_Oct.pdf")
    indexer = FAISSIndexer()
    indexer.build_index(chunks)
    llm = LocalLLaMAModel(model_path=r"C:\Users\HP\Bajaj_techno\llama-3-8b.gguf")  # Update with your local llama model path
    chatbot = RAGChatbot(indexer, llm)
    return chatbot

st.title("Bajaj AMC Fund Factsheet RAG Chatbot (Fully Local LLaMA)")
uploaded_file = st.file_uploader("Upload Bajaj AMC Fund Factsheet PDF", type="pdf")

if uploaded_file:
    with open("bajaj_finserv_factsheet_Oct.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    chatbot = initialize_chatbot("bajaj_finserv_factsheet_Oct.pdf")

    if "history" not in st.session_state:
        st.session_state.history = []

    for speaker, message in st.session_state.history:
        st.chat_message(speaker).markdown(message)

    query = st.chat_input("Ask a question about the factsheet")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.history.append(("user", query))

        answer = chatbot.chat(query)
        st.chat_message("assistant").markdown(answer)
        st.session_state.history.append(("assistant", answer))

else:
    st.write("Please upload the Bajaj AMC Fund Factsheet PDF to start.")
