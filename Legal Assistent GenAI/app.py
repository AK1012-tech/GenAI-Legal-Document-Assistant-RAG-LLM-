# Intelligent Legal Document Assistant (RAG + LLM with PDF & URL support)

# --- STEP 1: Setup and Installation ---
# Run this once in your terminal or Jupyter Notebook:
# !pip install langchain openai streamlit chromadb pypdf sentence-transformers requests beautifulsoup4 python-dotenv

# --- STEP 2: Import Required Libraries ---
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tempfile

# --- STEP 3: Load API Key ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- STEP 4: Streamlit UI ---
st.title("📚 Intelligent Legal Document Assistant")

uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf")
url_input = st.text_input("Or paste a link to a legal blog/article:")
query = st.text_input("Ask a legal question:")

all_docs = []

# --- STEP 5: Handle PDF Upload ---
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        loader = PyPDFLoader(temp_file.name)
        pdf_docs = loader.load()
        all_docs.extend(pdf_docs)

# --- STEP 6: Handle URL Input ---
if url_input:
    try:
        response = requests.get(url_input)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p') if p.get_text().strip() != ""]
        text = "\n".join(paragraphs)
        web_doc = Document(page_content=text, metadata={"source": url_input})
        all_docs.append(web_doc)
    except Exception as e:
        st.error(f"Failed to retrieve content from URL: {e}")

# --- STEP 7: Create Chroma Vector Store ---
if all_docs:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chroma_store")

    # --- STEP 8: Retrieval QA Chain ---
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # --- STEP 9: Run Query ---
    if query:
        response = qa_chain.run(query)
        st.markdown("### 🧠 Answer")
        st.write(response)

# --- Optional: Cleanup or Logging ---
