# Intelligent Legal Document Assistant (RAG + LLM with PDF & URL support)

# --- STEP 1: Setup and Installation ---
# Run this once in your terminal or Jupyter Notebook:
# !pip install langchain streamlit chromadb pypdf sentence-transformers requests beautifulsoup4 python-dotenv huggingface_hub

# --- STEP 2: Import Required Libraries ---
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tempfile

# --- STEP 3: Load API Key ---
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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
        try:
            with st.spinner("Loading PDF..."): # Added spinner
                loader = PyPDFLoader(temp_file.name)
                pdf_docs = loader.load()
                all_docs.extend(pdf_docs)
            st.success("PDF loaded successfully!") # Added success message
        except Exception as e:
            st.error(f"Failed to load PDF: {e}")
        finally:
            os.remove(temp_file.name) # Crucial: Clean up temp file

# --- STEP 6: Handle URL Input ---
if url_input:
    try:
        with st.spinner("Fetching URL content..."): # Added spinner
            response = requests.get(url_input, timeout=10) # Added timeout
            response.raise_for_status() # Raise an exception for bad status codes
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p') if p.get_text().strip() != ""]
            text = "\n".join(paragraphs)
            web_doc = Document(page_content=text, metadata={"source": url_input})
            all_docs.append(web_doc)
        st.success("URL content loaded successfully!") # Added success message
    except requests.exceptions.RequestException as e: # More specific exception
        st.error(f"Failed to retrieve content from URL: {e}. Please check the URL or your internet connection.")
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the URL: {e}")

# --- User guidance if no documents are loaded ---
if not all_docs:
    st.info("Please upload a PDF document or paste a URL to a legal article to get started.")

# --- STEP 7: Create Chroma Vector Store (Cached for efficiency) ---
# --- STEP 7: Create Chroma Vector Store (Cached for efficiency) ---
@st.cache_resource # Cache the embeddings and vectorstore creation
def get_vectorstore(_docs): # Changed 'docs' to '_docs'
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(_docs) # Use _docs here too
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Using a unique persist_directory based on content hash or a fixed one for simplicity
    # For a persistent store across runs, ensure the directory is outside the temp folder
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_store")


# --- STEP 8: Retrieval QA Chain with Free LLM (Cached for efficiency) ---
@st.cache_resource # Cache the LLM initialization
def get_llm(api_key):
    return HuggingFaceHub(
        repo_id="bigscience/bloom-560m",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
        huggingfacehub_api_token=api_key
    )

if all_docs:
    with st.spinner("Processing documents and preparing AI..."): # Spinner for processing
        vectorstore = get_vectorstore(all_docs)
        llm = get_llm(hf_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    st.success("AI assistant is ready!")

    # --- STEP 9: Run Query ---
    if query:
        if st.button("Get Answer"): # Added "Get Answer" button
            with st.spinner("Finding the answer..."): # Spinner for query
                response = qa_chain.run(query)
                st.markdown("### 🧠 Answer")
                st.write(response)
else:
    # If no documents are loaded, ensure query submission doesn't try to run
    if query:
        st.warning("Please load documents first before asking a question.")

# --- Optional: Cleanup or Logging ---