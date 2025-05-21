# Intelligent Legal Document Assistant (RAG + LLM with PDF & URL support)

# --- STEP 1: Setup and Installation ---
# Run this once in your terminal or Jupyter Notebook:
# !pip install --upgrade langchain langchain_community langchain_huggingface huggingface_hub pypdf sentence-transformers requests beautifulsoup4 python-dotenv streamlit

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
from langchain_huggingface import HuggingFaceEndpoint # Updated import for LLM
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tempfile

# --- STEP 3: Load API Key ---
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- STEP 4: Streamlit UI ---
st.title("📚 Intelligent Legal Document Assistant")

# Initialize session state for all_docs if not present
# This ensures that documents accumulate across Streamlit reruns
if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []

# Display API key warning early
if not hf_api_key:
    st.error("HuggingFace API key not found. Please set HUGGINGFACEHUB_API_TOKEN in your .env file.")
    st.stop() # Stop the app execution if the key is missing

uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf")
url_input = st.text_input("Or paste a link to a legal blog/article:")
query = st.text_input("Ask a legal question:")

# --- STEP 5: Handle PDF Upload ---
if uploaded_file:
    # Use a simple check to prevent re-adding the same file on every rerun if the app reruns
    if not any(doc.metadata.get("file_name") == uploaded_file.name for doc in st.session_state.all_docs):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            try:
                with st.spinner("Loading PDF..."):
                    loader = PyPDFLoader(temp_file.name)
                    pdf_docs = loader.load()
                    # Add original file name to metadata for tracking
                    for doc in pdf_docs:
                        doc.metadata["file_name"] = uploaded_file.name
                    st.session_state.all_docs.extend(pdf_docs) # Append to session state
                st.success(f"PDF '{uploaded_file.name}' loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load PDF: {e}")
            finally:
                os.remove(temp_file.name)
    else:
        st.info(f"PDF '{uploaded_file.name}' already loaded.")

# --- STEP 6: Handle URL Input ---
if url_input:
    # Check if URL content has already been added to avoid redundant processing
    if not any(doc.metadata.get("source") == url_input for doc in st.session_state.all_docs):
        try:
            with st.spinner("Fetching URL content..."):
                response = requests.get(url_input, timeout=10)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Improved text extraction to get more comprehensive content
                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                text = "\n".join([elem.get_text().strip() for elem in text_elements if elem.get_text().strip()])

                if text:
                    web_doc = Document(page_content=text, metadata={"source": url_input})
                    st.session_state.all_docs.append(web_doc) # Append to session state
                    st.success(f"URL content from '{url_input}' loaded successfully!")
                else:
                    st.warning(f"No significant text content found at {url_input}.")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to retrieve content from URL: {e}. Please check the URL or your internet connection.")
        except Exception as e:
            st.error(f"An unexpected error occurred while processing the URL: {e}")
    else:
        st.info(f"Content from '{url_input}' already loaded.")

# Display currently loaded documents for user awareness
if st.session_state.all_docs:
    st.markdown("---")
    st.subheader("Loaded Documents:")
    # Using a set to keep track of unique sources/file names to avoid duplicates in the display
    unique_sources = set()
    for doc in st.session_state.all_docs:
        source = doc.metadata.get("source") or doc.metadata.get("file_name")
        if source and source not in unique_sources:
            st.write(f"- {source}")
            unique_sources.add(source)
    st.markdown("---")
else:
    st.info("Please upload a PDF document or paste a URL to a legal article to get started.")

# --- STEP 7: Create Chroma Vector Store ---
@st.cache_resource
def get_vectorstore(_docs):
    """
    Creates and caches the Chroma vector store.
    The cache is cleared if the input documents (_docs) change.
    """
    if not _docs:
        return None # Return None if no documents are provided

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Note: For persistence across app restarts (e.g., if you close and reopen Streamlit),
    # you'd need to load from persist_directory first and then add new documents.
    # For this Streamlit app, @st.cache_resource handles caching for the current run.
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_store")

# --- STEP 8: Load a Supported LLM (Fixed) ---
@st.cache_resource
def get_llm(api_key):
    """
    Loads and caches the HuggingFaceEndpoint LLM.
    Using HuggingFaceEndpoint for better compatibility with newer Hugging Face API.
    """
    return HuggingFaceEndpoint(
        repo_id="google/flan-t5-small", # Consider larger models for better legal accuracy
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=api_key,
        # task="text2text-generation" # Often inferred, but can be specified if issues arise
    )

# Only proceed if documents are loaded and an API key is available
# Also, ensure 'qa_chain' is defined in the local scope for query processing
if st.session_state.all_docs and hf_api_key:
    with st.spinner("Processing documents and preparing AI..."):
        vectorstore = get_vectorstore(st.session_state.all_docs)
        if vectorstore:
            try:
                llm = get_llm(hf_api_key)
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
                st.session_state.qa_chain_ready = True # Set a flag in session state
                st.success("AI assistant is ready!")
            except Exception as e:
                st.error(f"Error initializing LLM or QA chain: {e}")
                st.session_state.qa_chain_ready = False
        else:
            st.warning("No documents processed into vector store. Please load valid documents.")
            st.session_state.qa_chain_ready = False
else:
    # This ensures the "AI assistant is ready!" message doesn't show prematurely
    st.session_state.qa_chain_ready = False


# --- STEP 9: Run Query ---
# Check the session state flag to determine if the QA chain is ready
if query:
    if st.session_state.qa_chain_ready:
        if st.button("Get Answer"):
            with st.spinner("Finding the answer..."):
                try:
                    # qa_chain is available in the scope due to the 'if st.session_state.all_docs...' block
                    response = qa_chain.run(query)
                    st.markdown("### 🧠 Answer")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
    else:
        st.warning("AI assistant is not ready. Please ensure documents are loaded and processed correctly.")