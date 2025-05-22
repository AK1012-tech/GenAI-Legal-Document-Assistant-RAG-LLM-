# import streamlit as st
# from utils import extract_text_from_pdf, extract_text_from_url
# from ingest import ingest_text_to_vectorstore
# from rag_pipeline import load_qa_chain
# from dotenv import load_dotenv
# import os

# import torch
# import pandas as pd
# import requests
# import streamlit as st
# # Import required libraries

# torch.classes.__path__ = [] # add this line to manually set it to empty.

# load_dotenv()

# st.set_page_config(page_title="Intelligent Legal Assistant", layout="wide")
# st.title("📜 Intelligent Legal Assistant")

# qa_chain = None

# st.sidebar.header("Upload Legal Sources")

# # PDF Upload
# pdf_file = st.sidebar.file_uploader("Upload Legal PDF", type=["pdf"])
# # URL Input
# url_input = st.sidebar.text_input("Or enter a legal website URL")

# # Ingest button
# if st.sidebar.button("Ingest Source"):
#     if pdf_file:
#         text = extract_text_from_pdf(pdf_file)
#     elif url_input:
#         text = extract_text_from_url(url_input)
#     else:
#         st.sidebar.warning("Please upload a PDF or enter a URL.")
#         st.stop()
    
#     ingest_text_to_vectorstore(text)
#     qa_chain = load_qa_chain()
#     st.sidebar.success("Source ingested and vectorstore updated!")

# # Ask Questions
# st.subheader("Ask Your Legal Question")
# question = st.text_input("What do you want to know?")

# if st.button("Get Answer"):
#     if not question:
#         st.warning("Please enter a question.")
#     elif qa_chain:
#         answer = qa_chain.run(question)
#         st.success("Answer:")
#         st.write(answer)
#     else:
#         st.error("No legal documents ingested yet.")


import streamlit as st
from utils import extract_text_from_pdf, extract_text_from_url
from ingest import ingest_text_to_vectorstore
from rag_pipeline import load_qa_chain
from dotenv import load_dotenv
import os

import torch
import pandas as pd
import requests
import streamlit as st
# Import required libraries

torch.classes.__path__ = [] # add this line to manually set it to empty.

load_dotenv()

st.set_page_config(page_title="Intelligent Legal Assistant", layout="wide")
st.title("📜 Intelligent Legal Assistant")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.sidebar.header("Upload Legal Sources")

# PDF Upload
pdf_file = st.sidebar.file_uploader("Upload Legal PDF", type=["pdf"])
# URL Input
url_input = st.sidebar.text_input("Or enter a legal website URL")

# Ingest button
if st.sidebar.button("Ingest Source"):
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
    elif url_input:
        text = extract_text_from_url(url_input)
    else:
        st.sidebar.warning("Please upload a PDF or enter a URL.")
        st.stop()
    
    ingest_text_to_vectorstore(text)
    st.session_state.qa_chain = load_qa_chain()
    st.sidebar.success("Source ingested and vectorstore updated!")

# Ask Questions
st.subheader("Ask Your Legal Question")
question = st.text_input("What do you want to know?")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    elif st.session_state.qa_chain:
        answer = st.session_state.qa_chain.run(question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.error("No legal documents ingested yet.")
