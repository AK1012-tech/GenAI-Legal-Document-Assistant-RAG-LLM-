from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_text_to_vectorstore(text, persist_directory="chroma_db"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

'''
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_text_to_vectorstore(text, persist_directory="chroma_db"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Use Legal-BERT model for embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb
'''
