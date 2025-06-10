from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def load_qa_chain():
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa_chain


'''
def load_qa_chain():
    # Use Legal-BERT embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

    # Legal-specific robust prompt
    prompt_template = PromptTemplate.from_template("""
You are a legal assistant. Use the following excerpts to answer the user query. 
If no information is found, say: "The document does not contain relevant information."

Context: {context}
Query: {question}
Answer:
""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain
'''
