📚 Intelligent Legal Document Assistant (RAG + LLM)

A GenAI-powered assistant that answers legal questions from uploaded PDF contracts or legal website/blog URLs using Retrieval-Augmented Generation (RAG) and LLMs.

🚀 Features

✅ Upload legal documents (PDFs)

✅ Paste URLs of legal blogs or case summaries

✅ Ask natural language legal questions

✅ Answers powered by Hugging Face LLM (e.g., bloom-560m)

✅ ChromaDB vector search with semantic similarity

✅ Interactive Streamlit UI

🧰 Tech Stack

LangChain – orchestration of RAG pipeline

ChromaDB – vector store for document embeddings

Hugging Face Hub – free LLM for text generation (bigscience/bloom-560m)

HuggingFace Embeddings – MiniLM for semantic chunking

Streamlit – for frontend UI

PyPDF + BeautifulSoup – document and web scraping

📦 Installation

pip install -r requirements.txt

Create a .env file:

HUGGINGFACEHUB_API_TOKEN=your_token_here

▶️ Usage

streamlit run app.py

Then:

Upload a legal PDF or paste a blog/article URL

Type a legal question

Get an AI-generated response based on document context

🧠 How It Works

Document Ingestion: PDFs and webpages are loaded using PyPDFLoader or requests + BeautifulSoup.

Text Chunking: Split into chunks using RecursiveCharacterTextSplitter

Embedding + Indexing: Chunks are embedded using MiniLM and stored in ChromaDB.

Query Processing: The question is embedded and matched to chunks.

Answer Generation: Retrieved chunks are passed to an LLM to generate a contextual answer.

🖼️ UI Preview
![image](https://github.com/user-attachments/assets/e753817b-0640-4d2d-8579-57a0a2e5651a)

