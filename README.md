`**📚 Intelligent Legal Document Assistant (RAG + LLM)**`

A GenAI-powered assistant that answers legal questions from uploaded PDF contracts or legal website/blog URLs using Retrieval-Augmented Generation (RAG) and LLMs.

🚀 Features

1. Upload legal documents (PDFs)
2. Paste URLs of legal blogs or case summaries
3. Ask natural language legal questions
4. Answers powered by Hugging Face LLM (e.g., bloom-560m)
5. ChromaDB vector search with semantic similarity
6. Interactive Streamlit UI

🧰 Tech Stack

1. LangChain – orchestration of RAG pipeline
2. ChromaDB – vector store for document embeddings
3. Hugging Face Hub – free LLM for text generation (bigscience/bloom-560m)
4. HuggingFace Embeddings – MiniLM for semantic chunking
5. Streamlit – for frontend UI
6. PyPDF + BeautifulSoup – document and web scraping

🧠 How It Works

1. Document Ingestion: PDFs and webpages are loaded using PyPDFLoader or requests + BeautifulSoup.
2. Text Chunking: Split into chunks using RecursiveCharacterTextSplitter
3. Embedding + Indexing: Chunks are embedded using MiniLM and stored in ChromaDB.
4. Query Processing: The question is embedded and matched to chunks.
5. Answer Generation: Retrieved chunks are passed to an LLM to generate a contextual answer.

🖼️ UI Preview

![image](https://github.com/user-attachments/assets/e753817b-0640-4d2d-8579-57a0a2e5651a)

