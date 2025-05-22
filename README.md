# 📚 Intelligent Legal Assistant (RAG + LLM via Groq)

A GenAI-powered legal Q\&A assistant that answers questions based on uploaded PDF documents or legal URLs using **Retrieval-Augmented Generation (RAG)** and **LLMs via Groq**.

---

## 🚀 Features

* 📄 Upload legal documents (PDFs)
* 🌐 Paste URLs of legal blogs, case summaries, or statutes
* ❓ Ask natural language legal questions
* 🧠 Answers powered by **LLAMA3 (Groq API)**
* 🔍 Context-aware responses using **ChromaDB** and semantic retrieval
* 💡 Clean, interactive **Streamlit** UI

---

## 🧰 Tech Stack

| Component              | Purpose                                |
| ---------------------- | -------------------------------------- |
| **LangChain**          | RAG pipeline orchestration             |
| **ChromaDB**           | Vector store for legal document chunks |
| **Groq (LLAMA3)**      | Fast LLM-powered answer generation     |
| **HuggingFace MiniLM** | For generating embeddings              |
| **Streamlit**          | Frontend interface                     |
| **PyPDF2**             | PDF extraction                         |
| **BeautifulSoup4**     | URL scraping and HTML parsing          |
| **dotenv**             | Environment variable management        |

---

## 🧠 How It Works

1. **Document Ingestion**

   * Upload PDFs or enter a legal URL
   * Content is extracted using `PyPDF2` or `BeautifulSoup`

2. **Text Chunking**

   * Text is split using `RecursiveCharacterTextSplitter` to fit context limits

3. **Embedding + Indexing**

   * Each chunk is embedded using `MiniLM`
   * Chunks are stored in a local **ChromaDB** vector database

4. **Query Processing**

   * User asks a legal question
   * The query is embedded and matched to relevant document chunks

5. **Answer Generation**

   * Retrieved context is passed to **LLAMA3 via Groq API**
   * A precise, legally grounded answer is generated

---

## ⚙️ Setup Instructions

1. **Clone this repo**

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Groq API key**
   Create a `.env` file with:

   ```env
   GROQ_API_KEY=your_actual_groq_api_key
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 🛡️ Disclaimer

This project is for educational and prototyping purposes. It does not replace professional legal advice.

---

