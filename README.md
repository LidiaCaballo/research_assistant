# Academic Research Assistant (Hybrid RAG)

A specialized Retrieval-Augmented Generation (RAG) system designed to process, index, and query academic research papers. This tool combines semantic vector search with keyword-based retrieval to provide highly accurate, cited answers from a private PDF library.

## Features
- **Hybrid Search:** Combines ChromaDB (Vector Search) and BM25 (Keyword Search) using an Ensemble Retriever for superior technical term recognition.
- **Source Verification:** Every answer includes specific PDF filenames and page numbers to eliminate hallucinations.
- **Local-First AI:** Powered by Ollama (Llama 3.2:1b) for 100% private, local processing.
- **Academic Grade Ingestion:** Processes 200+ pages with recursive character splitting to maintain context.
- **Interactive UI:** Built with Streamlit for a seamless research experience.

## Tech Stack
- **Framework:** LangChain (LCEL)
- **LLM:** Ollama (Local) / OpenAI (Cloud Fallback)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Frontend:** Streamlit

## Project Structure
- `ingest.py`: Handles PDF loading, text splitting, and vector database creation.
- `query.py`: Terminal-based testing script with support for CLI arguments.
- `app.py`: The main Streamlit web application.
- `pdfs/`: Directory for input research papers.
- `chroma_db/`: Persistent local vector storage.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/research-assistant.git](https://github.com/YOUR_USERNAME/research-assistant.git)
   cd research-assistant