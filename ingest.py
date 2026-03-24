import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load API Key (Create a .env file with OPENAI_API_KEY=your_key)
load_dotenv()

def build_vector_db():
    # 1. Load all PDFs from the /pdfs folder
    print("--- Loading Documents ---")
    loader = PyPDFDirectoryLoader("pdfs/")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDFs.")

    # 2. Split text into manageable chunks
    # We use RecursiveCharacterTextSplitter to keep paragraphs together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Create Embeddings and Store in ChromaDB
    print("--- Creating Vector Database ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # This creates a local folder named 'chroma_db' to persist your data
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("--- Done! Database saved to ./chroma_db ---")
    return vector_db

if __name__ == "__main__":
    build_vector_db()