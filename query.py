import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# NEW: Imports for Hybrid Search
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

def query_system():
    # 2. Setup Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Load Vector Database
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    vector_retriever = db.as_retriever(search_kwargs={"k": 3})

    # --- NEW: INITIALIZE KEYWORD SEARCH (BM25) ---
    print("--- Initializing Hybrid Search ---")
    # We pull the text and metadata directly from your ChromaDB to build the keyword index
    db_data = db.get()
    docs_for_bm25 = [
        Document(page_content=txt, metadata=meta) 
        for txt, meta in zip(db_data['documents'], db_data['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 3

    # --- NEW: CREATE ENSEMBLE (HYBRID) RETRIEVER ---
    # This combines Vector (Meaning) + BM25 (Keywords)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

    # 4. Setup Local LLM
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    # 5. Academic Prompt (Forces the LLM to use context and stay grounded)
    template = """You are an Academic Research Assistant. Answer the question based ONLY on the following context.
    If the context doesn't contain the answer, say you don't know.

    Context:
    {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Build the "Pipe" with Citations
    # We use RunnableParallel so we can keep the 'context' docs to print citations later
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel({
            "context": ensemble_retriever, 
            "question": RunnablePassthrough()
        })
        .assign(answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        ))
    )

    # 7. Execute Query
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "What is the main conclusion of the research regarding fish stress?"
        
    print(f"\n--- Researching: {query} ---")
    
    try:
        result = rag_chain.invoke(query)
        
        # Print the AI's answer
        print(f"\nAI Response: {result['answer']}")
        
        # --- NEW: SOURCE VERIFICATION (CITATIONS) ---
        print("\nSOURCES CITED:")
        # We extract unique sources and pages from the retrieved chunks
        sources = set()
        for doc in result['context']:
            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 'Unknown')
            sources.add(f"- {source_file} (Page {page})")
        
        for s in sources:
            print(s)

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    query_system()