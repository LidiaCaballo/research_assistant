import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Academic Research Assistant", layout="wide")
st.title("📚 Academic Research Assistant")
st.markdown("Query your research library using Hybrid Search (Vector + BM25)")

# --- APP LOGIC FUNCTIONS ---
@st.cache_resource 
def initialize_system():
    # 1. Setup Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Safety Check: Verify Database Folder
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        st.error(f"📁 Database folder not found. Please ensure 'chroma_db' is uploaded to GitHub.")
        st.stop()

    # 3. Load Vector Database
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # 4. Safety Check: Verify Database Content for BM25
    db_data = db.get()
    if not db_data or not db_data['documents']:
        st.warning("⚠️ The database appears to be empty. Ensure you ran ingest.py and pushed the data.")
        st.stop()

    # 5. Build Retrievers
    vector_retriever = db.as_retriever(search_kwargs={"k": 3})

    docs_for_bm25 = [
        Document(page_content=txt, metadata=meta) 
        for txt, meta in zip(db_data['documents'], db_data['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 3

    # 6. Hybrid Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

    # 7. LLM Setup (Hybrid Cloud/Local Logic)
    # Checks if running on Streamlit Cloud (via secrets) or locally
    if "OPENAI_API_KEY" in st.secrets:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            api_key=st.secrets["OPENAI_API_KEY"],
            temperature=0
        )
    else:
        # Fallback to local Ollama
        llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    # 8. Define Prompt Template
    template = """You are an Academic Research Assistant. Answer the question based ONLY on the following context.
    If the context doesn't contain the answer, say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 9. The Modern LCEL Chain
    # We keep the context separate so we can extract citations in the UI
    chain = (
        RunnableParallel({"context": ensemble_retriever, "question": RunnablePassthrough()})
        .assign(answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt 
            | llm 
            | StrOutputParser()
        ))
    )
    return chain

# --- UI LAYOUT ---
# Initialize system once (cached)
try:
    chain = initialize_system()

    query = st.text_input("Enter your research question:", placeholder="e.g., What are the main findings on fish immunity?")

    if query:
        with st.spinner("Analyzing research papers..."):
            try:
                result = chain.invoke(query)
                
                # Display Answer
                st.subheader("AI Response")
                st.write(result['answer'])
                
                # Display Citations in an Expander
                with st.expander("View Source Citations"):
                    sources = set()
                    for doc in result['context']:
                        file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        page = doc.metadata.get('page', 'Unknown')
                        sources.add(f"📄 **{file}** (Page {page})")
                    
                    if sources:
                        for s in sources:
                            st.write(s)
                    else:
                        st.write("No specific sources found.")
                        
            except Exception as e:
                st.error(f"Error during query execution: {e}")

except Exception as e:
    st.error(f"System Initialization Failed: {e}")