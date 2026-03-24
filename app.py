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
@st.cache_resource # This prevents the app from reloading the brain every time you click a button
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 1. Vector Retriever
    vector_retriever = db.as_retriever(search_kwargs={"k": 3})

    # 2. BM25 Retriever (Keywords)
    db_data = db.get()
    docs_for_bm25 = [
        Document(page_content=txt, metadata=meta) 
        for txt, meta in zip(db_data['documents'], db_data['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 3

    # 3. Hybrid Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    
    template = """You are an Academic Research Assistant. Answer the question based ONLY on the following context.
    If the context doesn't contain the answer, say you don't know.
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The modern LCEL Chain
    chain = (
        RunnableParallel({"context": ensemble_retriever, "question": RunnablePassthrough()})
        .assign(answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt | llm | StrOutputParser()
        ))
    )
    return chain

# --- UI LAYOUT ---
chain = initialize_system()

query = st.text_input("Enter your research question:", placeholder="e.g., What are the effects of Bacillus subtilis?")

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
                
                for s in sources:
                    st.write(s)
                    
        except Exception as e:
            st.error(f"Error processing query: {e}")