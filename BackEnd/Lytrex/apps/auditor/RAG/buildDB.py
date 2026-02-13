import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def initialize_standards_db():
    """
    Loads 'file.pdf' (National Standards) into the Vector Store ONCE.
    This database represents the 'Ground Truth' rules.
    """
    file_path = "frameworks/file.pdf"
    
    if not os.path.exists(file_path):
        print(f"❌ Critical Error: '{file_path}' not found in directory. Please add the Standards PDF.")
        return None

    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split text (Keeping chunks small-ish to find specific rules)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Embed and Store
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("DB")
    return vectorstore


print("Start process")
initialize_standards_db()
print("Supposed to be stored successfully")