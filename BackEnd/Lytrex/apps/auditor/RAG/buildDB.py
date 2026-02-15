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


def initialize_standards_db(file_paths, db_name="DB"):
    """
    Loads a list of PDF paths into the Vector Store ONCE.
    This database represents the unified 'Ground Truth' rules.
    """
    all_documents = []

    # 1. Loop through all paths and load them
    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ Warning: '{path}' not found in directory. Skipping.")
            continue

        print(f"Loading {path}...")
        loader = PyPDFLoader(path)
        documents = loader.load()
        all_documents.extend(documents)  # Add the pages to our master list

    if not all_documents:
        print("❌ Critical Error: No documents were loaded. Exiting.")
        return None

    # 2. Split all text at once
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)

    # 3. Embed and Store everything into a single FAISS index
    print("Generating embeddings and building FAISS index...")
    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save the unified database
    vectorstore.save_local(db_name)
    print(f"✅ Database successfully created and saved to '{db_name}' directory.")
    return vectorstore


# --- Execution ---
print("Start process")
framework_paths = [
    'frameworks/ECC--2024-EN.pdf',
    'frameworks/SAMA_EN_5888_VER1.pdf'
]

# Call the function ONCE with the list of paths
initialize_standards_db(framework_paths)