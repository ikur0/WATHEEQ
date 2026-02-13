import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from groq import Groq
# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
# Replace with your actual key
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
# Page Config
st.set_page_config(page_title="Smart Compliance", layout="wide")
st.title("🛡️ Smart Compliance Auditor")

# ----------------------------------------------------------------
# 1. SETUP RESOURCES
# ----------------------------------------------------------------

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0.0, # Zero temp for strict auditing
        model_name="llama-3.1-8b-instant"
    )

@st.cache_resource
def initialize_standards_db():
    DB_FOLDER_PATH = "DB"
    
    if not os.path.exists(DB_FOLDER_PATH):
        st.error(f"❌ The folder '{DB_FOLDER_PATH}' was not found. Please run the setup script first.")
        return None

    try:
        embeddings = get_embeddings_model()
        # allow_dangerous_deserialization is required for local pickle files
        vectorstore = FAISS.load_local(
            DB_FOLDER_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error loading Standards DB: {e}")
        return None

# Load the Standards DB immediately
if "standards_db" not in st.session_state:
    with st.spinner(""):
        st.session_state.standards_db = initialize_standards_db()

# ----------------------------------------------------------------
# 2. REPORT GENERATION LOGIC
# ----------------------------------------------------------------
def generate_compliance_report(policy_text, is_full_report):
    llm = get_llm()
    vectorstore = st.session_state.standards_db
    
    # -------------------------------------------------------
    # DYNAMIC RETRIEVAL LOGIC
    # -------------------------------------------------------
    # We use the USER'S POLICY TEXT as the query. 
    # This automatically finds the specific standards relevant to what the user wrote.
    # We fetch top 10 relevant chunks to ensure we catch all rules.
    docs = vectorstore.similarity_search(policy_text, k=10)
    
    # This is the "Ground Truth" context specific to the user's topic
    standards_context = "\n\n".join([doc.page_content for doc in docs])
    
    # -------------------------------------------------------
    # PROMPT ENGINEERING
    # -------------------------------------------------------
    if is_full_report:
        instructions = "Provide a detailed Gap Analysis, specific ISO/NIST mapping (if applicable), and specific recommendations."
    else:
        instructions = "Provide a Brief Summary of strengths and one major area for improvement."

    prompt = f"""
    You are a Strict Compliance Auditor. Your ONLY job is to compare the "User's Policy" against the "National Standards" provided in the CONTEXT.

    CONTEXT (National Standards - The Rules):
    {standards_context}

    USER INPUT (Company Policy - The Target):
    {policy_text}

    INSTRUCTIONS:
    1. IGNORE your general knowledge. Use ONLY the standards in the CONTEXT.
    2. For every algorithm mentioned in the User Input, check the CONTEXT for:
       - Allowed Key Lengths?
       - Allowed Zones (Moderate vs Advanced)?
    3. If the Policy mentions a justification (e.g., "for speed"), IGNORE IT. If the math does not match the standard, it is a VIOLATION.
    4. RSA must be 3072 bits minimum (if mentioned in Context).
    5. SOSEMANUK is NOT allowed for Advanced (if mentioned in Context).
    
    OUTPUT FORMAT:
    Create a Markdown Table with columns: [Policy Statement, Verdict, Reason].
    Then provide the {instructions}.
    """

    # Call LLM
    with st.spinner("🔍 Comparing User Policy against Framework..."):
        response = llm.invoke(prompt)
        return response.content

# ----------------------------------------------------------------
# 3. STREAMLIT UI
# ----------------------------------------------------------------


# Sidebar
with st.sidebar:
    st.header("⚙️ Audit Settings")
    st.write("Upload the **Company Policy** you want to check.")
    # This upload is the "Target"
    uploaded_file = st.file_uploader("Upload Policy (PDF)", type=["pdf"])
    
    st.markdown("---")
    full_report_mode = st.toggle("Generate Detailed Report", value=True)

# Main Area
if uploaded_file:
    # Extract text from the uploaded PDF immediately
    # We do NOT vectorise this file. We just read it as a string to query the Standards DB.
    try:
        reader = PdfReader(uploaded_file)
        policy_text = ""
        for page in reader.pages:
            policy_text += page.extract_text() or ""
            
        st.info(f"📄 Policy '{uploaded_file.name}' extracted successfully. ({len(policy_text)} characters)")
        
        # Action Button
        if st.button("📊 Run Compliance Check", type="primary"):
            if st.session_state.standards_db is None:
                st.error("Standards DB not loaded. Check file.pdf.")
            else:
                report = generate_compliance_report(policy_text, full_report_mode)
                st.markdown("### 📑 Audit Results")
                st.markdown(report)

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

else:
    st.info("👋 Upload a company policy PDF to audit it against the `file.pdf` standards.")