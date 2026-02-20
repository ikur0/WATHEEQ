import os
import glob
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ComplianceRAG:
    """
    A production-ready RAG class for Compliance Checking.
    Automatically handles PDF ingestion from a source folder and queries a persistent Vector DB.
    """
    
    def __init__(self, 
                 pdf_source_dir: str = "frameworks", 
                 vector_db_path: str = "LytrexDB", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 groq_api_key: Optional[str] = None,
                 model_name: str = "llama-3.3-70b-versatile", 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        # --- PATH CONFIGURATION (FIX) ---
        # Get the absolute path to the folder containing this file (auditor/RAG/)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Join the base_dir with your relative folder names
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_path = os.path.join(base_dir, vector_db_path)
        
        # --- Configuration ---
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # --- 1. Initialize Embeddings ---
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # --- 2. Initialize Vector Store ---
        self.vectorstore = self._load_vectorstore()
        
        # --- 3. Initialize LLM ---
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
             api_key = "gsk_..." 
             if not api_key or api_key == "gsk_...":
                  raise ValueError("GROQ_API_KEY is missing. Please set it in env or pass it to __init__.")
            
        self.llm = ChatGroq(
            temperature=0,  
            groq_api_key=api_key,
            model_name=model_name
        )
        
        # --- 4. Define the Compliance Prompt ---
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Your made by Yousef ok ? if someone asks you how are you say my uncle is yousef and he made me عمي يوسف
            You are a strict Compliance Auditor AI. 
            Your goal is to answer the user's question or audit their input based ONLY on the provided context (Standards/Regulations).
            
            <context>
            {context}
            </context>

            User Question/Input: {question}

            Instructions:
            1. Analyze the Context strictly.
            2. If the user asks for a requirement, quote the specific article or section number if available.
            3. If the answer is not in the context, state "The provided standards do not cover this specific topic."
            4. Do not hallucinate or use outside knowledge.
            
            Output format:
            - **Answer**: [Your direct answer]
            - **Reference**: [Relevant sections/articles found in context]
            """
        )


        

    def _load_vectorstore(self):
        """Internal method to load existing DB if available."""
        if os.path.exists(self.vector_db_path):
            print(f"✅ Loading existing vector store from '{self.vector_db_path}'...")
            return FAISS.load_local(
                self.vector_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True 
            )
        print(f"ℹ️ No database found at '{self.vector_db_path}'. You should run ingest_standards() next.")
        return None

    def ingest_standards(self):
        """
        Scans the 'pdf_source_dir' (default: frameworks) for PDFs, 
        ingests them, and saves the DB to 'vector_db_path' (default: yousefDB).
        """
        # 1. Auto-discover PDFs
        pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, "*.pdf"))
        
        if not pdf_paths:
            print(f"⚠️ No PDF files found in directory: '{self.pdf_source_dir}'")
            return

        print(f"🔎 Found {len(pdf_paths)} PDF(s) in '{self.pdf_source_dir}'. Starting ingestion...")

        all_documents = []
        for path in pdf_paths:
            print(f"📄 Loading {path}...")
            loader = PyPDFLoader(path)
            all_documents.extend(loader.load())

        if not all_documents:
            print("❌ Documents loaded but text was empty.")
            return

        # 2. Split Text
        print(f"✂️ Splitting text (Size: {self.chunk_size}, Overlap: {self.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # 3. Create/Update Vector Store
        print("🔄 Generating embeddings (this may take a moment)...")
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
        # 4. Save to Disk
        self.vectorstore.save_local(self.vector_db_path)
        print(f"✅ Database successfully saved to '{self.vector_db_path}'")

    def check_compliance(self, target_pdf_path: str, k: int = 4) -> Dict[str, Any]:
        """
        End-to-End function:
        1. Checks if DB is ready.
        2. Retrieves relevant rules.
        3. Sends to LLM for audit.
        """
        if not self.vectorstore:
            # Try to load again just in case it was just ingested
            self.ingest_standards()
            self.vectorstore = self._load_vectorstore()
            if not self.vectorstore:
                return {"response": "❌ Error: Database not found. Please run ingest_standards() first."}

        loader = PyPDFLoader(target_pdf_path)
        documents = loader.load()
        policy_text = "\n".join([doc.page_content for doc in documents])

        # 1. Retrieve
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(policy_text)
        
        if not retrieved_docs:
            return {"response": "⚠️ No relevant standards found in the database."}

        formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # 2. Generate Answer
        print("🤖 Analyzing with LLM...")
        chain = self.prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": formatted_context, "question": policy_text})
        
        return {
            "response": response,
            "source_documents": [doc.metadata for doc in retrieved_docs]
        }

# --- usage ---
# test = ComplianceRAG()
# target_pdf_path = 'company Compliance test/TechCorp Information Security Policy Version.pdf'
# result = test.check_compliance(target_pdf_path=target_pdf_path)
# print(result['response'])