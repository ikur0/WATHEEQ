import os
from typing import List, Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ComplianceRAG:
    """
    A production-ready RAG class for Compliance Checking with configurable chunking strategies.
    """
    def __init__(self, 
                 persist_directory: str = "frameworks", 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 groq_api_key: Optional[str] = "None",
                 model_name: str = "llama3-70b-8192",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        # Configuration
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 1. Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # 2. Initialize Vector Store (Load if exists, else None)
        self.vectorstore = self._load_vectorstore()
        
        # 3. Initialize LLM
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Please provide it or set it as an env variable.")
            
        self.llm = ChatGroq(
            temperature=0,  # Strict compliance checking
            groq_api_key=api_key,
            model_name=model_name
        )
        
        # 4. Define the Compliance Prompt
        self.prompt_template = ChatPromptTemplate.from_template(
            """
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
        if os.path.exists(self.persist_directory):
            print(f"✅ Loading existing vector store from '{self.persist_directory}'...")
            return FAISS.load_local(
                self.persist_directory, 
                self.embeddings, 
                allow_dangerous_deserialization=True 
            )
        print("ℹ️ No existing vector store found. Please run 'ingest_standards' first.")
        return None

    def ingest_standards(self, pdf_paths: List[str]):
        """
        Ingests a list of PDF paths into the vector store and saves it to disk.
        Uses the chunk_size and chunk_overlap defined in __init__.
        """
        all_documents = []
        
        for path in pdf_paths:
            if not os.path.exists(path):
                print(f"⚠️ File not found: {path}")
                continue
                
            print(f"📄 Loading {path}...")
            loader = PyPDFLoader(path)
            all_documents.extend(loader.load())

        if not all_documents:
            raise ValueError("No documents were loaded. Check your file paths.")

        # Split Text using instance configurations
        print(f"✂️ Splitting text (Size: {self.chunk_size}, Overlap: {self.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Create/Update Vector Store
        print("🔄 Generating embeddings (this may take a moment)...")
        if self.vectorstore:
            # If DB exists, add new docs to it
            self.vectorstore.add_documents(chunks)
        else:
            # If DB is new, create it
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
        # Save
        self.vectorstore.save_local(self.persist_directory)
        print(f"✅ Database saved to '{self.persist_directory}'")

    def check_compliance(self, query: str, k: int = 4) -> Dict[str, Any]:
        """
        The main Chain: Retrieves context and generates a compliance answer.
        """
        if not self.vectorstore:
            return {"error": "Vector store not initialized. Please ingest documents first."}

        # 1. Setup Retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # 2. Retrieve Documents
        retrieved_docs = retriever.invoke(query)
        formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # 3. Generate Answer
        chain = self.prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": formatted_context, "question": query})
        
        return {
            "response": response,
            "source_documents": [doc.metadata for doc in retrieved_docs]
        }
    












import os
import glob

def main():
    # --- Configuration ---
    frameworks_path = "frameworks"   # Folder where your PDFs are
    db_name = "yousefDB"             # The specific DB name you requested

    # 1. Find all PDF files in the frameworks directory
    pdf_files = glob.glob(os.path.join(frameworks_path, "*.pdf"))

    if not pdf_files:
        print(f"❌ No PDF files found in '{frameworks_path}'. Please check the path.")
        return

    print(f"🔎 Found {len(pdf_files)} PDF(s) to ingest: {pdf_files}")

    # 2. Initialize the RAG Class with your custom DB name
    # We pass 'yousefDB' as the persist_directory
    rag = ComplianceRAG(persist_directory=db_name)

    # 3. Ingest the found standards
    try:
        rag.ingest_standards(pdf_files)
        print(f"\n🎉 Success! All frameworks ingested and saved to '{db_name}'.")
        
        # Optional: Run a quick test query to ensure it works
        # print("\n--- Running Test Query ---")
        # result = rag.check_compliance("What are the requirements for password security?")
        # print(result['response'])

    except Exception as e:
        print(f"❌ An error occurred during ingestion: {e}")

if __name__ == "__main__":
    main()













## backend
# Initialize with custom chunking for specific document types (e.g., legal docs often need larger chunks)



# rag_engine = ComplianceRAG(
#     persist_directory="DB_Standards_Custom",
#     chunk_size=2000,    # Larger chunks to keep full articles together
#     chunk_overlap=400   # Larger overlap to ensure context isn't lost at edges
# )

# Ingest (uses the 2000/400 logic)
# rag_engine.ingest_standards(['frameworks/SAMA_EN_5888_VER1.pdf'])

# Run check
# result = rag_engine.check_compliance("What are the access control requirements?")