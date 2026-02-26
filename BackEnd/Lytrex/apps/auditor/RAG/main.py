import os
import glob
from typing import Optional, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class ComplianceRAG:
    """
    A production-ready RAG class for Compliance Checking.
    Optimized for web backends to return structured JSON scoring.
    """
    
    def __init__(self, 
                 pdf_source_dir: str = "frameworks", 
                 vector_db_path: str = "LytrexDB", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 groq_api_key: Optional[str] = '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',
                 model_name: str = "llama-3.3-70b-versatile", 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_path = os.path.join(base_dir, vector_db_path)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = self._load_vectorstore()
        
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key or api_key.startswith("gsk_..."):
            raise ValueError("Valid GROQ_API_KEY is missing.")
            
        self.llm = ChatGroq(
            temperature=0,  
            groq_api_key=api_key,
            model_name=model_name
        )
        
        # --- DETAILED PROMPT ---
        self.detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite, strict Compliance Auditor AI developed by the Lytrex Team.
            Evaluate the <company_document> against the <framework_context>.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a comprehensive, highly detailed analysis. Quote specific article/section numbers.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "compliance_score": 85,
                "executive_summary": "A detailed 3-4 sentence summary.",
                "compliant_areas": ["List of precise things they did right"],
                "violations": ["List of specific breaches with framework section references"],
                "recommendations": ["Detailed actionable steps"]
            }}
            """
        )

        # --- CONCISE PROMPT (Saves Tokens) ---
        self.concise_prompt = ChatPromptTemplate.from_template(
            """
            You are a fast Compliance Auditor AI developed by the Lytrex Team.
            Evaluate the <company_document> against the <framework_context>.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a strictly brief, top-level overview. Do not over-explain.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "compliance_score": 85,
                "summary": "A strict 1-sentence summary.",
                "key_issues": ["Top 1-3 critical issues only"]
            }}
            """
        )
        
        self.output_parser = JsonOutputParser()

    def _load_vectorstore(self):
        if os.path.exists(self.vector_db_path):
            return FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    def ingest_standards(self):
        pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, "*.pdf"))
        if not pdf_paths: return
        all_documents = []
        for path in pdf_paths:
            all_documents.extend(PyPDFLoader(path).load())
        if not all_documents: return

        chunks = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap).split_documents(all_documents)
        
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.vector_db_path)

    def check_compliance(self, target_pdf_path: str, k: int = 5, detailed: bool = True) -> Dict[str, Any]:
        if not self.vectorstore:
            self.ingest_standards()
            self.vectorstore = self._load_vectorstore()
            if not self.vectorstore: return {"error": "Database not found."}

        documents = PyPDFLoader(target_pdf_path).load()
        target_chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(documents)
        full_company_text = "\n".join([doc.page_content for doc in documents])

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = []
        
        for chunk in target_chunks:
            retrieved_docs.extend(retriever.invoke(chunk.page_content))


        # Deduplicate retrieved standards    
        unique_docs = {doc.page_content: doc for doc in retrieved_docs}.values()
        formatted_context = "\n\n---\n\n".join(doc.page_content for doc in unique_docs)
        
        if not formatted_context: return {"error": "No relevant standards found."}

        # Select the prompt based on the 'detailed' flag
        active_prompt = self.detailed_prompt if detailed else self.concise_prompt
        chain = active_prompt | self.llm | self.output_parser
        
        try:
            return chain.invoke({
                "context": formatted_context, 
                "company_doc": full_company_text
            })
        except Exception as e:
            return {"error": f"Failed to parse LLM response: {str(e)}"}