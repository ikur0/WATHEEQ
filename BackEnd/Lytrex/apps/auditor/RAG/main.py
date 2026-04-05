import os
import glob
import re
import json
from typing import Optional, Dict, Any, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class ComplianceRAG:
    SUPPORTED_FRAMEWORKS = ["NCA", "ECC", "SAMA"]

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
        self.vector_db_base_path = os.path.join(base_dir, vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstores: Dict[str, Any] = self._load_all_vectorstores()

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it as an environment variable.")

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name=model_name
        )

        self._setup_prompts()
        self.output_parser = JsonOutputParser()

    def _setup_prompts(self):
        self.detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite, strict Compliance Auditor AI developed by the Lytrex Team.
            The framework being evaluated is: {framework_name}

            <framework_context>
            {context}
            </framework_context>

            <company_document_chunks>
            {company_doc}
            </company_document_chunks>

            TASK:
            1. Determine if the document chunks are relevant to {framework_name} compliance.
            2. If NOT relevant, set "is_relevant" to false, "compliance_score" to 0, and explain why in "executive_summary".
            3. If IS relevant, set "is_relevant" to true and provide analysis based ONLY on the provided chunks.

            Respond ONLY with valid JSON using this exact schema (Do NOT add an "error" key or any unlisted keys):
            {{
                "framework": "{framework_name}",
                "is_relevant": true,
                "compliance_score": 85,
                "executive_summary": "Summary text.",
                "compliant_areas": [],
                "violations": [],
                "recommendations": []
            }}
            """
        )

        self.concise_prompt = ChatPromptTemplate.from_template(
            """
            You are a fast Compliance Auditor AI.
            Evaluate against {framework_name}.
            <framework_context>{context}</framework_context>
            <company_document_chunks>{company_doc}</company_document_chunks>

            TASK: 
            1. Check relevance. 
            2. If NOT relevant, set "is_relevant" to false, "compliance_score" to 0, and explain why in "summary".

            Respond ONLY with valid JSON using this exact schema (Do NOT add an "error" key):
            {{
                "framework": "{framework_name}",
                "is_relevant": true,
                "compliance_score": 85,
                "summary": "Summary text.",
                "key_issues": []
            }}
            """
        )

    def _framework_db_path(self, framework: str) -> str:
        return os.path.join(self.vector_db_base_path, framework)

    def _framework_db_exists(self, framework: str) -> bool:
        path = self._framework_db_path(framework)
        return os.path.exists(os.path.join(path, "index.faiss"))

    def _load_all_vectorstores(self) -> Dict[str, Any]:
        stores = {}
        for fw in self.SUPPORTED_FRAMEWORKS:
            if self._framework_db_exists(fw):
                try:
                    stores[fw] = FAISS.load_local(self._framework_db_path(fw), self.embeddings, allow_dangerous_deserialization=True)
                except Exception: stores[fw] = None
            else: stores[fw] = None
        return stores

    def _ingest_single_framework(self, framework: str) -> bool:
        fw_dir = os.path.join(self.pdf_source_dir, framework)
        pdf_paths = glob.glob(os.path.join(fw_dir, "*.pdf"))
        if not pdf_paths: return False

        all_docs = []
        for p in pdf_paths: all_docs.extend(PyPDFLoader(p).load())
        
        chunks = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap).split_documents(all_docs)
        db = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstores[framework] = db
        db.save_local(self._framework_db_path(framework))
        return True

    def _ensure_framework_ready(self, framework: str) -> bool:
        if self.vectorstores.get(framework): return True
        if self._framework_db_exists(framework):
            try:
                self.vectorstores[framework] = FAISS.load_local(self._framework_db_path(framework), self.embeddings, allow_dangerous_deserialization=True)
                return True
            except: pass
        return self._ingest_single_framework(framework)

    def _check_single_framework(self, framework: str, company_db: FAISS, k: int, detailed: bool) -> Dict[str, Any]:
        if not self._ensure_framework_ready(framework):
            return {"framework": framework, "error": f"Standards not found for {framework}."}

        # --- 1. RAG on the UPLOADED Document ---
        company_retriever = company_db.as_retriever(search_kwargs={"k": 15})
        query = f"{framework} compliance, cybersecurity policy, access control, data protection, risk management, incident response"
        company_relevant_docs = company_retriever.invoke(query)
        company_doc_chunks = "\n\n---\n\n".join(list(set(d.page_content for d in company_relevant_docs)))

        # --- 2. RAG on the FRAMEWORK Document ---
        framework_retriever = self.vectorstores[framework].as_retriever(search_kwargs={"k": k})
        framework_docs = []
        for c in company_relevant_docs[:3]: 
            framework_docs.extend(framework_retriever.invoke(c.page_content))
        
        framework_context = "\n\n---\n\n".join(list(set(d.page_content for d in framework_docs)))

        prompt = self.detailed_prompt if detailed else self.concise_prompt

        try:
            return (prompt | self.llm | self.output_parser).invoke({
                "framework_name": framework, 
                "context": framework_context, 
                "company_doc": company_doc_chunks 
            })
        except Exception:
            # --- Bulletproof Fallback Parsing ---
            try:
                raw_response = (prompt | self.llm).invoke({
                    "framework_name": framework, "context": framework_context, "company_doc": company_doc_chunks
                })
                text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
                
                start = text.find('{')
                end = text.rfind('}')
                
                if start != -1 and end != -1 and start < end:
                    json_str = text[start:end+1]
                    return json.loads(json_str)
                else:
                    return {
                        "framework": framework, 
                        "is_relevant": False, 
                        "executive_summary": text.strip()[:300], 
                        "compliance_score": 0
                    }
            except Exception as e:
                return {"framework": framework, "is_relevant": False, "error": f"JSON parse failed: {str(e)}"}

    def check_compliance(self, target_pdf_path: str, frameworks: Optional[List[str]] = None, k: int = 5, detailed: bool = True) -> Dict[str, Any]:
        if not os.path.exists(target_pdf_path): return {"error": "File not found"}
        
        requested = frameworks or self.SUPPORTED_FRAMEWORKS
        try:
            docs = PyPDFLoader(target_pdf_path).load()
        except Exception as e: return {"error": f"PDF load failed: {e}"}

        if not docs: return {"error": "Empty PDF"}

        # Chunk the uploaded document
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
        
        # Create a temporary Vector Database for the uploaded document
        temp_company_db = FAISS.from_documents(chunks, self.embeddings)

        results = {}
        for fw in requested:
            if fw in self.SUPPORTED_FRAMEWORKS:
                results[fw] = self._check_single_framework(fw, temp_company_db, k, detailed)
                
        return {"results": results}