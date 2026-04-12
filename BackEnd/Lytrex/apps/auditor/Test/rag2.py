import os
import glob
import json
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

class ComplianceRAG:
    """
    Elite Lytrex Compliance RAG with Map-Reduce and Context Budgeting.
    Powered by OpenAI (e.g., GPT-4o-mini) for stable, high-volume processing.
    """

    def __init__(self,
                 pdf_source_dir: str = "frameworks",
                 vector_db_base_path: str = "LytrexDB_Nomic",
                 embedding_model: str = "nomic-embed-text-v1.5",
                 model_name: str = "gpt-4o-mini", # Switched to OpenAI's fast model
                 chunk_size: int = 6000, 
                 chunk_overlap: int = 800,
                 nomic_api_key: Optional[str] = '',
                 openai_api_key: Optional[str] = ''): # Pass your OpenAI key here

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_base_path = os.path.join(base_dir, vector_db_base_path)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # --- Setup AI Services ---
        n_api_key = nomic_api_key or os.getenv("NOMIC_API_KEY")
        if not n_api_key: raise ValueError("NOMIC_API_KEY missing.")
        self.embeddings = NomicEmbeddings(model=embedding_model, nomic_api_key=n_api_key, dimensionality=768)

        o_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not o_api_key: raise ValueError("OPENAI_API_KEY missing. Please provide an OpenAI API key.")
        
        # --- Initialize OpenAI LLM ---
        self.llm = ChatOpenAI(
            temperature=0, 
            openai_api_key=o_api_key, 
            model_name=model_name,
            max_tokens=4096 
        )
        
        self.output_parser = JsonOutputParser()
        self._setup_prompts()

    def _setup_prompts(self):
        # --- MAPPER: Audits one section ---
        self.detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite, uncompromising Lead Compliance Auditor. Strictly audit the company document against the framework context.

            CRITICAL INSTRUCTIONS:
            1. DO NOT lazily copy the JSON template. Actively evaluate and compare the text against controls.
            2. if Not mentioned, don't say it's a violation
            3. Explicit Contradictions Only: Primarily flag violations where the company document explicitly contradicts the framework (e.g., Framework requires 256-bit encryption, but the document explicitly states 128-bit).
            4. Chain of Thought: Use 'internal_audit_reasoning' to map your logic. Detail what you checked, what was silent (and therefore ignored), and what contradicted the framework.
            5. Dynamic Scoring: Base score is 100. Calculate the 'compliance_score' by deducting 10-25 points for each explicit, critical violation found based on severity. 
            6. No Security Controls: If this section contains no security data, state this in reasoning, keep the score at 100, and leave lists empty.

            <framework_context>
            {context}
            </framework_context>

            <company_document_section>
            {company_doc}
            </company_document_section>

            Return ONLY valid JSON:
            {{
                "internal_audit_reasoning": "Step-by-step analysis: Checked X against Y. Document was silent on Z (ignored). Found explicit contradiction in W.",
                "compliance_score": 0,
                "executive_summary": "1-2 sentence strict assessment of this section's compliance posture.",
                "compliant_areas": ["[Control Name]: Requirement met with [specific evidence]"],
                "violations": ["[Control Name]: Framework requires [X], but document explicitly states [Y] (-15 pts)"],
                "recommendations": ["[Control Name]: [Specific remediation action]"]
            }}
            """
        )

        # --- REDUCER: Merges all section reports ---
        self.reduce_prompt = ChatPromptTemplate.from_template(
            """
            You are the Chief Auditor. Merge these section-based JSON reports into one master audit.
            
            CRITICAL INSTRUCTIONS:
            1. Deduplicate findings: If the same violation appears in multiple sections, only list it once.
            2. Master Scoring: Look at ALL unique violations across the reports. Calculate a final 'final_compliance_score'. Start at 100 and deduct points based on the severity and number of the unique violations.
            3. Ignore empty sections or sections that reported no relevant security controls.

            <raw_reports>
            {reports}
            </raw_reports>

            Return ONLY valid JSON matching this structure:
            {{
                "final_compliance_score": 0,
                "master_executive_summary": "A comprehensive summary of the entire document's compliance posture.",
                "all_compliant_areas": ["..."],
                "all_unique_violations": ["..."],
                "master_recommendations": ["..."]
            }}
            """
        )

    # --- TOKEN SAFETY HELPERS ---
    def prune_text(self, text: str, max_chars: int) -> str:
        """Truncates text to prevent LLM prompt rejection."""
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [TRUNCATED FOR TOKEN LIMIT] ..."
        return text

    def _get_fw_db_path(self, framework_name: str) -> str:
        return os.path.join(self.vector_db_base_path, framework_name.upper())

    def _load_fw_vectorstore(self, framework_name: str):
        path = self._get_fw_db_path(framework_name)
        if os.path.exists(os.path.join(path, "index.faiss")):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    def ingest_single_framework(self, framework_name: str):
        print(f"Creating Hierarchical DB for: {framework_name}")
        if framework_name.upper() == "ALL":
            pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, "**", "*.pdf"), recursive=True)
        else:
            pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, framework_name.upper(), "*.pdf"))
        
        if not pdf_paths: return None
        all_docs = []
        for path in pdf_paths: all_docs.extend(PyPDFLoader(path).load())

        p_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        c_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        p_docs = p_splitter.split_documents(all_docs)
        final_chunks = []
        for p in p_docs:
            children = c_splitter.split_text(p.page_content)
            for c in children:
                doc = p.copy()
                doc.page_content = c 
                doc.metadata["parent_content"] = p.page_content 
                final_chunks.append(doc)

        vectorstore = FAISS.from_documents(final_chunks, self.embeddings)
        out = self._get_fw_db_path(framework_name)
        os.makedirs(out, exist_ok=True)
        vectorstore.save_local(out)
        return vectorstore

    def evaluate_with_llm(self, context: str, doc: str) -> Dict[str, Any]:
        safe_ctx = self.prune_text(context, 12000)
        safe_doc = self.prune_text(doc, 4000)
        
        chain = self.detailed_prompt | self.llm | self.output_parser
        try:
            return chain.invoke({"context": safe_ctx, "company_doc": safe_doc})
        except Exception as e:
            return {"error": f"LLM Mapping Error: {str(e)}"}

    def audit_large_document(self, target_pdf_path: str, framework_name: str, k: int = 4) -> Dict[str, Any]:
        vectorstore = self._load_fw_vectorstore(framework_name) or self.ingest_single_framework(framework_name)
        if not vectorstore: return {"error": "Framework not found."}

        documents = PyPDFLoader(target_pdf_path).load()
        section_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        sections = section_splitter.split_documents(documents)
        
        all_reports = []
        print(f"\n[LYTREX] Auditing {len(sections)} sections (Forced Evaluation Mode)...")

        for i, section in enumerate(sections, 1):
            results = vectorstore.similarity_search(section.page_content, k=k*4)
            unique_parents = []
            seen = set()
            for res in results:
                p_text = res.metadata.get("parent_content", res.page_content)
                if p_text not in seen:
                    seen.add(p_text)
                    unique_parents.append(p_text)
                if len(unique_parents) == k: break
                    
            formatted_context = "\n\n---\n\n".join(unique_parents)
            
            # Map Phase (No Relevance Check)
            report = self.evaluate_with_llm(formatted_context, section.page_content)
            
            if "error" not in report:
                all_reports.append(report)
                # Safely extract the score, defaulting to 'N/A' just in case the LLM misses it
                score = report.get("compliance_score", "N/A")
                print(f"  -> Section {i}: Audited successfully. [Score: {score}]")
            else:
                print(f"  -> Section {i}: Failed parsing - {report['error']}")

        if not all_reports: return {"error": "Failed to generate any valid section reports."}

        # Reduce Phase
        print(f"\n[LYTREX] Reducing {len(all_reports)} section reports into Master Report...")
        chain = self.reduce_prompt | self.llm | self.output_parser
        try:
            return chain.invoke({"reports": json.dumps(all_reports, indent=2)})
        except Exception as e:
            return {"error": f"Reducer LLM Error: {str(e)}"}

    def check_compliance_text(self, text: str, framework_name: str, k: int = 5) -> Dict[str, Any]:
        vectorstore = self._load_fw_vectorstore(framework_name) or self.ingest_single_framework(framework_name)
        
        results = vectorstore.similarity_search(text, k=k*4)
        unique_parents = []
        seen = set()
        for res in results:
            p_text = res.metadata.get("parent_content", res.page_content)
            if p_text not in seen:
                seen.add(p_text)
                unique_parents.append(p_text)
            if len(unique_parents) == k: break
        
        context = "\n\n---\n\n".join(unique_parents)
        return self.evaluate_with_llm(context, text)