import os
import glob
import json
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

class ComplianceRAG:
    """
    Elite Lytrex Compliance RAG with Two-Stage Retrieval.
    Architecture: OpenAI Embeddings -> FAISS (Top 20) -> BGE Reranker (Top K) -> GPT-4o.
    """

    def __init__(self,
                 pdf_source_dir: str = "frameworks",
                 vector_db_base_path: str = "LytrexDB_OpenAI", 
                 embedding_model: str = "text-embedding-3-large", 
                 model_name: str = "gpt-4o",
                 reranker_model: str = "BAAI/bge-reranker-base", 
                 parent_chunk_size: int = 8000,   
                 parent_chunk_overlap: int = 800,
                 child_chunk_size: int = 800,     
                 child_chunk_overlap: int = 150,
                 map_chunk_size: int = 4000,      
                 map_chunk_overlap: int = 500,
                 openai_api_key: Optional[str] = None):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_base_path = os.path.join(base_dir, vector_db_base_path)

        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.map_chunk_size = map_chunk_size
        self.map_chunk_overlap = map_chunk_overlap

        o_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not o_api_key: raise ValueError("OPENAI_API_KEY missing. Please provide an OpenAI API key.")
        
        self.embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=o_api_key)
        
        print(f"[INIT] Loading BGE Cross-Encoder Reranker ({reranker_model})...")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        self.llm = ChatOpenAI(temperature=0, openai_api_key=o_api_key, model_name=model_name, max_tokens=4096)
        self.output_parser = JsonOutputParser()
        self._setup_prompts()

    def _setup_prompts(self):
        """Configures the system prompts for Map-Reduce and Relevance check."""
        
        # --- RELEVANCE GATE ---
        self.relevance_prompt = ChatPromptTemplate.from_template(
            """
            You are a strict Document Classification AI.
            Analyze this excerpt from an uploaded document. Is this document a corporate policy, cybersecurity framework, technical procedure, or business manual?
            If it is a menu, novel, random article, or irrelevant, reject it.
            
            Excerpt: {text}
            
            Respond ONLY with a JSON object:
            {{
                "is_relevant": true/false,
                "reasoning": "1 sentence explaining why."
            }}
            """
        )

        # --- DETAILED MAPPER ---
        self.detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite, strict Compliance Auditor AI developed by the Lytrex Team.
            Evaluate the <company_document> against the <framework_context>.

            CRITICAL INSTRUCTIONS:
            1. DO NOT lazily copy the JSON template. Actively evaluate.
            2. If a control is not mentioned, do NOT say it's a violation. Silence is ignored.
            3. Explicit Contradictions Only: Flag violations where the document explicitly contradicts the framework.
            4. Use 'internal_audit_reasoning' to map your logic BEFORE scoring.
            5. Base score is 100. Deduct 10-25 points for each explicit violation found.
            6. Traceability: For EVERY single comparison, sentence, compliant area, and violation, you MUST explicitly cite the specific section number, heading, and/or page number from the company document.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a comprehensive analysis.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "internal_audit_reasoning": "Step-by-step logic. I checked [Section X, Page Y]. Z was missing (ignored). Found violation in [Section W].",
                "compliance_score": 0,
                "executive_summary": "A detailed 3-4 sentence summary.",
                "compliant_areas": ["[Page X, Section Y] precise detail of what they did right"],
                "violations": ["[Page X, Section Y] specific breach with framework section references (-15 pts)"],
                "recommendations": ["[Page X, Section Y] Detailed actionable steps to fix the specific violation"]
            }}
            """
        )

        # --- CONCISE MAPPER (Summary Mode) ---
        self.concise_prompt = ChatPromptTemplate.from_template(
            """
            You are a fast Compliance Auditor AI developed by the Lytrex Team.
            Evaluate the <company_document> against the <framework_context>.

            CRITICAL INSTRUCTIONS:
            1. If a control is not mentioned, ignore it. Do NOT invent violations.
            2. Explicit Contradictions Only.
            3. Use 'internal_audit_reasoning' to do math. Base score 100, deduct for explicit violations.
            4. Traceability: You MUST explicitly cite the specific section number, heading, or page number from the company document for EVERY key issue and comparison sentence.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a strictly brief, top-level overview. Do not over-explain.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "internal_audit_reasoning": "Brief check of contradictions for scoring based on [Section X].",
                "compliance_score": 0,
                "summary": "A strict 1-sentence summary.",
                "key_issues": ["[Page X, Section Y] Top 1-3 critical explicit issues only"]
            }}
            """
        )

        # --- DETAILED REDUCER ---
        self.reduce_detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are the Chief Auditor. Merge these section-based detailed JSON reports into one master audit.
            Deduplicate findings and synthesize the final compliance score based on all unique violations.
            Ignore empty sections. Keep the page/section citations intact.

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

        # --- CONCISE REDUCER ---
        self.reduce_concise_prompt = ChatPromptTemplate.from_template(
            """
            You are the Chief Auditor. Merge these section-based summary JSON reports into one master overview.
            Deduplicate key issues and calculate the final score based on unique critical issues. Keep citations intact.

            <raw_reports>
            {reports}
            </raw_reports>

            Return ONLY valid JSON matching this structure:
            {{
                "final_compliance_score": 0,
                "master_summary": "A strict 1-2 sentence overall summary.",
                "all_unique_key_issues": ["Merged list of top critical issues"]
            }}
            """
        )

    def prune_text(self, text: str, max_chars: int) -> str:
        if len(text) > max_chars: return text[:max_chars] + "\n... [TRUNCATED FOR TOKEN LIMIT] ..."
        return text

    def _get_fw_db_path(self, framework_name: str) -> str:
        return os.path.join(self.vector_db_base_path, framework_name.upper())

    def _load_fw_vectorstore(self, framework_name: str):
        path = self._get_fw_db_path(framework_name)
        if os.path.exists(os.path.join(path, "index.faiss")):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    # =========================================================================
    # FRAMEWORK PARSING UTILITIES (Offloaded from views.py)
    # =========================================================================
    def get_framework_full_text(self, framework_name: str) -> str:
        """Parses framework PDFs directly so Django views.py doesn't have to."""
        fw_dir = os.path.join(self.pdf_source_dir, framework_name.upper())
        pdf_paths = glob.glob(os.path.join(fw_dir, "*.pdf"))
        
        if not pdf_paths:
            return ""
            
        full_text_parts = []
        for path in pdf_paths:
            docs = PyPDFLoader(path).load()
            full_text_parts.extend(doc.page_content for doc in docs)
            
        return "\n\n".join(full_text_parts).strip()

    def ingest_single_framework(self, framework_name: str):
        print(f"Creating Hierarchical DB for: {framework_name} using text-embedding-3-large")
        if framework_name.upper() == "ALL":
            pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, "**", "*.pdf"), recursive=True)
        else:
            pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, framework_name.upper(), "*.pdf"))
        
        if not pdf_paths: return None
        all_docs = []
        for path in pdf_paths: all_docs.extend(PyPDFLoader(path).load())

        p_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size, chunk_overlap=self.parent_chunk_overlap)
        c_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size, chunk_overlap=self.child_chunk_overlap)

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

    # =========================================================================
    # CORE EVALUATION LOGIC
    # =========================================================================
    def evaluate_with_llm(self, context: str, doc: str, summary_mode: bool = False) -> Dict[str, Any]:
        safe_ctx = self.prune_text(context, 12000)
        safe_doc = self.prune_text(doc, 4000)
        
        active_prompt = self.concise_prompt if summary_mode else self.detailed_prompt
        chain = active_prompt | self.llm | self.output_parser
        try:
            return chain.invoke({"context": safe_ctx, "company_doc": safe_doc})
        except Exception as e:
            return {"error": f"LLM Mapping Error: {str(e)}"}

    def audit_large_document(self, target_pdf_path: str, framework_name: str, k: int = 4, summary_mode: bool = False, evaluate_llm: bool = True) -> Dict[str, Any]:
        vectorstore = self._load_fw_vectorstore(framework_name) or self.ingest_single_framework(framework_name)
        if not vectorstore: return {"error": "Framework not found."}

        documents = PyPDFLoader(target_pdf_path).load()
        section_splitter = RecursiveCharacterTextSplitter(chunk_size=self.map_chunk_size, chunk_overlap=self.map_chunk_overlap)
        sections = section_splitter.split_documents(documents)
        
        all_reports = []
        raw_retrieval_log = {}
        
        mode_str = "SUMMARY" if summary_mode else "DETAILED"
        if not evaluate_llm: mode_str = "RETRIEVAL ONLY (Bypassing LLM)"
            
        print(f"\n[LYTREX] Processing {len(sections)} sections ({mode_str})...")

        for i, section in enumerate(sections, 1):
            results = vectorstore.similarity_search(section.page_content, k=k*5)
            unique_parents = list(dict.fromkeys([res.metadata.get("parent_content", res.page_content) for res in results]))
            
            if not unique_parents: continue

            pairs = [[section.page_content, parent] for parent in unique_parents]
            scores = self.reranker.predict(pairs)
            
            ranked_parents = [doc for _, doc in sorted(zip(scores, unique_parents), reverse=True)][:k]
            formatted_context = "\n\n---\n\n".join(ranked_parents)
            
            if not evaluate_llm:
                raw_retrieval_log[f"Section_{i}"] = formatted_context
                print(f"  -> Section {i}: Context retrieved and reranked successfully.")
                continue
                
            report = self.evaluate_with_llm(formatted_context, section.page_content, summary_mode)
            
            if "error" not in report:
                all_reports.append(report)
                score = report.get("compliance_score", "N/A")
                print(f"  -> Section {i}: Audited successfully. [Score: {score}]")
            else:
                print(f"  -> Section {i}: Failed parsing - {report['error']}")

        if not evaluate_llm: return {"raw_retrieval_results": raw_retrieval_log}
        if not all_reports: return {"error": "Failed to generate any valid section reports."}

        print(f"\n[LYTREX] Reducing {len(all_reports)} section reports into Master Report...")
        active_reduce_prompt = self.reduce_concise_prompt if summary_mode else self.reduce_detailed_prompt
        chain = active_reduce_prompt | self.llm | self.output_parser
        try:
            return chain.invoke({"reports": json.dumps(all_reports, indent=2)})
        except Exception as e:
            return {"error": f"Reducer LLM Error: {str(e)}"}

    # =========================================================================
    # THE API BRIDGE METHOD (Called by views.py)
    # =========================================================================
    def check_compliance(self, target_pdf_path: str, framework_name: str, k: int = 10, detailed: bool = False) -> Dict[str, Any]:
        """
        Wrapper method designed specifically to connect with Django views.py.
        Includes a Relevance Gate to reject non-policy documents.
        """
        # 1. Quick File Load for Gatekeeping
        try:
            docs = PyPDFLoader(target_pdf_path).load()
            if not docs: return {"error": "Uploaded PDF is empty or unreadable."}
            
            # Check first 3000 chars of document for relevance
            first_page_text = docs[0].page_content[:3000] 
            
            gate_chain = self.relevance_prompt | self.llm | self.output_parser
            gate_result = gate_chain.invoke({"text": first_page_text})
            
            if not gate_result.get("is_relevant", True):
                return {
                    "error": "Document Rejected: The file does not appear to be a corporate policy or procedure.",
                    "llm_reasoning": gate_result.get("reasoning", "Irrelevant document type.")
                }
        except Exception as e:
            print(f"[Warning] Relevance Gate failed, proceeding to audit: {str(e)}")

        # 2. Proceed to full Map-Reduce Audit
        summary_mode = not detailed
        return self.audit_large_document(
            target_pdf_path=target_pdf_path,
            framework_name=framework_name,
            k=k,
            summary_mode=summary_mode,
            evaluate_llm=True
        )

    def check_compliance_text(self, text: str, framework_name: str, k: int = 4, summary_mode: bool = False, evaluate_llm: bool = True) -> Dict[str, Any]:
        """Streamlined version for short text snippets (CLI testing)."""
        vectorstore = self._load_fw_vectorstore(framework_name) or self.ingest_single_framework(framework_name)
        results = vectorstore.similarity_search(text, k=k*5)
        unique_parents = list(dict.fromkeys([res.metadata.get("parent_content", res.page_content) for res in results]))
        
        if unique_parents:
            pairs = [[text, parent] for parent in unique_parents]
            scores = self.reranker.predict(pairs)
            ranked_parents = [doc for _, doc in sorted(zip(scores, unique_parents), reverse=True)][:k]
            context = "\n\n---\n\n".join(ranked_parents)
        else:
            context = ""
        
        if not evaluate_llm: return {"retrieved_framework_context": context}
        return self.evaluate_with_llm(context, text, summary_mode)