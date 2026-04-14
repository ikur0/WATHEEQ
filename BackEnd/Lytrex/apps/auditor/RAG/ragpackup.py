"""
The rag4.py Pipeline Flow
The Input (Chunking): You upload a 50-page PDF. RecursiveCharacterTextSplitter chops this massive document into smaller, manageable pieces (roughly 4,000 characters each). The system grabs Section 1 and starts the loop.

Hybrid Search (The Dragnet):
Section 1 is sent to the vector database.

FAISS uses OpenAI embeddings to find the top 20 framework rules that share the same meaning or concept as Section 1.

BM25 scans the exact words in Section 1 and finds the top 20 framework rules that contain the exact same keywords or acronyms.

Deduplication (The Filter):
The results from FAISS and BM25 are dumped into a single list. The Python dictionary logic (unique_children_map) scans this list and deletes any identical duplicates, leaving a clean list of unique framework rules.

Cross-Encoder Reranking (The Judge):
The clean list of rules is paired with Section 1 and sent to the bge-reranker-base model. Because we are sending the small 500-character "Child" chunks, the reranker can read them perfectly. It scores every pair from 0.0 to 1.0 based on absolute relevance, and we keep only the top K (e.g., the top 4).

Context Assembly (The Expansion):
For those top 4 winning child chunks, the system looks up their hidden metadata and pulls their massive 2,500-character "Parent" chunks. It staples the Framework Page Numbers to the top of them and combines them into one giant Context string.

LLM Auditing (The Verdict):
GPT-4o receives Section 1 (tagged with its document page number) and the giant Framework Context string. Following the strict dual-citation prompt, it writes the JSON report detailing the violations, gaps, and scores.

"""

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
from rank_bm25 import BM25Okapi  # <--- Added BM25

load_dotenv()

class ComplianceRAG:
    """
    Elite Lytrex Compliance RAG with Hybrid Retrieval.
    Architecture: OpenAI Embeddings + BM25 -> Union -> BGE Reranker (on Children) -> GPT-4o (on Parents).
    """

    def __init__(self,
                 pdf_source_dir: str = "frameworks",
                 vector_db_base_path: str = "LytrexDB_OpenAI", 
                 embedding_model: str = "text-embedding-3-large", 
                 model_name: str = "gpt-4o",
                 reranker_model: str = "BAAI/bge-reranker-base", 
                 parent_chunk_size: int = 2500,   
                 parent_chunk_overlap: int = 250,
                 child_chunk_size: int = 500,     
                 child_chunk_overlap: int = 100,
                 map_chunk_size: int = 1000,      
                 map_chunk_overlap: int = 250,
                 openai_api_key: Optional[str] = ''):

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

        self.llm = ChatOpenAI(
            temperature=0, 
            openai_api_key=o_api_key, 
            model_name=model_name, 
            max_tokens=4096,
            model_kwargs={"seed": 42}
        )
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

            hey don't be strict, we don't want to miss anything
            """
        )

    def prune_text(self, text: str, max_chars: int) -> str:
        if len(text) > max_chars: return text[:max_chars] + "\n... [TRUNCATED FOR TOKEN LIMIT] ..."
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer for BM25."""
        return text.lower().replace("\n", " ").split(" ")

    def _get_fw_db_path(self, framework_name: str) -> str:
        return os.path.join(self.vector_db_base_path, framework_name.upper())

    def _load_fw_vectorstore(self, framework_name: str):
        path = self._get_fw_db_path(framework_name)
        if os.path.exists(os.path.join(path, "index.faiss")):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    # =========================================================================
    # FRAMEWORK PARSING UTILITIES
    # =========================================================================
    def get_framework_full_text(self, framework_name: str) -> str:
        fw_dir = os.path.join(self.pdf_source_dir, framework_name.upper())
        pdf_paths = sorted(glob.glob(os.path.join(fw_dir, "*.pdf")))
        
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
            pdf_paths = sorted(glob.glob(os.path.join(self.pdf_source_dir, "**", "*.pdf"), recursive=True))
        else:
            pdf_paths = sorted(glob.glob(os.path.join(self.pdf_source_dir, framework_name.upper(), "*.pdf")))
        
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

        all_docs = sorted(list(vectorstore.docstore._dict.values()), key=lambda x: x.page_content)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)

        documents = PyPDFLoader(target_pdf_path).load()
        section_splitter = RecursiveCharacterTextSplitter(chunk_size=self.map_chunk_size, chunk_overlap=self.map_chunk_overlap)
        sections = section_splitter.split_documents(documents)
        
        all_reports = []
        raw_retrieval_log = {}
        
        mode_str = "SUMMARY" if summary_mode else "DETAILED"
        if not evaluate_llm: mode_str = "RETRIEVAL ONLY (Bypassing LLM)"
            
        print(f"\n[LYTREX] Processing {len(sections)} sections ({mode_str})...")

        for i, section in enumerate(sections, 1):
            # 1. Hybrid Search (FAISS + BM25)
            faiss_results = vectorstore.similarity_search(section.page_content, k=k*5)
            
            tokenized_query = self._tokenize(section.page_content)
            bm25_results = bm25.get_top_n(tokenized_query, all_docs, n=k*5)
            
            combined_children = faiss_results + bm25_results

            # Deduplicate child chunks based on their text
            unique_children_map = {}
            for child in combined_children:
                if child.page_content not in unique_children_map:
                    unique_children_map[child.page_content] = child

            unique_children = list(unique_children_map.values())
            if not unique_children: continue

            # 2. Rerank CHILD chunks directly (prevents BGE 512 token limit truncation)
            pairs = [[section.page_content, child.page_content] for child in unique_children]
            scores = self.reranker.predict(pairs)
            
            # Sort children by score
            scored_children = sorted(zip(scores, unique_children), key=lambda x: x[0], reverse=True)
            top_k_children = [child for score, child in scored_children[:k]]
            
            # 3. Extract the full PARENT chunks only for the winning children
            unique_parents = []
            parent_meta_map = {}
            for child in top_k_children:
                parent_text = child.metadata.get("parent_content", child.page_content)
                if parent_text not in parent_meta_map:
                    # Explicitly inject Framework Page & Source Metadata
                    fw_page = child.metadata.get("page", 0) + 1
                    fw_source = os.path.basename(child.metadata.get("source", "Framework"))
                    tagged_fw_text = f"--- FRAMEWORK SOURCE: {fw_source} | PAGE: {fw_page} ---\n{parent_text}"
                    parent_meta_map[parent_text] = tagged_fw_text
                    unique_parents.append(parent_text)
            
            formatted_context = "\n\n---\n\n".join([parent_meta_map[p] for p in unique_parents])
            
            # Explicitly inject Company Document Page & Source Metadata
            actual_doc_page = section.metadata.get("page", 0) + 1
            doc_source = os.path.basename(section.metadata.get("source", "Company Document"))
            chunk_with_metadata = f"--- COMPANY DOCUMENT | FILE: {doc_source} | PAGE: {actual_doc_page} ---\n{section.page_content}"

            if not evaluate_llm:
                raw_retrieval_log[f"Section_{i}"] = {
                    "query": chunk_with_metadata,
                    "context": formatted_context
                }
                print(f"  -> Section {i}: Context retrieved and reranked successfully.")
                continue
                
            # Pass the tagged metadata string, NOT the raw page_content
            report = self.evaluate_with_llm(formatted_context, chunk_with_metadata, summary_mode)
            
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
        try:
            docs = PyPDFLoader(target_pdf_path).load()
            if not docs: return {"error": "Uploaded PDF is empty or unreadable."}
            
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

        summary_mode = not detailed
        return self.audit_large_document(
            target_pdf_path=target_pdf_path,
            framework_name=framework_name,
            k=k,
            summary_mode=summary_mode,
            evaluate_llm=True
        )

    def check_compliance_text(self, text: str, framework_name: str, k: int = 4, summary_mode: bool = False, evaluate_llm: bool = True) -> Dict[str, Any]:
        """Streamlined version for short text snippets with Hybrid Search and Child-Reranking."""
        vectorstore = self._load_fw_vectorstore(framework_name) or self.ingest_single_framework(framework_name)
        
        all_docs = sorted(list(vectorstore.docstore._dict.values()), key=lambda x: x.page_content)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)

        faiss_results = vectorstore.similarity_search(text, k=k*5)
        tokenized_query = self._tokenize(text)
        bm25_results = bm25.get_top_n(tokenized_query, all_docs, n=k*5)

        combined_children = faiss_results + bm25_results

        unique_children_map = {}
        for child in combined_children:
            if child.page_content not in unique_children_map:
                unique_children_map[child.page_content] = child

        unique_children = list(unique_children_map.values())
        
        if unique_children:
            pairs = [[text, child.page_content] for child in unique_children]
            scores = self.reranker.predict(pairs)
            
            scored_children = sorted(zip(scores, unique_children), key=lambda x: x[0], reverse=True)
            top_k_children = [child for score, child in scored_children[:k]]
            
            unique_parents = []
            parent_meta_map = {}
            for child in top_k_children:
                parent_text = child.metadata.get("parent_content", child.page_content)
                if parent_text not in parent_meta_map:
                    # Explicitly inject Framework Page & Source Metadata
                    fw_page = child.metadata.get("page", 0) + 1
                    fw_source = os.path.basename(child.metadata.get("source", "Framework"))
                    tagged_fw_text = f"--- FRAMEWORK SOURCE: {fw_source} | PAGE: {fw_page} ---\n{parent_text}"
                    parent_meta_map[parent_text] = tagged_fw_text
                    unique_parents.append(parent_text)
                    
            context = "\n\n---\n\n".join([parent_meta_map[p] for p in unique_parents])
        else:
            context = ""
        
        if not evaluate_llm: return {"retrieved_framework_context": context}
        
        # Explicitly inject mock Company Document Metadata for text snippets
        chunk_with_metadata = f"--- COMPANY DOCUMENT | FILE: Text Snippet | PAGE: N/A ---\n{text}"
        
        return self.evaluate_with_llm(context, chunk_with_metadata, summary_mode)