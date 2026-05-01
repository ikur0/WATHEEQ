""" framework is from json (for NCA only)
From NCA.json


PDF Document
 ↓
Gatekeeper (Relevance Check)
 ↓
Split → Sections
 ↓
FOR EACH Section:
    ↓
    Split → small chunks
    ↓
    For each chunk:
        BM25 + FAISS
    ↓
    Merge all candidates
    ↓
    Deduplicate
    ↓
    Cross-Encoder (Section vs Candidate)
    ↓
    Top-K results
    ↓
    Context Assembly (Parent chunks)
    ↓
    LLM (Micro-Evaluation for this section)
    ↓
    Store JSON result
END LOOP
 ↓
Collect all section results
 ↓
Deduplicate findings (across sections)
 ↓
Aggregate scores
 ↓
LLM (Final Evaluation / Executive Summary)
 ↓
Final JSON Output
"""

import os
import glob
import json
import re
import random
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document  
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi 

load_dotenv()

class ComplianceRAG:
    """
    Elite Lytrex Compliance RAG with Structural Hybrid Retrieval.
    Architecture: JSON-Structured Parents -> Child Chunks -> FAISS+BM25 -> Reranker -> GPT-4o (OpenAI).
    """

    def __init__(self,
                 # Directory containing the source framework files (PDFs/JSONs)
                 pdf_source_dir: str = "frameworks",
                 # Base path where the FAISS vector database will be saved/loaded
                 vector_db_base_path: str = "LytrexDB_OpenAI", 
                 # The OpenAI embedding model used to vectorize the text
                 embedding_model: str = "text-embedding-3-large", 
                 # The OpenAI LLM used for auditing and evaluation
                 model_name: str = "gpt-4o",
                 # The Cross-Encoder model used to rerank retrieved candidates
                 reranker_model: str = "BAAI/bge-reranker-base", 
                 # Character size for the larger parent context chunks
                 parent_chunk_size: int = 2500,   
                 # Character overlap for the larger parent context chunks
                 parent_chunk_overlap: int = 250,
                 # Character size for the smaller child chunks stored in the vector DB
                 child_chunk_size: int = 500,     
                 # Character overlap for the smaller child chunks
                 child_chunk_overlap: int = 100,
                 # Character size for initially splitting the user's uploaded target document into sections
                 map_chunk_size: int = 500,      
                 # Character overlap for the target document sections
                 map_chunk_overlap: int = 100,
                 # Character size for breaking down target document sections into micro-queries for retrieval
                 section_chunk_size: int = 200,    
                 # Character overlap for the target document micro-queries
                 section_chunk_overlap: int = 20,  
                 # Optional explicit API key; if not provided, it will look for OPENAI_API_KEY in the .env file
                 openai_api_key: Optional[str] = ''):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_base_path = os.path.join(base_dir, vector_db_base_path)

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.map_chunk_size = map_chunk_size
        self.map_chunk_overlap = map_chunk_overlap
        self.section_chunk_size = section_chunk_size
        self.section_chunk_overlap = section_chunk_overlap

        o_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not o_api_key: raise ValueError("OPENAI_API_KEY missing. Please check your .env file.")
        
        print(f"[INIT] Loading OpenAI Embeddings ({embedding_model})...")
        self.embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=o_api_key)
        
        print(f"[INIT] Loading BGE Cross-Encoder Reranker ({reranker_model})...")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        self.llm = ChatOpenAI(
            temperature=0, 
            openai_api_key=o_api_key, 
            model_name=model_name
        )
        self.output_parser = JsonOutputParser()
        self._setup_prompts()

    def _setup_prompts(self):
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
            6. Traceability: For EVERY single comparison, sentence, compliant area, and violation, you MUST explicitly cite it in this exact format: [Company Page: X | Company Section: Y | Framework Control: Z]. Extract the page from the metadata provided or the printed text.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a comprehensive analysis.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "internal_audit_reasoning": "Step-by-step logic. I checked [Company Section: X]. Z was missing (ignored). Found violation in [Company Section: W].",
                "compliance_score": 1-100,
                "executive_summary": "A detailed 3-4 sentence summary.",
                "compliant_areas": ["[Company Page: X | Company Section: Y (Never mention Section Title) | Framework section: Z (never mention framework ID)] precise detail of what they did right"],
                "violations": ["[Company Page: X | Company Section: Y (Never mention Section Title) | Framework section: Z (never mention framework ID)] specific breach (-15 pts)"],
                "recommendations": ["[Company Page: X | Company Section: Y (Never mention Section Title) | Framework section: Z (never mention framework ID)] Detailed actionable steps to fix the specific violation"]
            }}
            """
        )

        self.concise_prompt = ChatPromptTemplate.from_template(
            """
            You are a fast Compliance Auditor AI developed by the Lytrex Team.
            Evaluate the <company_document> against the <framework_context>.

            CRITICAL INSTRUCTIONS:
            1. If a control is not mentioned, ignore it. Do NOT invent violations.
            2. Explicit Contradictions Only.
            3. Use 'internal_audit_reasoning' to do math. Base score 100, deduct for explicit violations.
            4. Traceability: You MUST explicitly format the citation exactly like this: [Company Page: X | Company Section: Y (Never mention Section Title) | Framework Control: Z (never mention framework ID)] for EVERY key issue and comparison sentence. Get the page from the metadata provided or printed text.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Provide a strictly brief, top-level overview. Do not over-explain.
            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "internal_audit_reasoning": "Brief check of contradictions for scoring based on [Company Section: X].",
                "compliance_score": 1-100,
                "summary": "A strict 1-sentence summary.",
                "key_issues": ["[Company Page: X | Company Section: Y | Framework section: Z] Top critical explicit issue"]
            }}
            """
        )

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
                "final_compliance_score": 1-100,
                "master_executive_summary": "A comprehensive summary of the entire document's compliance posture.",
                "all_compliant_areas": ["..."],
                "all_unique_violations": ["..."],
                "master_recommendations": ["..."]
            }}
            """
        )

        self.reduce_concise_prompt = ChatPromptTemplate.from_template(
            """
            You are the Chief Auditor. Merge these section-based summary JSON reports into one master overview.
            Deduplicate key issues and calculate the final score based on unique critical issues. Keep citations intact.

            <raw_reports>
            {reports}
            </raw_reports>

            Return ONLY valid JSON matching this structure:
            {{
                "final_compliance_score": 1-100,
                "master_summary": "A strict 1-2 sentence overall summary.",
                "all_unique_key_issues": ["Merged list of top critical issues"]
            }}

            hey don't be strict, we don't want to miss anything
            """
        )

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_framework_section(self, sec: Dict[str, Any]) -> Dict[str, str]:
        # Pull cleanly from structured JSON
        internal_section_id = self._clean_text(sec.get("section_id")) or self._clean_text(sec.get("control_id")) or "Unknown"
        chapter = self._clean_text(sec.get("chapter"))
        section = self._clean_text(sec.get("section"))
        control = self._clean_text(sec.get("control"))
        subsection = self._clean_text(sec.get("subsection"))
        
        # Build the best available display number
        display_number = (
            self._clean_text(sec.get("section_number"))
            or self._clean_text(sec.get("control_number"))
            or control
            or section
            or chapter
        )
        
        title = self._clean_text(sec.get("title")) or "Untitled Section"
        start_page = self._clean_text(sec.get("start_page")) or "N/A"
        end_page = self._clean_text(sec.get("end_page")) or start_page
        page_label = start_page if start_page == end_page or end_page in {"", "N/A"} else f"{start_page}-{end_page}"

        # Combine for a full display title
        full_title_parts = []
        if display_number:
            full_title_parts.append(display_number)
        if title and title != "Untitled Section":
            full_title_parts.append(title)
        full_title = " ".join(full_title_parts).strip() or "Untitled Section"

        return {
            "internal_section_id": internal_section_id,
            "display_number": display_number,
            "chapter": chapter,
            "section": section,
            "control": control,
            "subsection": subsection,
            "title": title,
            "raw_title": self._clean_text(sec.get("title")),
            "full_title": full_title,
            "start_page": start_page,
            "end_page": end_page,
            "page_label": page_label,
            "text": self._clean_text(sec.get("text")),
        }

    def _extract_company_section_label(self, text: str) -> str:
        # Regex is still needed here *only* to parse the raw unstructured text from the user's uploaded PDF
        patterns = [
            r'(?im)^\s*((?:section|chapter|control|policy)\s+\d+(?:\.\d+)*)\b',
            r'(?im)^\s*(\d+(?:\.\d+){1,5})\s*[-–:.]?\s+(.{3,120})$',
        ]

        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if not match:
                continue
            if idx == 0:
                return match.group(1).strip()
            number = match.group(1).strip()
            title = match.group(2).strip()
            return f"{number} {title}"

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[0][:120]
        return "Unknown Section"

    def prune_text(self, text: str, max_chars: int) -> str:
        if len(text) > max_chars: return text[:max_chars] + "\n... [TRUNCATED] ..."
        return text

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace("\n", " ").split(" ")

    def _get_fw_db_path(self, framework_name: str) -> str:
        return os.path.join(self.vector_db_base_path, framework_name.upper())

    def _load_fw_vectorstore(self, framework_name: str):
        path = self._get_fw_db_path(framework_name)
        if os.path.exists(os.path.join(path, "index.faiss")):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    # =========================================================================
    # STRUCTURAL INGESTION (JSON-DRIVEN)
    # =========================================================================
    def ingest_single_framework(self, framework_name: str):
        fw_name = framework_name.upper()
        json_path = os.path.join(self.pdf_source_dir, fw_name, f"{fw_name}.json")
        
        if not os.path.exists(json_path):
            print(f"[!] {json_path} not found. Falling back to blind PDF loading.")
            return self._legacy_ingest_pdf(framework_name)
        
        print(f"[STRUCTURAL INGEST] Loading sections from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict) and "sections" in raw_data:
            sections = raw_data["sections"]
        elif isinstance(raw_data, list):
            sections = raw_data
        else:
            sections = [raw_data]

        c_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size, 
            chunk_overlap=self.child_chunk_overlap
        )

        final_chunks = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue

            normalized = self._normalize_framework_section(sec)
            full_text = normalized["text"]
            if not full_text:
                continue

            children = c_splitter.split_text(full_text)

            for idx, c in enumerate(children):
                doc = Document(
                    page_content=c,
                    metadata={
                        "parent_content": full_text,
                        "fw_internal_section_id": normalized["internal_section_id"],
                        "fw_section_number": normalized["display_number"],
                        "control_id": normalized["control"],
                        "fw_title": normalized["title"],
                        "fw_raw_title": normalized["raw_title"],
                        "domain": normalized["full_title"],
                        "chapter": normalized["chapter"],
                        "section": normalized["section"],
                        "subsection": normalized["subsection"],
                        "source": f"{fw_name}_Framework",
                        "page": normalized["page_label"],
                        "page_start": normalized["start_page"],
                        "page_end": normalized["end_page"],
                        "child_index": idx,
                    }
                )
                final_chunks.append(doc)

        if not final_chunks:
            print("[!] No text could be parsed from the JSON. Check the format.")
            return None

        vectorstore = FAISS.from_documents(final_chunks, self.embeddings)
        out = self._get_fw_db_path(framework_name)
        os.makedirs(out, exist_ok=True)
        vectorstore.save_local(out)
        return vectorstore

    def _legacy_ingest_pdf(self, framework_name: str):
        print("legaccylegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnnlegaccy runnn runnn")
        pdf_paths = sorted(glob.glob(os.path.join(self.pdf_source_dir, framework_name.upper(), "*.pdf")))
        if not pdf_paths: return None
        all_docs = []
        for path in pdf_paths: all_docs.extend(PyPDFLoader(path).load())
        
        p_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size, chunk_overlap=250)
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
    def audit_large_document(self, target_pdf_path: str, framework_name: str, k: int = 10, summary_mode: bool = False, evaluate_llm: bool = True) -> Dict[str, Any]:
        vectorstore = self._load_fw_vectorstore(framework_name) or self.ingest_single_framework(framework_name)
        if not vectorstore: return {"error": "Framework not found."}

        all_docs = sorted(list(vectorstore.docstore._dict.values()), key=lambda x: x.page_content)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)

        documents = PyPDFLoader(target_pdf_path).load()
        section_splitter = RecursiveCharacterTextSplitter(chunk_size=self.map_chunk_size, chunk_overlap=self.map_chunk_overlap)
        sections = section_splitter.split_documents(documents)
        
        # USE CONSTRUCTOR PARAMETERS HERE
        small_chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.section_chunk_size, 
            chunk_overlap=self.section_chunk_overlap
        )
        
        all_reports = []
        raw_retrieval_log = {}
        
        mode_str = "SUMMARY" if summary_mode else "DETAILED"
        if not evaluate_llm: mode_str = "RETRIEVAL ONLY (Bypassing LLM)"
            
        print(f"\n[LYTREX] Processing {len(sections)} sections ({mode_str})...")

        for i, section in enumerate(sections, 1):
            
            # --- NEW PIPELINE: Split -> small chunks -> BM25+FAISS for each chunk ---
            section_small_chunks = small_chunk_splitter.split_text(section.page_content)
            
            combined_children = []
            
            for small_chunk in section_small_chunks:
                # Retrieve candidates per chunk (using dynamic multiplier to manage load)
                chunk_faiss = vectorstore.similarity_search(small_chunk, k=k*2)
                
                tokenized_chunk = self._tokenize(small_chunk)
                chunk_bm25 = bm25.get_top_n(tokenized_chunk, all_docs, n=k*2)
                
                combined_children.extend(chunk_faiss)
                combined_children.extend(chunk_bm25)
            # -------------------------------------------------------------------------

            # Merge all candidates and Deduplicate
            unique_children_map = {}
            for child in combined_children:
                if child.page_content not in unique_children_map:
                    unique_children_map[child.page_content] = child

            unique_children = list(unique_children_map.values())
            if not unique_children: continue

            # Cross-Encoder (Section vs Candidate)
            pairs = [[section.page_content, child.page_content] for child in unique_children]
            scores = self.reranker.predict(pairs)
            
            # Top-K results
            scored_children = sorted(zip(scores, unique_children), key=lambda x: x[0], reverse=True)
            top_k_children = [child for score, child in scored_children[:k]]
            
            # Context Assembly (Parent chunks)
            unique_parents = []
            parent_meta_map = {}
            for child in top_k_children:
                parent_text = child.metadata.get("parent_content", child.page_content)
                if parent_text not in parent_meta_map:
                    fw_internal_section_id = child.metadata.get("fw_internal_section_id", "N/A")
                    fw_section_number = child.metadata.get("fw_section_number", "")
                    fw_title = child.metadata.get("fw_title") or child.metadata.get("fw_raw_title") or child.metadata.get("domain", "N/A")
                    fw_page = child.metadata.get("page", "N/A")
                    
                    meta_bits = [f"FRAMEWORK ID: {fw_internal_section_id}", f"FRAMEWORK PAGE(S): {fw_page}"]
                    if fw_section_number:
                        meta_bits.insert(1, f"FRAMEWORK SECTION/CONTROL: {fw_section_number}")
                    if fw_title:
                        meta_bits.append(f"FRAMEWORK TITLE: {fw_title}")

                    tagged_fw_text = f"--- {' | '.join(meta_bits)} ---\n{parent_text}"
                    parent_meta_map[parent_text] = tagged_fw_text
                    unique_parents.append(parent_text)

            formatted_context = "\n\n---\n\n".join([parent_meta_map[p] for p in unique_parents])

            actual_doc_page = int(section.metadata.get("page", 0)) + 1
            doc_source = os.path.basename(section.metadata.get("source", "Company Document"))
            company_section_label = self._extract_company_section_label(section.page_content)
            
            chunk_with_metadata = (
                f"--- COMPANY DOCUMENT | FILE: {doc_source} | COMPANY PAGE: {actual_doc_page} | "
                f"COMPANY SECTION: {company_section_label} ---\n{section.page_content}"
            )

            if not evaluate_llm:
                raw_retrieval_log[f"Section_{i}"] = {
                    "query": chunk_with_metadata,
                    "context": formatted_context
                }
                print(f"  -> Section {i}: Context retrieved and reranked successfully.")
                continue
                
            # LLM (Micro-Evaluation for this section)
            active_prompt = self.concise_prompt if summary_mode else self.detailed_prompt
            chain = active_prompt | self.llm | self.output_parser
            
            try:
                report = chain.invoke({"context": self.prune_text(formatted_context, 12000), "company_doc": self.prune_text(chunk_with_metadata, 4000)})
                
                # --- NEW COMPLIANCE SCORING LOGIC COMPUTED WITHOUT LLM INTERVENTION ---
                violations_list = report.get("key_issues", []) if summary_mode else report.get("violations", [])
                calc_score = 100.0
                for _ in violations_list:
                    # Decrease by ~5% randomized slightly to prevent duplicate scores like 75 75
                    # calc_score -= (5.0 * random.uniform(0.4, 0.6))
                    calc_score -= (5.0)
                
                report["compliance_score"] = round(max(0.0, calc_score), 2)
                # -----------------------------------------------------------------------

                all_reports.append(report) # Store JSON result
                score = report.get("compliance_score", "N/A")
                print(f"  -> Section {i}: Audited successfully. [Score: {score}]")
            except Exception as e:
                print(f"  -> Section {i}: Failed parsing - {str(e)}")

        if not evaluate_llm: return {"raw_retrieval_results": raw_retrieval_log}
        if not all_reports: return {"error": "Failed to generate any valid section reports."}

        # Deduplicate findings -> Aggregate scores -> LLM Final Evaluation -> Final JSON Output
        print(f"\n[LYTREX] Reducing {len(all_reports)} section reports into Master Report...")
        active_reduce_prompt = self.reduce_concise_prompt if summary_mode else self.reduce_detailed_prompt
        chain = active_reduce_prompt | self.llm | self.output_parser
        try:
            final_report = chain.invoke({"reports": json.dumps(all_reports, indent=2)})
            
            # --- NEW FINAL COMPLIANCE SCORING LOGIC COMPUTED WITHOUT LLM INTERVENTION ---
            master_violations = final_report.get("all_unique_key_issues", []) if summary_mode else final_report.get("all_unique_violations", [])
            master_score = 100.0
            for _ in master_violations:
                # Decrease by ~5% randomized slightly 
                # master_score -= (5.0 * random.uniform(0.8, 1.2))
                master_score -= (5.0)
            
            final_report["final_compliance_score"] = round(max(0.0, master_score), 2)
            # ----------------------------------------------------------------------------

            return final_report
        except Exception as e:
            return {"error": f"Reducer LLM Error: {str(e)}"}

    # =========================================================================
    # THE API BRIDGE METHOD 
    # =========================================================================
    def check_compliance(self, target_pdf_path: str, framework_name: str, k: int = 10, detailed: bool = False) -> Dict[str, Any]:
        try:
            docs = PyPDFLoader(target_pdf_path).load()
            if not docs: return {"error": "Uploaded PDF is empty or unreadable."}
            
            # Gatekeeper (Relevance Check)
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

    def get_framework_full_text(self, framework_name: str) -> str:
        fw_name = framework_name.upper()
        json_path = os.path.join(self.pdf_source_dir, fw_name, f"{fw_name}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            if isinstance(raw_data, dict) and "sections" in raw_data:
                sections = raw_data["sections"]
            elif isinstance(raw_data, list):
                sections = raw_data
            else:
                sections = [raw_data]
                
            return "\n\n".join([s.get("text", "") for s in sections if isinstance(s, dict)])
        
        pdf_paths = sorted(glob.glob(os.path.join(self.pdf_source_dir, fw_name, "*.pdf")))
        full_text = []
        for p in pdf_paths: full_text.extend(doc.page_content for doc in PyPDFLoader(p).load())
        return "\n\n".join(full_text)