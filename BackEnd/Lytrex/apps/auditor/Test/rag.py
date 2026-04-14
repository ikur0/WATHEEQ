"""
From NCA.json


The ComplianceRAG Pipeline Flow

The Gatekeeper (Relevance Check): 
Before anything starts, you upload a target PDF. The system grabs the first 3,000 characters and asks a strict LLM classifier to determine if it is a legitimate corporate policy or security document. If it looks like a recipe, a novel, or random text, the system slams the door shut and rejects it immediately.

The Input (Chunking): 
Once approved, a RecursiveCharacterTextSplitter chops your company document into smaller, digestible pieces (roughly 500 characters each). The system grabs Section 1, extracts its specific headings or section numbers using a regex parser, and starts the loop.

Hybrid Search (The Dragnet):
Section 1 is sent to the vector database and the tokenized corpus.
- FAISS uses OpenAI embeddings (text-embedding-3-large) to find the top framework "Child" chunks (e.g., top 20) that share the same conceptual meaning as Section 1.
- BM25 scans the exact words in Section 1 and finds the top framework "Child" chunks (e.g., top 20) that contain the exact same keywords or acronyms.

Deduplication (The Filter):
The results from FAISS and BM25 are dumped into a single combined list. The Python dictionary logic (unique_children_map) scans this list and deletes any identical duplicate chunks, leaving a highly refined, clean list of unique framework rules.

Cross-Encoder Reranking (The Judge):
The clean list of rules is paired with Section 1 and sent to the bge-reranker-base model. Because we are sending the small 500-character "Child" chunks, the reranker can read them perfectly. It scores every pair based on absolute relevance, and we keep only the top K absolute winners (e.g., the top 4 or 10).

Context Assembly (The Expansion):
For those top winning child chunks, the system looks up their hidden metadata and pulls their massive "Parent" chunks (originally ingested from structured JSON). It staples the Framework Section Numbers, IDs, and Titles to the top of them and combines them into one giant, highly accurate Framework Context string.

LLM Auditing (The Micro-Verdict):
GPT-4o receives Section 1 (tagged with its company document page number and section label) and the giant Framework Context string. Following either a strict "Detailed" or "Concise" prompt, it evaluates every single requirement and outputs a structured JSON report detailing the specific violations, key issues, and a compliance score for just that section.

The Reducer (The Master Verdict):
After the system loops through every single section of your PDF, it gathers all the individual JSON reports. The "Chief Auditor" LLM prompt merges these raw reports together, deduplicates identical findings from different pages, synthesizes the final overall compliance score, and generates the master executive summary in one final JSON output.
"""


"""framework is from json (for NCA only)
Lytrex ComplianceRAG Pipeline Flow (Structural Hybrid Architecture)

[Start: check_compliance API Bridge]
       ↓
[PDF Document Uploaded]
       ↓
[Gatekeeper (Relevance Check)]
       │ (Extracts first 3,000 chars. LLM classifies if it's a valid policy/framework)
       ├──> If Irrelevant: Reject Document & Return Error.
       ↓
[Document Splitting]
       │ (Splits company PDF into small "Sections" of 500 characters)
       ↓
=========================================================
                 LOOP: FOR EACH 500-CHAR SECTION
=========================================================
       ↓
[Hybrid Search]
       │ (Uses the 500-char Section text as the query)
       ├──> FAISS Vector Search pulls Top K*5 Child Chunks (Semantic)
       └──> BM25 Keyword Search pulls Top K*5 Child Chunks (Lexical)
       ↓
[Merge & Deduplicate]
       │ (Combines results into one list, purges identical overlapping chunks)
       ↓
[Cross-Encoder Reranking]
       │ (BGE-Reranker scores Section vs. each Unique Child Chunk)
       │ (Focuses purely on absolute relevance score)
       ↓
[Filter Top-K]
       │ (Keeps only the absolute highest-scoring Child Chunks)
       ↓
[Parent Context Assembly]
       │ (Looks up the larger "Parent" chunks for the winning Children)
       │ (Deduplicates Parents, staples Framework ID, Section, and Title metadata)
       ↓
[LLM Micro-Evaluation]
       │ (Staples Company Document Section/Page metadata to the query chunk)
       │ (GPT-4o evaluates the tagged Section against the Parent Context)
       │ (Outputs a structured JSON Report with violations/scores for this section)
       ↓
=========================================================
                      END OF LOOP
=========================================================
       ↓
[Collect All Section Reports]
       │ (Gathers the list of individual JSON micro-verdicts)
       ↓
[The Reducer (Chief Auditor LLM)]
       │ (GPT-4o merges all section reports into one master string)
       │ (Deduplicates identical findings, calculates final synthesized score)
       ↓
[Final Master JSON Output]
"""



import os
import glob
import json
import re
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
    Architecture: JSON-Structured Parents -> Child Chunks -> FAISS+BM25 -> Reranker -> GPT-4o.
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
                 map_chunk_size: int = 500,      
                 map_chunk_overlap: int = 100,
                 openai_api_key: Optional[str] = ''):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_base_path = os.path.join(base_dir, vector_db_base_path)

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.map_chunk_size = map_chunk_size
        self.map_chunk_overlap = map_chunk_overlap

        o_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not o_api_key: raise ValueError("OPENAI_API_KEY missing. Please check your .env file.")
        
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
            First, determine if the <company_document> is actually a company policy, security document, or architecture document relevant to the <framework_context>.

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            If the document is unrelated (e.g., a random story, a recipe, or non-corporate text), set "is_relevant" to false and leave the rest blank.
            If it IS relevant, evaluate it strictly and quote specific article/section numbers.

            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "is_relevant": true,
                "compliance_score": 85,
                "executive_summary": "A detailed 3-4 sentence summary or reason for rejection.",
                "compliant_areas": ["List of precise things they did right"],
                "violations": ["List of specific breaches with framework section references"],
                "recommendations": ["Detailed actionable steps"]
            }}
            """
        )

        self.concise_prompt = ChatPromptTemplate.from_template(
            """
            You are a meticulous, uncompromising Lead Compliance Auditor developed by the Lytrex Team. 

            Step 1: Determine relevance. If the <company_document> is not a corporate or security document related to the <framework_context>, set "is_relevant" to false and stop.
            Step 2: If relevant, strictly evaluate the <company_document> against the <framework_context>.

            Evaluation Rules:
            If the document is unrelated (e.g., a random story, a recipe, or non-corporate text), set "is_relevant" to false and leave the rest blank.
            If it IS relevant, evaluate it strictly and quote specific article/section numbers. 
            Cross-Reference: You MUST check every single requirement in the framework against the company document.
            Be Specific: State the exact discrepancy (e.g., "SAMA requires annual assessments; the company does them biennially").
            Strict Scoring: Start at 100. Deduct 15-20 points for every critical missing control. Be ruthless.


            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "is_relevant": true,
                "internal_audit_reasoning": "Briefly map which framework controls pass or fail before generating the score.",
                "compliance_score": 0,
                "summary": "A strict 1-sentence summary of the compliance posture or reason for rejection.",
                "key_issues": [
                    "Specific Issue 1: Expected [Framework Metric] but found [Company Metric]",
                    "Specific Issue 2: Expected [Framework Metric] but found [Company Metric]"
                ]
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
                "final_compliance_score": 0,
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
                "final_compliance_score": 0,
                "master_summary": "A strict 1-2 sentence overall summary.",
                "all_unique_key_issues": ["Merged list of top critical issues"]
            }}
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
            faiss_results = vectorstore.similarity_search(section.page_content, k=k*5)
            
            tokenized_query = self._tokenize(section.page_content)
            bm25_results = bm25.get_top_n(tokenized_query, all_docs, n=k*5)
            
            combined_children = faiss_results + bm25_results

            unique_children_map = {}
            for child in combined_children:
                if child.page_content not in unique_children_map:
                    unique_children_map[child.page_content] = child

            unique_children = list(unique_children_map.values())
            if not unique_children: continue

            pairs = [[section.page_content, child.page_content] for child in unique_children]
            scores = self.reranker.predict(pairs)
            
            scored_children = sorted(zip(scores, unique_children), key=lambda x: x[0], reverse=True)
            top_k_children = [child for score, child in scored_children[:k]]
            
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
                
            active_prompt = self.concise_prompt if summary_mode else self.detailed_prompt
            chain = active_prompt | self.llm | self.output_parser
            
            try:
                report = chain.invoke({"context": self.prune_text(formatted_context, 12000), "company_doc": self.prune_text(chunk_with_metadata, 4000)})
                all_reports.append(report)
                score = report.get("compliance_score", "N/A")
                print(f"  -> Section {i}: Audited successfully. [Score: {score}]")
            except Exception as e:
                print(f"  -> Section {i}: Failed parsing - {str(e)}")

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
    # THE API BRIDGE METHOD 
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