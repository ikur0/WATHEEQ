import os
import glob
from typing import Optional, Dict, Any

from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document


class ComplianceRAG:
    """
    A production-ready RAG class for Compliance Checking using Nomic AI.
    Maintains separate vector databases for each framework (NCA, ECC, SAMA).
    Uses Parent-Child chunking logic with Oversampling & Deduplication.
    """

    def __init__(self,
                 pdf_source_dir: str = "frameworks",
                 vector_db_base_path: str = "LytrexDB_Nomic",
                 embedding_model: str = "nomic-embed-text-v1.5",
                 model_name: str = "llama-3.3-70b-versatile",
                 chunk_size: int = 6000, # Parent Chunk Size
                 chunk_overlap: int = 800,
                 nomic_api_key: Optional[str] = '',
                 groq_api_key: Optional[str] = ''):

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_source_dir = os.path.join(base_dir, pdf_source_dir)
        self.vector_db_base_path = os.path.join(base_dir, vector_db_base_path)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # --- Nomic AI Embedding Setup ---
        n_api_key = nomic_api_key or os.getenv("NOMIC_API_KEY")
        if not n_api_key:
            raise ValueError("NOMIC_API_KEY is missing.")
            
        self.embeddings = NomicEmbeddings(
            model=embedding_model, 
            nomic_api_key=n_api_key,
            dimensionality=768
        )

        # --- Groq LLM Setup ---
        g_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not g_api_key:
             raise ValueError("GROQ_API_KEY is missing.")

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=g_api_key,
            model_name=model_name
        )

        self._setup_prompts()
        self.output_parser = JsonOutputParser()

    def _setup_prompts(self):
        # --- DETAILED PROMPT ---
        self.detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite, uncompromising Lead Compliance Auditor AI developed by the Lytrex Team. 
            Your primary objective is to protect national infrastructure by enforcing absolute adherence to the provided Framework.

            PHASE 1: RELEVANCE GATE (CRITICAL)
            Analyze the <company_document>. If it is not a formal corporate policy, security manual, or technical standard related to the <framework_context> (e.g., it is a recipe, fictional story, or general news), set "is_relevant" to false and stop immediately.

            PHASE 2: RIGOROUS EVALUATION
            If relevant, perform a granular, line-by-line audit. Compare every specific technical metric (bit-lengths, algorithms, timeframes) against the <framework_context>.

            UNBIASED AUDIT RULES:
            1. Zero Tolerance for "Close Enough": If the framework requires 3072-bit keys and the company uses 2048-bit, this is a CRITICAL VIOLATION.
            2. Personas: Be ruthless. Do not praise general effort; only acknowledge exact technical alignment.
            3. Contextual Strength: Strictly differentiate between "MODERATE" and "ADVANCED" levels. Applying a "MODERATE" control to an "ADVANCED" requirement is a failure.
            4. Specificity: You MUST quote the exact Section or Article numbers from the <framework_context> for every finding.
            5. Scoring: Start at 100. Deduct 15-20 points for every critical control failure (e.g., weak PRNG, reused IVs, insufficient key lengths, or unauthorized storage modules).

            <framework_context>
            {context}
            </framework_context>

            <company_document>
            {company_doc}
            </company_document>

            Respond ONLY with a valid JSON object matching this exact structure:
            {{
                "is_relevant": true,
                "internal_audit_reasoning": "A technical mapping of company metrics against framework requirements. Explicitly justify the score based on deductions.",
                "compliance_score": 0,
                "executive_summary": "A 3-4 sentence professional audit conclusion. Highlight the most dangerous security gaps found.",
                "compliant_areas": [
                    "Control [Section X]: [Exact technical metric that matched the framework]"
                ],
                "violations": [
                    "Violation [Section Y]: [Specific discrepancy. State exactly what was found vs. what is required]"
                ],
                "recommendations": [
                    "Remediation [Section Y]: [Clear technical instruction to bring the violation into absolute compliance]"
                ]
            }}
            """
        )

        # --- CONCISE PROMPT ---
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

    def _get_fw_db_path(self, framework_name: str) -> str:
        return os.path.join(self.vector_db_base_path, framework_name.upper())

    def _load_fw_vectorstore(self, framework_name: str):
        path = self._get_fw_db_path(framework_name)
        index_file = os.path.join(path, "index.faiss")
        if os.path.exists(index_file):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    def ingest_single_framework(self, framework_name: str):
        print(f"Creating Hierarchical DB for framework: {framework_name}")
        
        if framework_name.upper() == "ALL":
            pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, "**", "*.pdf"), recursive=True)
        else:
            fw_folder = os.path.join(self.pdf_source_dir, framework_name.upper())
            pdf_paths = glob.glob(os.path.join(fw_folder, "*.pdf"))
        
        if not pdf_paths:
            return None

        all_docs = []
        for path in pdf_paths:
            all_docs.extend(PyPDFLoader(path).load())

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50 
        )

        parent_docs = parent_splitter.split_documents(all_docs)
        
        final_chunks = []
        for i, parent in enumerate(parent_docs):
            children = child_splitter.split_text(parent.page_content)
            for child in children:
                new_doc = parent.copy()
                new_doc.page_content = child 
                new_doc.metadata["parent_content"] = parent.page_content 
                final_chunks.append(new_doc)

        vectorstore = FAISS.from_documents(final_chunks, self.embeddings)
        
        out_path = self._get_fw_db_path(framework_name)
        os.makedirs(out_path, exist_ok=True)
        vectorstore.save_local(out_path)
        return vectorstore

    def evaluate_with_llm(self, formatted_context: str, company_doc: str, detailed: bool = False) -> Dict[str, Any]:
        active_prompt = self.detailed_prompt if detailed else self.concise_prompt
        chain = active_prompt | self.llm | self.output_parser

        try:
            print("\nSending to LLM for evaluation...")
            result = chain.invoke({
                "context": formatted_context,
                "company_doc": company_doc
            })

            if not result or not isinstance(result, dict):
                return {"error": "Document Rejected: AI returned invalid response."}

            if not result.get("is_relevant", True):
                return {
                    "error": "Document Rejected: The uploaded file does not appear to be a relevant security or policy document.",
                    "llm_reasoning": result.get("summary", result.get("executive_summary", "No reason provided."))
                }
            return result
        except Exception as e:
            return {"error": f"Failed to parse LLM response: {str(e)}"}

    def check_compliance(self, target_pdf_path: str, framework_name: str, k: int = 5, run_llm: bool = True, detailed: bool = False) -> Dict[str, Any]:
        vectorstore = self._load_fw_vectorstore(framework_name)
        if not vectorstore:
            vectorstore = self.ingest_single_framework(framework_name)
            if not vectorstore:
                return {"error": f"No Standards found for: {framework_name}"}

        if not os.path.exists(target_pdf_path):
            return {"error": f"Target PDF not found at path: {target_pdf_path}"}

        documents = PyPDFLoader(target_pdf_path).load()
        target_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        full_company_text = "\n".join([doc.page_content for doc in documents])

        retrieved_parents = []
        for chunk in target_chunks:
            print(f"\nquery:\n{chunk.page_content}")
            print("\\" * 100)
            
            # --- NEW LOGIC: Oversample & Deduplicate ---
            # Retrieve a lot of child chunks (k * 5) to guarantee we find k distinct parents
            fetch_k = k * 5
            results = vectorstore.similarity_search(chunk.page_content, k=fetch_k)
            
            unique_parents_for_chunk = []
            seen_parents = set()
            
            for res in results:
                parent_text = res.metadata.get("parent_content", res.page_content)
                if parent_text not in seen_parents:
                    seen_parents.add(parent_text)
                    unique_parents_for_chunk.append(parent_text)
                
                # Stop exactly when we hit 'k' unique parents
                if len(unique_parents_for_chunk) == k:
                    break
            
            # Print and store the exact 'k' unique parents
            for idx, parent_text in enumerate(unique_parents_for_chunk, 1):
                print(f"context {idx} (Retrieved Unique Parent)\n{parent_text[:500]}...") 
                print("-" * 40)
                
            retrieved_parents.extend(unique_parents_for_chunk)
            print("\n\n")

        # Global deduplication across all queries (preserves order of importance)
        formatted_context = "\n\n---\n\n".join(list(dict.fromkeys(retrieved_parents)))

        if not formatted_context:
            return {"error": "No relevant standards found in the database."}

        if not run_llm:
            return {"formatted_context": formatted_context, "company_doc": full_company_text}

        return self.evaluate_with_llm(formatted_context, full_company_text, detailed)

    def check_compliance_text(self, text: str, framework_name: str, k: int = 5, run_llm: bool = True, detailed: bool = False) -> Dict[str, Any]:
        vectorstore = self._load_fw_vectorstore(framework_name)
        if not vectorstore:
            vectorstore = self.ingest_single_framework(framework_name)
            if not vectorstore:
                return {"error": f"No Standards found for: {framework_name}"}

        documents = [Document(page_content=text)]
        target_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        full_company_text = text

        retrieved_parents = []
        for chunk in target_chunks:
            print(f"\nquery:\n{chunk.page_content}")
            print("\\" * 100)
            
            # --- NEW LOGIC: Oversample & Deduplicate ---
            fetch_k = k * 5
            results = vectorstore.similarity_search(chunk.page_content, k=fetch_k)
            
            unique_parents_for_chunk = []
            seen_parents = set()
            
            for res in results:
                parent_text = res.metadata.get("parent_content", res.page_content)
                if parent_text not in seen_parents:
                    seen_parents.add(parent_text)
                    unique_parents_for_chunk.append(parent_text)
                
                if len(unique_parents_for_chunk) == k:
                    break
            
            for idx, parent_text in enumerate(unique_parents_for_chunk, 1):
                print(f"context {idx} (Retrieved Unique Parent)\n{parent_text[:1000]}...")
                print("-" * 40)
                
            retrieved_parents.extend(unique_parents_for_chunk)
            print("\n\n")

        # Global deduplication across all queries
        formatted_context = "\n\n---\n\n".join(list(dict.fromkeys(retrieved_parents)))

        if not formatted_context:
            return {"error": "No relevant standards found in the database."}

        if not run_llm:
            return {"formatted_context": formatted_context, "company_doc": full_company_text}

        return self.evaluate_with_llm(formatted_context, full_company_text, detailed)