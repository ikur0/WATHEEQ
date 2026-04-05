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
    Includes a Relevance Gate to reject non-framework documents.
    """

    def __init__(self,
                 pdf_source_dir: str = "frameworks",
                 vector_db_path: str = "LytrexDB",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
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

        # Secure API Key Loading
        api_key ='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'
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing. Please set it before running.")

        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name=model_name
        )

        # --- DETAILED PROMPT (With Relevance Gate) ---
        self.detailed_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite, uncompromising Lead Compliance Auditor AI developed by the Lytrex Team.

            Step 1: Determine relevance. If the <company_document> is not a corporate or security document related to the <framework_context>, set "is_relevant" to false and stop.
            Step 2: If relevant, strictly evaluate the <company_document> against the <framework_context>.

            Evaluation Rules:
            If the document is unrelated (e.g., a random story, a recipe, or non-corporate text), set "is_relevant" to false and leave the rest blank.
            If it IS relevant, evaluate it strictly and quote specific article/section numbers. 
            Cross-Reference: You MUST check every single requirement in the framework against the company document.
            Be Specific: State the exact discrepancy or alignment (e.g., "SAMA requires annual assessments; the company does them biennially").
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
                "internal_audit_reasoning": "Briefly map which framework controls pass or fail before calculating the final score.",
                "compliance_score": 85,
                "executive_summary": "A detailed 3-4 sentence summary of the compliance posture or reason for rejection.",
                "compliant_areas": [
                    "Control [Section X]: [Precise description of what they did right]"
                ],
                "violations": [
                    "Violation [Section Y]: [Specific breach and why it fails the framework]"
                ],
                "recommendations": [
                    "Actionable step to remediate Violation [Section Y]"
                ]
            }}
            """
        )

        # --- CONCISE PROMPT (With Relevance Gate & Unified JSON) ---
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

        self.output_parser = JsonOutputParser()

    def _load_vectorstore(self):
        if os.path.exists(self.vector_db_path):
            return FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    def ingest_standards(self):
        print(f"Checking for frameworks in: {self.pdf_source_dir}")
        pdf_paths = glob.glob(os.path.join(self.pdf_source_dir, "*.pdf"))
        if not pdf_paths:
            print("No framework PDFs found to ingest.")
            return

        all_documents = []
        for path in pdf_paths:
            print(f"Ingesting framework: {path}")
            all_documents.extend(PyPDFLoader(path).load())

        if not all_documents: return

        chunks = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                chunk_overlap=self.chunk_overlap).split_documents(all_documents)

        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        self.vectorstore.save_local(self.vector_db_path)
        print("Framework ingestion complete.")

    def check_compliance(self, target_pdf_path: str, k: int = 5, detailed: bool = False) -> Dict[str, Any]:
        if not self.vectorstore:
            self.ingest_standards()
            self.vectorstore = self._load_vectorstore()
            if not self.vectorstore:
                return {"error": "Database not found and could not be created."}

        if not os.path.exists(target_pdf_path):
            return {"error": f"Target PDF not found at path: {target_pdf_path}"}

        print(f"Analyzing target document: {target_pdf_path}")
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

        if not formatted_context:
            return {"error": "No relevant standards found in the database."}

        # Select the prompt based on the 'detailed' flag
        active_prompt = self.detailed_prompt if detailed else self.concise_prompt
        chain = active_prompt | self.llm | self.output_parser

        try:
            print("Sending to LLM for evaluation...")
            result = chain.invoke({
                "context": formatted_context,
                "company_doc": full_company_text
            })

            # --- SAFETY CHECK: Prevent NoneType errors on empty AI responses ---
            if not result or not isinstance(result, dict):
                return {
                    "error": "Document Rejected: The uploaded file does not appear to be a relevant security or policy document.",
                    "llm_reasoning": "The AI returned an empty or non-JSON response."
                }

            # THE RELEVANCE GATE
            if not result.get("is_relevant", True):
                return {
                    "error": "Document Rejected: The uploaded file does not appear to be a relevant security or policy document.",
                    "llm_reasoning": result.get("summary", result.get("executive_summary", "No reason provided."))
                }

            return result

        except Exception as e:
            return {"error": f"Failed to parse LLM response: {str(e)}"}


# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    import json

    # Initialize the RAG system
    rag_system = ComplianceRAG()

    # Define your target test file here
    # Make sure this path actually exists on your machine!
    target_file = 'company Compliance test/TechCorp Information Security Policy Version.pdf'

    # Run the compliance check (set detailed=False to use the concise, strict prompt)
    report = rag_system.check_compliance(target_pdf_path=target_file, detailed=False)

    # Print the output beautifully
    print("\n" + "=" * 50)
    print("FINAL AUDIT REPORT")
    print("=" * 50)
    print(json.dumps(report, indent=4))