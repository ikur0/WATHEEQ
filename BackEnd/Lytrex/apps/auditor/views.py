import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Import PDF extractor
from langchain_community.document_loaders import PyPDFLoader

# Import your RAG Class
# Adjust the import path depending on where your RAGCLASS.py is located
from .RAG.main import ComplianceRAG 

@csrf_exempt
def doc(request):
    """
    Returns documentation about the auditor endpoints.
    """
    data = {
        "status": "success",
        "description": "Compliance Auditor API",
        "endpoints": {
            "/match-compliance": {
                "method": "POST",
                "params": {"file": "PDF file to be audited"},
                "action": "Extracts text from uploaded PDF and checks compliance against stored frameworks."
            }
        }
    }
    return JsonResponse(data)

@csrf_exempt
def match_compliance(request):
    """
    1. Receives a PDF file from the user.
    2. Extracts text from the PDF.
    3. Runs RAG Check against the Frameworks DB.
    """
    if request.method == "POST":
        uploaded_file = request.FILES.get('file')
        
        if not uploaded_file:
            return JsonResponse({"error": "No file uploaded. Please send a file with key 'file'."}, status=400)

        # --- 1. Save File Temporarily ---
        # We need to save the file to disk so PyPDFLoader can read it
        temp_file_path = default_storage.save(f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        full_temp_path = os.path.join(default_storage.location, temp_file_path)

        try:
            # --- 2. Extract Text from PDF ---
            loader = PyPDFLoader(full_temp_path)
            pages = loader.load()
            extracted_text = "\n".join([page.page_content for page in pages])

            if not extracted_text:
                return JsonResponse({"error": "Could not extract text from PDF. It might be empty or scanned images."}, status=400)

            # Limit text length to avoid token limits (optional, but recommended)
            # truncating to first 10,000 characters for the 'query'
            query_text = f"Audit the following document content against the standards: \n\n {extracted_text[:10000]}"

            # --- 3. Initialize RAG ---
            # We point to the existing database "LytrexDB"
            # We explicitly set groq_api_key if needed, or rely on os.environ
            rag = ComplianceRAG(
                vector_db_path="LytrexDB", # Ensure this folder exists in your root
                model_name="llama-3.3-70b-versatile" # Ensure using supported model
            )

            # NOTE: We do NOT run ingest_standards() here every time.
            # Ingestion takes time. We assume 'LytrexDB' is already built.
            # If you absolutely MUST ingest every time, uncomment the next line:
            # rag.ingest_standards() 

            # --- 4. Check Compliance ---
            print("🤖 Running Compliance Check...")
            result = rag.check_compliance(query_text)

            # --- 5. Cleanup ---
            # Remove the temp file
            # if os.path.exists(full_temp_path):
            #     os.remove(full_temp_path)

            return JsonResponse({
                "status": "success",
                "audit_result": result["response"],
                "source_docs": result.get("source_documents", [])
            })
        

        except Exception as e:
            # Cleanup on error
            # if os.path.exists(full_temp_path):
            #     os.remove(full_temp_path)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method allowed"}, status=405)