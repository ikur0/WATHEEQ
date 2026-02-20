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


import os
from django.http import JsonResponse
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

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
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded. Please send a file with key 'file'."}, status=400)

    temp_file_path = default_storage.save(f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
    full_temp_path = os.path.join(default_storage.location, temp_file_path)

    try:
        # IMPORTANT: Don’t create this inside every request in production
        rag = ComplianceRAG(
            # vector_db_path="LytrexDB",
            # model_name="llama-3.3-70b-versatile", ## defualt (if you want to change them change them in the class default values)
        )

        # ✅ Pass PDF path (this matches your current ComplianceRAG.check_compliance signature)
        result = rag.check_compliance(full_temp_path, k=4)

        return JsonResponse({
            "status": "success",
            "audit_result": result["response"],
            "source_docs": result.get("source_documents", []),
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    finally:
        # Cleanup (recommended)
        try:
            if os.path.exists(full_temp_path):
                os.remove(full_temp_path)
        except Exception:
            pass