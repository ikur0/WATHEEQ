import os
import json
import glob

# Django core imports
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Django REST Framework imports
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

# Third-party imports
from langchain_community.document_loaders import PyPDFLoader

# Local app imports
from .RAG.main import ComplianceRAG 
from .models import Framework, ComplianceRecord


# =============================================================================
# API DOCUMENTATION
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def doc(request):
    """
    Returns documentation detailing the available endpoints, required parameters, 
    and actions for the Compliance Auditor API.
    """
    data = {
        "status": "success",
        "description": "Compliance Auditor API",
        "endpoints": {
            "/match-compliance": {
                "method": "POST",
                "auth_required": True,
                "params": {
                    "file": "PDF file to be audited (Multipart form-data).",
                    "framework_id": "ID of the Framework being assessed against (Required). 1--> ECC.  2 --> NCA.    3 ----> SAMA",
                    "detailed": "Optional boolean (true/false). Defaults to true."
                },
                "action": "Extracts text, checks compliance, and saves a ComplianceRecord to the DB."
            }
        }
    }
    return Response(data)


# =============================================================================
# COMPLIANCE AUDIT & ANALYSIS
# =============================================================================
@api_view(['POST'])
@permission_classes([IsAuthenticated]) 
def match_compliance(request):
    """
    Core auditing endpoint. Receives a user's PDF, processes it through the 
    Lytrex AI RAG system against a specified framework, and persists the 
    calculated scores and detailed JSON report to the database.
    """
    # --- 0. Auto-Load Frameworks if Database is Empty ---
    if not Framework.objects.exists():
        try:
            _auto_load_frameworks()
        except Exception as e:
            return Response({"error": f"Database was empty, and auto-loading failed: {str(e)}"}, status=500)

    # --- 1. Input Extraction & Validation ---
    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return Response({"error": "No file uploaded. Please send a file with key 'file'."}, status=400)

    framework_id_raw = request.data.get("framework_id")
    if not framework_id_raw:
        return Response({"error": "Missing 'framework_id' in request body."}, status=400)

    try:
        framework = Framework.objects.get(id=int(framework_id_raw))
    except (Framework.DoesNotExist, ValueError):
        return Response({"error": f"Framework with ID {framework_id_raw} not found or invalid."}, status=404)

    detailed_flag_str = str(request.data.get("detailed", "true")).lower()
    is_detailed = detailed_flag_str in ['true', '1', 't', 'y', 'yes']

    # --- 2. Temporary File Storage ---
    temp_file_path = default_storage.save(f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
    full_temp_path = os.path.join(default_storage.location, temp_file_path)

    try:
        # --- 3. AI Processing (RAG) ---
        rag = ComplianceRAG()
        result = rag.check_compliance(full_temp_path, k=4, detailed=is_detailed)

        if "error" in result:
            return Response({"status": "error", "message": result["error"]}, status=500)

        # --- 4. Database Persistence ---
        score = float(result.get("compliance_score", 0))

        if score >= 90:
            calc_status = ComplianceRecord.Status.COMPLIANT
        elif score >= 50:
            calc_status = ComplianceRecord.Status.PARTIAL
        else:
            calc_status = ComplianceRecord.Status.NON_COMPLIANT

        record = ComplianceRecord(
            user=request.user,
            assessed_against=framework,
            score=score,
            status=calc_status
        )

        report_json_str = json.dumps(result, indent=4)
        report_filename = f"audit_report_{record.id}.json"
        
        record.report_path.save(report_filename, ContentFile(report_json_str))

        return Response({
            "status": "success",
            "record_id": record.id,
            "calculated_status": calc_status,
            "is_detailed_response": is_detailed,
            "audit_result": result, 
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

    finally:
        # --- 5. Cleanup ---
        try:
            if os.path.exists(full_temp_path):
                os.remove(full_temp_path)
        except Exception:
            pass


# =============================================================================
# RECORD RETRIEVAL & HISTORY
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_compliance_record(request, record_id):
    """
    Retrieves a specific ComplianceRecord by its ID.
    Reads the associated JSON report file from storage and injects its contents 
    directly into the API response for the frontend to render.
    """
    try:
        # Security: Filter by request.user to prevent unauthorized access to other companies' audits
        record = ComplianceRecord.objects.get(id=record_id, user=request.user)
        
        report_data = None
        if record.report_path:
            try:
                # Safely open and read the JSON file associated with this record
                with record.report_path.open('r') as f:
                    file_content = f.read()
                    
                    # Cloud storage backends (like AWS S3) often return bytes instead of strings
                    if isinstance(file_content, bytes):
                        file_content = file_content.decode('utf-8')
                        
                    report_data = json.loads(file_content)
            except json.JSONDecodeError:
                report_data = {"error": "The report file exists but is not valid JSON."}
            except Exception as e:
                report_data = {"error": f"Failed to read report file: {str(e)}"}

        data = {
            "id": str(record.id),
            "framework_title": record.assessed_against.title if record.assessed_against else None,
            "framework_version": record.assessed_against.version if record.assessed_against else None,
            "assessment_date": record.assessment_date.isoformat(),
            "status": record.status,
            "score": record.score,
            "report_data": report_data
        }

        return Response({"status": "success", "data": data})

    except ComplianceRecord.DoesNotExist:
        return Response({"error": "Record not found or you do not have permission to view it."}, status=404)
    except ValueError:
        return Response({"error": "Invalid record ID format."}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_user_compliance_records(request):
    """
    Returns a lightweight, chronological list of all past compliance audits 
    for the logged-in user. Ideal for populating user dashboards.
    """
    try:
        records = ComplianceRecord.objects.filter(user=request.user).order_by('-assessment_date')
        
        records_list = [
            {
                "id": str(record.id),
                "framework_title": record.assessed_against.title if record.assessed_against else "Unknown Framework",
                "score": record.score,
                "status": record.status,
                "assessment_date": record.assessment_date.isoformat(),
            }
            for record in records
        ]
        
        return Response({
            "status": "success",
            "count": len(records_list),
            "records": records_list
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# =============================================================================
# SYSTEM UTILITIES
# =============================================================================
def _auto_load_frameworks():
    """
    Internal helper function to scan the frameworks folder and load PDFs 
    into the database if it is currently empty.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, 'RAG', 'frameworks')

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"Frameworks directory not found: {pdf_dir}")

    pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    if not pdf_files:
        raise ValueError(f"No PDFs found in {pdf_dir} to load.")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        title = os.path.splitext(filename)[0] 
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        if not full_text.strip():
            continue

        Framework.objects.update_or_create(
            title=title,
            defaults={
                'version': '1.0', 
                'full_content': full_text
            }
        )