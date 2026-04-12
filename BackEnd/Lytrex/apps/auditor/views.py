import os
import json

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

# Import directly from the renamed rag.py file
from .RAG.rag import ComplianceRAG
from .models import Framework, ComplianceRecord


FRAMEWORK_ID_MAP = {
    1: "ECC",
    2: "NCA",
    3: "SAMA",
}


# =============================================================================
# API DOCUMENTATION
# =============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def doc(request):
    data = {
        "status": "success",
        "description": "Compliance Auditor API",
        "endpoints": {
            "/match-compliance": {
                "method": "POST",
                "auth_required": True,
                "params": {
                    "file": "PDF file to be audited (Multipart form-data).",
                    "framework_id": "ID or Name of the Framework being assessed against (Required). e.g., 2 or 'NCA'.",
                    "detailed": "Optional boolean (true/false). Defaults to false for concise prompt."
                },
                "action": "Checks compliance against the selected framework. Saves a ComplianceRecord to the DB.",
                "response_shape": {
                    "status": "success",
                    "record_id": "uuid",
                    "framework": "NCA",
                    "score": 85.0,
                    "calculated_status": "PARTIAL",
                    "is_detailed_response": False,
                    "audit_result": {}
                }
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
    # --- 0. Auto-load frameworks if DB is empty ---
    if not Framework.objects.exists():
        try:
            _auto_load_frameworks()
        except Exception as e:
            return Response({"error": f"Auto-loading frameworks failed: {str(e)}"}, status=500)

    # --- 1. Validate inputs ---
    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return Response({"error": "No file uploaded. Send a PDF with key 'file'."}, status=400)

    # Allow either integer ID or string name for framework_id
    framework_id_raw = request.data.get("framework_id")
    if not framework_id_raw:
        return Response({"error": "Missing 'framework_id' in request body."}, status=400)

    fw_input = str(framework_id_raw).strip().upper()

    if fw_input in FRAMEWORK_ID_MAP.values():
        framework_name = fw_input
    else:
        try:
            framework_id = int(fw_input)
            framework_name = FRAMEWORK_ID_MAP.get(framework_id)
        except ValueError:
            return Response({"error": "'framework_id' must be an integer ID (1,2,3) or a valid name (ECC, NCA, SAMA)."}, status=400)

    if not framework_name:
        return Response({
            "error": f"Invalid framework '{fw_input}'. Valid options: 1=ECC, 2=NCA, 3=SAMA"
        }, status=400)

    # Check if framework is in DB
    framework_obj = Framework.objects.filter(title__iexact=framework_name).first()
    if not framework_obj:
        try:
            _auto_load_frameworks()
        except Exception:
            pass
        framework_obj = Framework.objects.filter(title__iexact=framework_name).first()
        if not framework_obj:
            return Response({"error": f"Framework '{framework_name}' not found in DB."}, status=404)

    detailed_flag_str = str(request.data.get("detailed", "false")).lower()
    is_detailed = detailed_flag_str in ['true', '1', 't', 'y', 'yes']

    # --- 2. Save temp file ---
    temp_file_path = default_storage.save(
        f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read())
    )
    full_temp_path = os.path.join(default_storage.location, temp_file_path)

    try:
        # --- 3. Run RAG ---
        rag = ComplianceRAG()
        
        # Calls the wrapper method in rag.py
        fw_result = rag.check_compliance(
            target_pdf_path=full_temp_path,
            framework_name=framework_name, 
            k=10, 
            detailed=is_detailed
        )

        if not fw_result:
            return Response({"error": "The AI returned an empty response."}, status=500)

        # --- Error & Relevance Check ---
        if "error" in fw_result:
            err_msg = fw_result["error"]
            
            if "Document Rejected" in err_msg:
                return Response({
                    "status": "irrelevant",
                    "message": err_msg,
                    "reasoning": fw_result.get("llm_reasoning", "The AI determined this document is not a corporate or security policy."),
                    "framework": framework_name
                }, status=422)
            
            return Response({"error": err_msg}, status=500)

        # --- 4. Persist ComplianceRecord ---
        # Maps final JSON keys from the Master Report output
        score = float(fw_result.get("final_compliance_score", 0))

        if score >= 90:
            calc_status = ComplianceRecord.Status.COMPLIANT
        elif score >= 50:
            calc_status = ComplianceRecord.Status.PARTIAL
        else:
            calc_status = ComplianceRecord.Status.NON_COMPLIANT

        record = ComplianceRecord.objects.create(
            user=request.user,
            assessed_against=framework_obj,
            score=score,
            status=calc_status
        )

        # Save the full audit report as a JSON file
        report_json_str = json.dumps(fw_result, indent=4)
        report_filename = f"audit_{framework_name}_{record.id}.json"
        record.report_path.save(report_filename, ContentFile(report_json_str))
        record.save()

        return Response({
            "status": "success",
            "record_id": str(record.id),
            "framework": framework_name,
            "score": score,
            "calculated_status": calc_status,
            "is_detailed_response": is_detailed,
            "audit_result": fw_result,
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

    finally:
        # Clean up temp file
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
    try:
        record = ComplianceRecord.objects.get(id=record_id, user=request.user)

        report_data = None
        if record.report_path:
            try:
                with record.report_path.open('r') as f:
                    content = f.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    report_data = json.loads(content)
            except json.JSONDecodeError:
                report_data = {"error": "Report file exists but is not valid JSON."}
            except Exception as e:
                report_data = {"error": f"Failed to read report: {str(e)}"}

        return Response({
            "status": "success",
            "data": {
                "id": str(record.id),
                "framework_title": record.assessed_against.title if record.assessed_against else None,
                "framework_version": record.assessed_against.version if record.assessed_against else None,
                "assessment_date": record.assessment_date.isoformat(),
                "status": record.status,
                "score": record.score,
                "report_data": report_data
            }
        })

    except ComplianceRecord.DoesNotExist:
        return Response({"error": "Record not found or access denied."}, status=404)
    except ValueError:
        return Response({"error": "Invalid record ID format."}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_user_compliance_records(request):
    try:
        records = ComplianceRecord.objects.filter(user=request.user).order_by('-assessment_date')
        return Response({
            "status": "success",
            "count": records.count(),
            "records": [
                {
                    "id": str(r.id),
                    "framework_title": r.assessed_against.title if r.assessed_against else "Unknown",
                    "score": r.score,
                    "status": r.status,
                    "assessment_date": r.assessment_date.isoformat(),
                }
                for r in records
            ]
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

def _auto_load_frameworks():
    """
    Asks the RAG engine to parse the PDFs and return the text, 
    keeping the Langchain PyPDFLoader logic completely out of Django.
    """
    rag = ComplianceRAG()
    loaded_any = False

    for fw_name in ["ECC", "NCA", "SAMA"]:
        # Let the RAG engine handle the file reading
        full_text = rag.get_framework_full_text(fw_name)
        
        if not full_text:
            continue

        Framework.objects.update_or_create(
            title=fw_name,
            defaults={"version": "1.0", "full_content": full_text}
        )
        loaded_any = True

    if not loaded_any:
        raise ValueError("No PDFs found under RAG/frameworks/ for ECC, NCA, or SAMA.")