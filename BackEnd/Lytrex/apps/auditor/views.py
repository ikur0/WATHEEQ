import os
import json
import glob

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from langchain_community.document_loaders import PyPDFLoader

from .RAG.main import ComplianceRAG
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
                    "framework_id": "ID of the Framework being assessed against (Required). 1 --> ECC.  2 --> NCA.  3 --> SAMA",
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

    framework_id_raw = request.data.get("framework_id")
    if not framework_id_raw:
        return Response({"error": "Missing 'framework_id' in request body."}, status=400)

    try:
        framework_id = int(framework_id_raw)
    except ValueError:
        return Response({"error": "'framework_id' must be an integer."}, status=400)

    framework_name = FRAMEWORK_ID_MAP.get(framework_id)
    if not framework_name:
        return Response({
            "error": f"Invalid framework_id '{framework_id}'. Valid options: 1=ECC, 2=NCA, 3=SAMA"
        }, status=400)

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
        
        # Matches the updated main.py signature: check_compliance(target_pdf_path, k, detailed)
        fw_result = rag.check_compliance(
            target_pdf_path=full_temp_path,
            k=5,
            detailed=is_detailed
        )

        # Catch empty or completely failed execution
        if not fw_result:
            return Response({"error": "The AI returned an empty response."}, status=500)

        # --- Error & Relevance Check ---
        # Matches the error dictionaries returned by the new main.py
        if "error" in fw_result:
            err_msg = fw_result["error"]
            
            # Catch the specific Relevance Gate rejection from main.py
            if "Document Rejected" in err_msg:
                return Response({
                    "status": "irrelevant",
                    "message": err_msg,
                    "reasoning": fw_result.get("llm_reasoning", "The AI determined this document is not a corporate or security policy."),
                    "framework": framework_name
                }, status=422)
            
            # Standard hard failures (e.g., file not found, json parse failed)
            return Response({"error": err_msg}, status=500)

        # --- 4. Persist ComplianceRecord ---
        score = float(fw_result.get("compliance_score", 0))

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frameworks_root = os.path.join(base_dir, 'RAG', 'frameworks')

    if not os.path.exists(frameworks_root):
        raise FileNotFoundError(f"Frameworks root not found: {frameworks_root}")

    loaded_any = False

    for fw_name in ComplianceRAG.SUPPORTED_FRAMEWORKS:
        fw_dir = os.path.join(frameworks_root, fw_name)
        pdf_files = glob.glob(os.path.join(fw_dir, '*.pdf'))

        if not pdf_files:
            continue

        full_text_parts = []
        for pdf_path in pdf_files:
            docs = PyPDFLoader(pdf_path).load()
            full_text_parts.extend(doc.page_content for doc in docs)

        full_text = "\n\n".join(full_text_parts).strip()
        if not full_text:
            continue

        Framework.objects.update_or_create(
            title=fw_name,
            defaults={"version": "1.0", "full_content": full_text}
        )
        loaded_any = True

    if not loaded_any:
        raise ValueError(f"No PDFs found under {frameworks_root}/NCA|ECC|SAMA/")