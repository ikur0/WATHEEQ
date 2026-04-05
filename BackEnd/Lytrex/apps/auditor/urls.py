from django.urls import path
from . import views

urlpatterns = [
    # Path 1: Explanation of the endpoint
    # path('', views.doc, name="auditor.audit.doc"),
    
    # Path 2: Upload PDF -> Extract -> Audit
    path('match-compliance', views.match_compliance, name="auditor.match-compliance"),

    path('compliance-records/all', views.list_user_compliance_records, name='auditor.all_compliance_records'),
    path('compliance-records/<uuid:record_id>', views.get_compliance_record, name='auditor.get_compliance_record'),
]