from django.urls import path
from . import views

urlpatterns = [
    # Path 1: Explanation of the endpoint
    path('', views.doc, name="auditor.audit.doc"),
    
    # Path 2: Upload PDF -> Extract -> Audit
    path('match-compliance', views.match_compliance, name="auditor.match-compliance"),
]