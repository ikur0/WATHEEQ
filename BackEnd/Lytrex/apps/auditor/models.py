import uuid
from django.db import models
from django.conf import settings

# 1. Framework Model
class Framework(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    version = models.CharField(max_length=50)
    full_content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def sync_to_vector_db(self):
        """
        Logic to trigger RAG ingestion could go here.
        For now, we assume ingestion is done via the script.
        """
        pass

    def __str__(self):
        return f"{self.title} ({self.version})"

# 2. Subclasses (Inheritance)
# In Django, if these don't have extra fields, we can just use the Framework model
# with a 'type' field, but to strictly follow your diagram, we can use Proxy models
# or OneToOne links. Here is a clean Proxy approach if they behave identically:

class ISO_IEC27001(Framework):
    class Meta:
        proxy = True
        verbose_name = "ISO/IEC 27001 Framework"

class CSF(Framework):
    class Meta:
        proxy = True
        verbose_name = "NIST CSF Framework"

class ECC(Framework):
    class Meta:
        proxy = True
        verbose_name = "ECC Framework"


# 3. Compliance Record Model
class ComplianceRecord(models.Model):
    class Status(models.TextChoices):
        COMPLIANT = 'COMPLIANT', 'Compliant'
        NON_COMPLIANT = 'NON_COMPLIANT', 'Non-Compliant'
        PARTIAL = 'PARTIAL', 'Partially Compliant'
        PENDING = 'PENDING', 'Pending Assessment'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Relationship to User (Assuming you have a User model)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="compliance_records", null=True, blank=True)
    
    # Relationship to Framework (The 'assessed_against' line)
    assessed_against = models.ForeignKey(Framework, on_delete=models.CASCADE, related_name="audit_records")
    
    assessment_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    score = models.FloatField(default=0.0)
    report_path = models.FileField(upload_to='reports/', null=True, blank=True)

    def generate_report(self):
        # Logic to generate PDF report
        pass

    def __str__(self):
        return f"Audit {self.id} - {self.status}"