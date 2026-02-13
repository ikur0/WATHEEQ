import uuid
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """
    User model representing the parent class for Individual and Organization.
    Inherits from AbstractUser to handle:
    - Username
    - Password (encryption/hashing)
    - Authentication logic
    """

    # Defining the Enum for user_type
    class UserType(models.TextChoices):
        INDIVIDUAL = 'INDIVIDUAL', 'Individual'
        ORGANIZATION = 'ORGANIZATION', 'Organization'

    # +UUID id
    id = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False
    )

    # +String email
    # Overriding standard email to ensure it is unique and required
    email = models.EmailField(unique=True, blank=False, null=False)

    # +Enum user_type
    user_type = models.CharField(
        max_length=20,
        choices=UserType.choices,
        default=UserType.INDIVIDUAL
    )

    # Note: 'username' and 'password' are provided by AbstractUser automatically.
    
    # Relationship Note:
    # The diagram shows User "has" 0..* ComplianceRecords.
    # In Django, this relationship is defined by a ForeignKey 
    # inside the ComplianceRecord model pointing TO the User.

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

    def __str__(self):
        return f"{self.username} ({self.get_user_type_display()})"

# ---------------------------------------------------------
# Optional: Quick look at how the Child classes would look
# based on the "inherits" arrows in your diagram.
# ---------------------------------------------------------

class Individual(User):
    first_name_field = models.CharField(max_length=100) # Renamed to avoid clash with AbstractUser.first_name
    last_name_field = models.CharField(max_length=100)  # Renamed to avoid clash with AbstractUser.last_name
    job_title = models.CharField(max_length=100)

    class Meta:
        verbose_name = "Individual"

class Organization(User):
    company_name = models.CharField(max_length=255)
    industry = models.CharField(max_length=100)
    location = models.CharField(max_length=255)

    class Meta:
        verbose_name = "Organization"