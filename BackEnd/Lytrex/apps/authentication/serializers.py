from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from .models import User, Individual, Organization

# 1. Serializer to view User details (Output only)
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'user_type']

# 2. Register Serializer (Input logic)
class RegisterSerializer(serializers.ModelSerializer):
    # Optional fields for child classes
    first_name = serializers.CharField(required=False)
    last_name = serializers.CharField(required=False)
    job_title = serializers.CharField(required=False)
    
    company_name = serializers.CharField(required=False)
    industry = serializers.CharField(required=False)
    location = serializers.CharField(required=False)

    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = [
            'username', 'email', 'password', 'user_type',
            'first_name', 'last_name', 'job_title',
            'company_name', 'industry', 'location'
        ]

    def validate_password(self, value):
        """
        Enforces Django's built-in password validators (configured in settings.AUTH_PASSWORD_VALIDATORS).
        By default this catches: too short, too common, entirely numeric, too similar to username.
        """
        try:
            validate_password(value)
        except DjangoValidationError as e:
            raise serializers.ValidationError(e.messages)
        return value

    def create(self, validated_data):
        user_type = validated_data.get('user_type')
        password = validated_data.pop('password')

        if user_type == User.UserType.INDIVIDUAL:
            user = Individual.objects.create(
                username=validated_data['username'],
                email=validated_data['email'],
                user_type=user_type,
                first_name_field=validated_data.get('first_name', ''),
                last_name_field=validated_data.get('last_name', ''),
                job_title=validated_data.get('job_title', '')
            )
        elif user_type == User.UserType.ORGANIZATION:
            user = Organization.objects.create(
                username=validated_data['username'],
                email=validated_data['email'],
                user_type=user_type,
                company_name=validated_data.get('company_name', ''),
                industry=validated_data.get('industry', ''),
                location=validated_data.get('location', '')
            )
        else:
            user = User.objects.create(**validated_data)

        user.set_password(password)
        user.save()
        return user

# 3. Login Serializer
class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()