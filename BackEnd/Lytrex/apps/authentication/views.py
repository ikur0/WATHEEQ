from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.reverse import reverse
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from django.contrib.auth import authenticate
from .serializers import RegisterSerializer, LoginSerializer, UserSerializer

# --- Helper Function ---
def get_tokens_for_user(user):
    """
    Generates a pair of JWT tokens (Access + Refresh) for a given user.
    """
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

# --- 0. Auth Root View (Discovery Endpoint) ---
@api_view(["GET"])
@permission_classes([AllowAny])
def auth_root_view(request, format=None):
    """
    Lists all available authentication endpoints with usage examples.
    """
    return Response({
        "message": "Authentication API - JWT Mode",
        "endpoints": {
            # --- REGISTER ---
            "register": {
                "url": reverse('register', request=request, format=format),
                "method": "POST",
                "description": "Create account and receive Access + Refresh tokens.",
                "example_usage": {
                    "request_body_individual": {
                        "username": "john_doe",
                        "email": "john@example.com",
                        "password": "securepassword123",
                        "user_type": "INDIVIDUAL",
                        "first_name": "John",
                        "last_name": "Doe",
                        "job_title": "Software Engineer"
                    },
                    "request_body_organization": {
                        "username": "tech_corp",
                        "email": "admin@techcorp.com",
                        "password": "securepassword123",
                        "user_type": "ORGANIZATION",
                        "company_name": "Tech Corp",
                        "industry": "IT Services",
                        "location": "New York, USA"
                    },
                    "response_success": {
                        "user": {
                            "id": "uuid-string",
                            "username": "john_doe",
                            "email": "john@example.com",
                            "user_type": "INDIVIDUAL"
                        },
                        "tokens": {
                            "access": "eyJ0eX... (Valid for 30 minutes)",
                            "refresh": "eyJ0eX... (Valid for 1 day)"
                        },
                        "message": "User created successfully"
                    }
                }
            },

            # --- LOGIN ---
            "login": {
                "url": reverse('login', request=request, format=format),
                "method": "POST",
                "description": "Login to receive Access + Refresh tokens.",
                "example_usage": {
                    "request_body": {
                        "username": "john_doe",
                        "password": "securepassword123"
                    },
                    "response_success": {
                        "user": {
                            "id": "uuid-string",
                            "username": "john_doe",
                            "email": "john@example.com",
                            "user_type": "INDIVIDUAL"
                        },
                        "tokens": {
                            "access": "eyJ0eX... (Valid for 30 minutes)",
                            "refresh": "eyJ0eX... (Valid for 1 day)"
                        },
                        "message": "Login successful"
                    }
                }
            },

            # --- REFRESH ---
            "refresh": {
                "url": reverse('token_refresh', request=request, format=format),
                "method": "POST",
                "description": "Get a new Access Token using your Refresh Token.",
                "example_usage": {
                    "request_body": {
                        "refresh": "<your_refresh_token_string>"
                    },
                    "response_success": {
                        "access": "<new_access_token_valid_for_30m>",
                        "refresh": "<optional_new_refresh_token>"
                    }
                }
            },

            # --- LOGOUT ---
            "logout": {
                "url": reverse('logout', request=request, format=format),
                "method": "POST",
                "description": "Blacklist your Refresh Token so it cannot be used again.",
                "example_usage": {
                    "headers": {
                        "Authorization": "Bearer <your_access_token_string>"
                    },
                    "request_body": {
                        "refresh": "<your_refresh_token_string>"
                    },
                    "response_success": {
                        "message": "Successfully logged out."
                    }
                }
            },

            # ✅ FIX: give the list a key (so dict is valid)
            "step_by_step_example": [
                {
                    "step": 1,
                    "action": "REGISTER (Create Organization Account)",
                    "endpoint": "/authentication/register/",
                    "method": "POST",
                    "request_body": {
                        "username": "omega_corp_admin",
                        "email": "contact@omegacorp.com",
                        "password": "StrongPassword123!",
                        "user_type": "ORGANIZATION",
                        "company_name": "Omega Cyber Solutions",
                        "industry": "Cybersecurity",
                        "location": "Riyadh, Saudi Arabia"
                    },
                    "expected_response": {
                        "user": {
                            "id": "a1b2c3d4-e5f6-7890-1234-56789abcdef0",
                            "username": "omega_corp_admin",
                            "email": "contact@omegacorp.com",
                            "user_type": "ORGANIZATION"
                        },
                        "tokens": {
                            "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY3ODkw... (long string)",
                            "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNj... (valid for 30m)"
                        },
                        "message": "User created successfully"
                    }
                },
                {
                    "step": 2,
                    "action": "LOGIN (Get Tokens for existing user)",
                    "endpoint": "/authentication/login/",
                    "method": "POST",
                    "request_body": {
                        "username": "omega_corp_admin",
                        "password": "StrongPassword123!"
                    },
                    "expected_response": {
                        "user": {
                            "username": "omega_corp_admin",
                            "email": "contact@omegacorp.com",
                            "user_type": "ORGANIZATION"
                        },
                        "tokens": {
                            "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY3ODkw...",
                            "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNj..."
                        },
                        "message": "Login successful"
                    }
                },
                {
                    "step": 3,
                    "action": "REFRESH TOKEN (Get new Access Token after 30m)",
                    "endpoint": "/authentication/refresh/",
                    "method": "POST",
                    "request_body": {
                        "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY3ODkw..."
                    },
                    "expected_response": {
                        "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.NEW_ACCESS_TOKEN_STRING...",
                        "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.OPTIONAL_NEW_REFRESH_TOKEN..."
                    }
                },
                {
                    "step": 4,
                    "action": "LOGOUT (Blacklist the Refresh Token)",
                    "endpoint": "/authentication/logout/",
                    "method": "POST",
                    "headers": {
                        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.NEW_ACCESS_TOKEN_STRING..."
                    },
                    "request_body": {
                        "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY3ODkw..."
                    },
                    "expected_response": {
                        "message": "Successfully logged out."
                    }
                }
            ],
        }
    }, status=status.HTTP_200_OK)
# --- 1. Registration View ---
class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            
            # Generate JWT Tokens
            tokens = get_tokens_for_user(user)
            
            return Response({
                "user": UserSerializer(user).data,
                "tokens": tokens,
                "message": "User created successfully"
            }, status=status.HTTP_201_CREATED)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# --- 2. Login View ---
class LoginView(generics.GenericAPIView):
    serializer_class = LoginSerializer
    permission_classes = [AllowAny]

    def post(self, request):
        # 1. Validate that username/password fields exist
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # 2. Extract credentials
        username = serializer.validated_data['username']
        password = serializer.validated_data['password']

        # 3. Authenticate against the database
        user = authenticate(username=username, password=password)
        
        if not user:
            return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)

        # 4. Generate JWT Tokens
        tokens = get_tokens_for_user(user)
        
        return Response({
            "user": UserSerializer(user).data,
            "tokens": tokens,
            # Use getattr to safely get user_type, defaulting to 'Unknown' if missing
            "user_type": getattr(user, 'user_type', 'Unknown'), 
            "message": "Login successful"
        }, status=status.HTTP_200_OK)

# --- 3. Logout View ---
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    Logout in JWT means blacklisting the Refresh Token so it can't be used again.
    The client must send the 'refresh' token in the body.
    """
    try:
        refresh_token = request.data["refresh"]
        token = RefreshToken(refresh_token)
        token.blacklist() # Requires 'rest_framework_simplejwt.token_blacklist' app
        return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)
    except KeyError:
        return Response({"error": "Refresh token is required in body."}, status=status.HTTP_400_BAD_REQUEST)
    except TokenError:
        return Response({"error": "Token is invalid or expired."}, status=status.HTTP_400_BAD_REQUEST)