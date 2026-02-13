from django.urls import path
from .views import RegisterView, LoginView, logout_view, auth_root_view
from rest_framework_simplejwt.views import TokenRefreshView 


urlpatterns = [
    path('', auth_root_view, name='auth-root'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', logout_view, name='logout'),
    path('refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]