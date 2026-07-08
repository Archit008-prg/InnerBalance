from django.urls import path, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import AuthStatusView, register_user, get_user_profile

urlpatterns = [
    path('', AuthStatusView.as_view(), name='auth-root'),  # Handles /auth/
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),  # Handles /auth/token/
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),  # Handles /auth/token/refresh/
    path('register/', register_user, name='register_user'),  # Handles /auth/register/
    path('me/', get_user_profile, name='get_user_profile'),  # Handles /auth/me/
]