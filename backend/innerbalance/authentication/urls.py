from django.urls import path, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import AuthStatusView

# urlpatterns = [
#     path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
#     path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
# ]




urlpatterns = [
    path('', AuthStatusView.as_view(), name='auth-root'),  # Handles /auth/
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),  # Handles /auth/token/
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),  # Handles /auth/token/refresh/
]