from django.shortcuts import render

# Create your views here.
# In your views.py
from django.http import JsonResponse
from django.views import View

class AuthStatusView(View):
    def get(self, request):
        return JsonResponse({
            'message': 'Authentication API is running',
            'endpoints': {
                'login': '/auth/token/',
                'refresh_token': '/auth/token/refresh/',
            }
        })