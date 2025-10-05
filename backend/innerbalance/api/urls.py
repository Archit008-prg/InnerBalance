from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from . import views, rag_views

router = DefaultRouter()
# We'll add viewset routes here later

urlpatterns = [
    path('', include(router.urls)),
    path('test/', views.test_api, name='test_api'),
    path('questions/', views.get_questions, name='get_questions'),
    
    # New RAG endpoints
    path('analyze-initial/', rag_views.analyze_initial_assessment, name='analyze_initial'),
    path('generate-report/', rag_views.generate_clinical_report, name='generate_report'),
    path('system-status/', rag_views.system_status, name='system_status'),
]




