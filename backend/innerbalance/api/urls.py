from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views, rag_views
from patients import views as patient_views

router = DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('test/', views.test_api, name='test_api'),
    path('questions/', views.get_questions, name='get_questions'),
    
    # New RAG endpoints
    path('analyze-initial/', rag_views.analyze_initial_assessment, name='analyze_initial'),
    path('generate-report/', rag_views.generate_clinical_report, name='generate_report'),
    path('system-status/', rag_views.system_status, name='system_status'),
    
    # Dashboard endpoints
    path('doctor/dashboard/', patient_views.doctor_dashboard, name='doctor_dashboard'),
    path('patient/dashboard/', patient_views.patient_dashboard, name='patient_dashboard'),
    path('doctor/assessments/<int:assessment_id>/delete/', patient_views.delete_assessment, name='delete_assessment'),
]




