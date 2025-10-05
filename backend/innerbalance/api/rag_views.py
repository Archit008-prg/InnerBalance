from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
import json
import logging

from rag.meditron_rag import MeditronRAGSystem

# Initialize RAG system
rag_system = MeditronRAGSystem()

logger = logging.getLogger(__name__)

@api_view(['POST'])
def analyze_initial_assessment(request):
    """
    Analyze first 10 answers and generate personalized follow-up questions
    """
    try:
        data = request.data
        assessment_id = data.get('assessment_id')
        answers = data.get('answers', {})
        
        # Convert string keys to integers if needed
        processed_answers = {}
        for key, value in answers.items():
            try:
                processed_answers[int(key)] = int(value)
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Analyzing assessment {assessment_id} with answers: {processed_answers}")
        
        # Analyze answers
        analysis = rag_system.analyze_initial_answers(processed_answers)
        
        # Generate follow-up questions
        follow_up_questions = rag_system.generate_follow_up_questions(analysis)
        
        response_data = {
            'assessment_id': assessment_id,
            'analysis': analysis,
            'follow_up_questions': follow_up_questions,
            'risk_level': analysis['suicide_risk'],
            'status': 'success'
        }
        
        # Log risk level for monitoring
        if analysis['suicide_risk'] == 'high':
            logger.warning(f"HIGH RISK detected in assessment {assessment_id}")
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error in analyze_initial_assessment: {str(e)}")
        return Response({
            'error': 'Failed to analyze assessment',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST']) 
def generate_clinical_report(request):
    """
    Generate comprehensive clinical report from all answers
    """
    try:
        data = request.data
        assessment_id = data.get('assessment_id')
        initial_answers = data.get('initial_answers', {})
        follow_up_responses = data.get('follow_up_responses', {})
        
        # Process answers
        processed_initial = {}
        for key, value in initial_answers.items():
            try:
                processed_initial[int(key)] = int(value)
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Generating report for assessment {assessment_id}")
        
        # Generate comprehensive report
        report = rag_system.generate_comprehensive_report(
            processed_initial, 
            follow_up_responses
        )
        
        response_data = {
            'assessment_id': assessment_id,
            'report': report,
            'status': 'success',
            'assessment_complete': True
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error in generate_clinical_report: {str(e)}")
        return Response({
            'error': 'Failed to generate clinical report',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def system_status(request):
    """
    Check RAG system status and capabilities
    """
    status_info = {
        'llm_loaded': rag_system.llm is not None,
        'vector_store_ready': True,  # We'll check this
        'knowledge_base_items': rag_system.vector_store._collection.count() if hasattr(rag_system.vector_store, '_collection') else 0,
        'system': 'Meditron-RAG Mental Health Assessment',
        'version': '1.0'
    }
    
    return Response(status_info, status=status.HTTP_200_OK)