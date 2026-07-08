from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db import transaction
import json
import logging
import traceback

from rag.meditron_rag import MeditronRAGSystem
from patients.models import Patient
from questionnaires.models import Assessment, Answer, AssessmentReport, Question

# Initialize RAG system
rag_system = MeditronRAGSystem()

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_initial_assessment(request):
    """
    Analyze first 10 answers and generate personalized follow-up questions
    """
    try:
        # Get patient profile for current user
        try:
            patient = request.user.patient
        except Patient.DoesNotExist:
            return Response({
                'error': 'Forbidden. Only registered patients can submit assessments.'
            }, status=status.HTTP_403_FORBIDDEN)
            
        data = request.data
        assessment_id = data.get('assessment_id')
        answers = data.get('answers', {})
        
        # Map incoming question ID keys to question orders from database
        questions_map = {q.id: q.order for q in Question.objects.filter(is_follow_up=False)}
        processed_answers = {}
        for key, value in answers.items():
            try:
                q_id = int(key)
                order = questions_map.get(q_id, q_id)
                val_float = float(value)
                if val_float.is_integer() and order not in [16, 17]:
                    processed_answers[order] = int(val_float)
                else:
                    processed_answers[order] = val_float
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
@permission_classes([IsAuthenticated])
def generate_clinical_report(request):
    """
    Generate comprehensive clinical report from all answers and persist it in PostgreSQL
    """
    try:
        user = request.user
        
        # Ensure user is a patient
        try:
            patient = user.patient
        except Patient.DoesNotExist:
            return Response({
                'error': 'Forbidden. Only registered patients can submit assessments.'
            }, status=status.HTTP_403_FORBIDDEN)
            
        data = request.data
        initial_answers = data.get('initial_answers', {})
        follow_up_responses = data.get('follow_up_responses', {})
        
        # Map incoming question ID keys to question orders (1 to 10) from database
        questions_map = {q.id: q.order for q in Question.objects.filter(is_follow_up=False)}
        processed_initial = {}
        for key, value in initial_answers.items():
            try:
                q_id = int(key)
                order = questions_map.get(q_id, q_id)
                processed_initial[order] = int(value)
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Generating and saving report for patient {patient.patient_id}")
        
        # Generate comprehensive report
        report = rag_system.generate_comprehensive_report(
            processed_initial, 
            follow_up_responses
        )
        
        # Calculate scores
        depression_score = sum(int(processed_initial.get(o, 0)) for o in [1, 2, 3, 4, 5, 11, 12, 13])
        anxiety_score = sum(int(processed_initial.get(o, 0)) for o in [6, 7, 8, 9, 10, 14, 15])
        overall_score = depression_score + anxiety_score
        
        # Map risk level choices
        risk_map = {
            'low': 'low',
            'moderate': 'moderate',
            'high': 'high',
            'crisis': 'crisis',
            'unreliable': 'low'
        }
        db_risk_level = risk_map.get(report.get('risk_level', 'low'), 'low')
        
        with transaction.atomic():
            # Create a new Assessment record for the patient
            assessment = Assessment.objects.create(
                patient=patient,
                status='completed' if report.get('risk_level') != 'crisis' else 'crisis',
                risk_level=db_risk_level,
                depression_score=depression_score,
                anxiety_score=anxiety_score,
                overall_score=overall_score,
                completed_at=timezone.now()
            )
            
            # Save MCQ answers (initial screening)
            for q_order, score_val in processed_initial.items():
                try:
                    question = Question.objects.get(order=q_order, is_follow_up=False)
                except Question.DoesNotExist:
                    category = 'depression' if q_order < 9 else 'anxiety'
                    question = Question.objects.create(
                        text=f"Initial Assessment Question #{q_order}",
                        question_type='scale',
                        category=category,
                        order=q_order,
                        is_follow_up=False
                    )
                Answer.objects.update_or_create(
                    assessment=assessment,
                    question=question,
                    defaults={'response': str(score_val), 'score': score_val}
                )
                
            # Save text answers (follow-up questions)
            for q_text, a_text in follow_up_responses.items():
                question, created = Question.objects.get_or_create(
                    text=q_text,
                    defaults={
                        'question_type': 'text',
                        'category': 'general',
                        'is_follow_up': True
                    }
                )
                Answer.objects.update_or_create(
                    assessment=assessment,
                    question=question,
                    defaults={'response': a_text}
                )
                
            # Create the AssessmentReport
            summary_str = "\n".join(report.get('clinical_insights', []))
            risk_factors_str = "\n".join(report.get('diagnostic_considerations', []))
            recommendations_str = "\n".join(report.get('recommendations', []))
            
            AssessmentReport.objects.create(
                assessment=assessment,
                summary=summary_str,
                risk_factors=risk_factors_str,
                recommendations=recommendations_str,
                full_report_json=json.dumps(report)
            )
            
        response_data = {
            'assessment_id': assessment.id,
            'report': report,
            'status': 'success',
            'assessment_complete': True
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error in generate_clinical_report: {str(e)}")
        traceback.print_exc()
        return Response({
            'error': 'Failed to generate and save clinical report',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def system_status(request):
    """
    Check RAG system status and capabilities
    """
    status_info = {
        'llm_loaded': rag_system.llm is not None,
        'vector_store_ready': True,
        'knowledge_base_items': rag_system.vector_store._collection.count() if hasattr(rag_system.vector_store, '_collection') else 0,
        'system': 'Meditron-RAG Mental Health Assessment',
        'version': '1.0'
    }
    
    return Response(status_info, status=status.HTTP_200_OK)