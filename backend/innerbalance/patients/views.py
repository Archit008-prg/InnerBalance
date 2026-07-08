from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
import json

from patients.models import Patient, Doctor
from questionnaires.models import Assessment, Answer, AssessmentReport

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def doctor_dashboard(request):
    """
    List all patient assessments and reports for the logged-in doctor
    """
    user = request.user
    
    # Verify user is a doctor
    try:
        doctor = user.doctor
    except Doctor.DoesNotExist:
        return Response({
            'error': 'Forbidden. Only registered clinicians/doctors can access this dashboard.'
        }, status=status.HTTP_403_FORBIDDEN)
        
    # Get all completed/processed assessments in reverse chronological order
    assessments = Assessment.objects.all().order_by('-started_at')
    dashboard_data = []
    
    for assessment in assessments:
        patient_user = assessment.patient.user
        report_data = None
        
        # Try to find associated clinical report
        try:
            report = assessment.assessmentreport
            if report.full_report_json:
                try:
                    report_data = json.loads(report.full_report_json)
                except Exception:
                    report_data = {
                        'summary': report.summary,
                        'risk_factors': report.risk_factors,
                        'recommendations': report.recommendations
                    }
            else:
                report_data = {
                    'summary': report.summary,
                    'risk_factors': report.risk_factors,
                    'recommendations': report.recommendations
                }
        except AssessmentReport.DoesNotExist:
            pass
            
        # Fetch raw patient answers
        answers_list = []
        for ans in assessment.answers.all().order_by('question__is_follow_up', 'question__order', 'id'):
            answers_list.append({
                'question_id': ans.question.id,
                'question_text': ans.question.text,
                'response': ans.response,
                'score': ans.score,
                'is_follow_up': ans.question.is_follow_up
            })

        dashboard_data.append({
            'assessment_id': assessment.id,
            'status': assessment.status,
            'risk_level': assessment.risk_level,
            'started_at': assessment.started_at,
            'completed_at': assessment.completed_at,
            'scores': {
                'depression': assessment.depression_score,
                'anxiety': assessment.anxiety_score,
                'overall': assessment.overall_score
            },
            'patient': {
                'id': assessment.patient.patient_id,
                'username': patient_user.username,
                'name': f"{patient_user.first_name} {patient_user.last_name}",
                'email': patient_user.email,
                'gender': assessment.patient.gender,
                'phone': assessment.patient.phone_number,
                'address': assessment.patient.address
            },
            'report': report_data,
            'answers': answers_list
        })
        
    return Response({
        'doctor': {
            'id': doctor.doctor_id,
            'name': f"Dr. {user.first_name} {user.last_name}",
            'specialization': doctor.specialization
        },
        'count': len(dashboard_data),
        'assessments': dashboard_data
    }, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def patient_dashboard(request):
    """
    List all assessments and reports for the logged-in patient
    """
    user = request.user
    
    # Verify user is a patient
    try:
        patient = user.patient
    except Patient.DoesNotExist:
        return Response({
            'error': 'Forbidden. Only registered patients can access this dashboard.'
        }, status=status.HTTP_403_FORBIDDEN)
        
    # Get all assessments for this patient
    assessments = Assessment.objects.filter(patient=patient).order_by('-started_at')
    dashboard_data = []
    
    for assessment in assessments:
        report_data = None
        try:
            report = assessment.assessmentreport
            if report.full_report_json:
                try:
                    report_data = json.loads(report.full_report_json)
                except Exception:
                    report_data = {
                        'summary': report.summary,
                        'risk_factors': report.risk_factors,
                        'recommendations': report.recommendations
                    }
            else:
                report_data = {
                    'summary': report.summary,
                    'risk_factors': report.risk_factors,
                    'recommendations': report.recommendations
                }
        except AssessmentReport.DoesNotExist:
            pass
            
        # Fetch raw patient answers
        answers_list = []
        for ans in assessment.answers.all().order_by('question__is_follow_up', 'question__order', 'id'):
            answers_list.append({
                'question_id': ans.question.id,
                'question_text': ans.question.text,
                'response': ans.response,
                'score': ans.score,
                'is_follow_up': ans.question.is_follow_up
            })

        dashboard_data.append({
            'assessment_id': assessment.id,
            'status': assessment.status,
            'risk_level': assessment.risk_level,
            'started_at': assessment.started_at,
            'completed_at': assessment.completed_at,
            'scores': {
                'depression': assessment.depression_score,
                'anxiety': assessment.anxiety_score,
                'overall': assessment.overall_score
            },
            'report': report_data,
            'answers': answers_list
        })
        
    return Response({
        'patient': {
            'id': patient.patient_id,
            'name': f"{user.first_name} {user.last_name}"
        },
        'count': len(dashboard_data),
        'assessments': dashboard_data
    }, status=status.HTTP_200_OK)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_assessment(request, assessment_id):
    """
    Delete a patient assessment record (Clinician only)
    """
    user = request.user
    
    # Verify user is a doctor
    try:
        doctor = user.doctor
    except Doctor.DoesNotExist:
        return Response({
            'error': 'Forbidden. Only registered clinicians/doctors can delete records.'
        }, status=status.HTTP_403_FORBIDDEN)
        
    try:
        assessment = Assessment.objects.get(id=assessment_id)
        assessment.delete()
        return Response({
            'message': f'Assessment record #{assessment_id} was successfully deleted.'
        }, status=status.HTTP_200_OK)
    except Assessment.DoesNotExist:
        return Response({
            'error': f'Assessment record #{assessment_id} not found.'
        }, status=status.HTTP_404_NOT_FOUND)
