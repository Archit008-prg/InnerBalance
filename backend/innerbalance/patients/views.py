import json
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from patients.models import Patient, Doctor
from patients.decorators import doctor_only, patient_only
from questionnaires.models import Assessment, Answer, AssessmentReport


# ---------------------------------------------------------------------------
# Doctor Dashboard  — GET /api/doctor/dashboard/
# Only users with a Doctor profile can access this. 403 for everyone else.
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@doctor_only
def doctor_dashboard(request):
    """
    Return all patient assessments + clinical reports for the logged-in doctor.
    Access: doctor role only. Returns 403 for patients or unauthenticated users.
    """
    doctor = request.user.doctor
    assessments = Assessment.objects.all().order_by('-started_at')
    dashboard_data = []

    for assessment in assessments:
        patient_user = assessment.patient.user
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
                        'recommendations': report.recommendations,
                    }
            else:
                report_data = {
                    'summary': report.summary,
                    'risk_factors': report.risk_factors,
                    'recommendations': report.recommendations,
                }
        except AssessmentReport.DoesNotExist:
            pass

        answers_list = []
        for ans in assessment.answers.all().order_by('question__is_follow_up', 'question__order', 'id'):
            answers_list.append({
                'question_id': ans.question.id,
                'question_text': ans.question.text,
                'response': ans.response,
                'score': ans.score,
                'is_follow_up': ans.question.is_follow_up,
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
                'overall': assessment.overall_score,
            },
            'patient': {
                'id': assessment.patient.patient_id,
                'username': patient_user.username,
                'name': f"{patient_user.first_name} {patient_user.last_name}",
                'email': patient_user.email,
                'gender': assessment.patient.gender,
                'phone': assessment.patient.phone_number,
                'address': assessment.patient.address,
            },
            'report': report_data,
            'answers': answers_list,
        })

    return Response({
        'doctor': {
            'id': doctor.doctor_id,
            'name': f"Dr. {request.user.first_name} {request.user.last_name}",
            'specialization': doctor.specialization,
            'is_verified': doctor.is_verified,
        },
        'count': len(dashboard_data),
        'assessments': dashboard_data,
    }, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Patient Dashboard  — GET /api/patient/dashboard/
# Only users with a Patient profile can access this. 403 for doctors.
# ---------------------------------------------------------------------------

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@patient_only
def patient_dashboard(request):
    """
    Return all assessments and reports for the logged-in patient.
    Access: patient role only. Returns 403 for doctors or unauthenticated users.
    """
    patient = request.user.patient
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
                        'recommendations': report.recommendations,
                    }
            else:
                report_data = {
                    'summary': report.summary,
                    'risk_factors': report.risk_factors,
                    'recommendations': report.recommendations,
                }
        except AssessmentReport.DoesNotExist:
            pass

        answers_list = []
        for ans in assessment.answers.all().order_by('question__is_follow_up', 'question__order', 'id'):
            answers_list.append({
                'question_id': ans.question.id,
                'question_text': ans.question.text,
                'response': ans.response,
                'score': ans.score,
                'is_follow_up': ans.question.is_follow_up,
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
                'overall': assessment.overall_score,
            },
            'report': report_data,
            'answers': answers_list,
        })

    return Response({
        'patient': {
            'id': patient.patient_id,
            'name': f"{request.user.first_name} {request.user.last_name}",
        },
        'count': len(dashboard_data),
        'assessments': dashboard_data,
    }, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Delete Assessment  — DELETE /api/doctor/assessments/<id>/delete/
# Doctor only.
# ---------------------------------------------------------------------------

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
@doctor_only
def delete_assessment(request, assessment_id):
    """
    Permanently delete a patient assessment record.
    Access: doctor role only.
    """
    try:
        assessment = Assessment.objects.get(id=assessment_id)
        assessment.delete()
        return Response(
            {'message': f'Assessment record #{assessment_id} was successfully deleted.'},
            status=status.HTTP_200_OK,
        )
    except Assessment.DoesNotExist:
        return Response(
            {'error': f'Assessment record #{assessment_id} not found.'},
            status=status.HTTP_404_NOT_FOUND,
        )
