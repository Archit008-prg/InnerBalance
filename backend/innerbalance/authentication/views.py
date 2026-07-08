from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from django.db import transaction
from patients.models import Patient, Doctor

class AuthStatusView(View):
    def get(self, request):
        return JsonResponse({
            'message': 'Authentication API is running',
            'endpoints': {
                'login': '/auth/token/',
                'refresh_token': '/auth/token/refresh/',
                'register': '/auth/register/',
                'me': '/auth/me/'
            }
        })

@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    """
    Register a new user (Patient or Doctor)
    """
    try:
        data = request.data
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        first_name = data.get('first_name', '')
        last_name = data.get('last_name', '')
        role = data.get('role')  # 'patient' or 'doctor'
        
        if not username or not password or not email or not role:
            return Response({
                'error': 'Missing required fields: username, password, email, and role are required.'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        if role not in ['patient', 'doctor']:
            return Response({
                'error': 'Invalid role. Must be either "patient" or "doctor".'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        if User.objects.filter(username=username).exists():
            return Response({
                'error': 'Username already exists.'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        if User.objects.filter(email=email).exists():
            return Response({
                'error': 'Email already registered.'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        with transaction.atomic():
            # Create user
            user = User.objects.create_user(
                username=username,
                password=password,
                email=email,
                first_name=first_name,
                last_name=last_name
            )
            
            # Create corresponding role profile
            if role == 'patient':
                patient = Patient.objects.create(
                    user=user,
                    gender=data.get('gender', ''),
                    phone_number=data.get('phone_number', ''),
                    address=data.get('address', ''),
                    city=data.get('city', ''),
                    education_or_occupation=data.get('education_or_occupation', ''),
                    age=int(data.get('age')) if data.get('age') else None,
                    marital_status=data.get('marital_status', '')
                )
                profile_id = patient.patient_id
            else:  # doctor
                doctor = Doctor.objects.create(
                    user=user,
                    specialization=data.get('specialization', 'General Psychiatry'),
                    license_number=data.get('license_number', 'N/A'),
                    years_of_experience=int(data.get('years_of_experience', 0)),
                    hospital_affiliation=data.get('hospital_affiliation', '')
                )
                profile_id = doctor.doctor_id
                
        return Response({
            'message': f'User registered successfully as a {role}.',
            'username': username,
            'role': role,
            'profile_id': profile_id
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response({
            'error': 'Registration failed',
            'details': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
 
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_profile(request):
    """
    Get the logged-in user's profile and role details
    """
    user = request.user
    role = None
    profile_id = None
    details = {}
    
    # Check if user is a Patient
    try:
        patient = user.patient
        role = 'patient'
        profile_id = patient.patient_id
        details = {
            'phone_number': patient.phone_number,
            'gender': patient.gender,
            'address': patient.address,
            'city': patient.city,
            'education_or_occupation': patient.education_or_occupation,
            'age': patient.age,
            'marital_status': patient.marital_status
        }
    except Patient.DoesNotExist:
        pass
        
    # Check if user is a Doctor
    if not role:
        try:
            doctor = user.doctor
            role = 'doctor'
            profile_id = doctor.doctor_id
            details = {
                'specialization': doctor.specialization,
                'license_number': doctor.license_number,
                'hospital_affiliation': doctor.hospital_affiliation,
                'years_of_experience': doctor.years_of_experience
            }
        except Doctor.DoesNotExist:
            pass
            
    # Default fallback if user has no specific profile
    if not role:
        role = 'admin' if user.is_staff else 'user'
        profile_id = 'N/A'
        
    return Response({
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'role': role,
        'profile_id': profile_id,
        'details': details
    }, status=status.HTTP_200_OK)