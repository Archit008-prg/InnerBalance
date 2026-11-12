from functools import wraps
from rest_framework.response import Response
from rest_framework import status


def doctor_only(view_func):
    """Allow only authenticated users with a Doctor profile."""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return Response(
                {'error': 'Authentication required.'},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        if not hasattr(request.user, 'doctor'):
            return Response(
                {'error': 'Forbidden. Only registered clinicians/doctors can access this endpoint.'},
                status=status.HTTP_403_FORBIDDEN,
            )
        return view_func(request, *args, **kwargs)
    return _wrapped


def patient_only(view_func):
    """Allow only authenticated users with a Patient profile."""
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return Response(
                {'error': 'Authentication required.'},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        if not hasattr(request.user, 'patient'):
            return Response(
                {'error': 'Forbidden. Only registered patients can access this endpoint.'},
                status=status.HTTP_403_FORBIDDEN,
            )
        return view_func(request, *args, **kwargs)
    return _wrapped
