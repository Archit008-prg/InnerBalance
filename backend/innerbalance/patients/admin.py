from django.contrib import admin

# Register your models here.

from .models import Patient, Doctor

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'user', 'date_of_birth', 'gender', 'created_at']
    list_filter = ['gender', 'created_at']
    search_fields = ['user__first_name', 'user__last_name', 'patient_id']

@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ['doctor_id', 'user', 'specialization', 'license_number']
    search_fields = ['user__first_name', 'user__last_name', 'specialization']