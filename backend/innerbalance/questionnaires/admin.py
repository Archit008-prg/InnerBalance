from django.contrib import admin

# Register your models here.

from .models import Question, Assessment, Answer, AssessmentReport

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ['text', 'question_type', 'category', 'order', 'is_follow_up']
    list_filter = ['question_type', 'category', 'is_follow_up']
    ordering = ['order']

@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = ['id', 'patient', 'status', 'risk_level', 'started_at']
    list_filter = ['status', 'risk_level', 'started_at']
    readonly_fields = ['started_at']

@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    list_display = ['assessment', 'question', 'response', 'score', 'answered_at']
    list_filter = ['answered_at']

@admin.register(AssessmentReport)
class AssessmentReportAdmin(admin.ModelAdmin):
    list_display = ['assessment', 'generated_at']