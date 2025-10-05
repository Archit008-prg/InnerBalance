from django.db import models

# Create your models here.

from patients.models import Patient

class Question(models.Model):
    QUESTION_TYPES = [
        ('scale', 'Rating Scale (0-3)'),
        ('multiple', 'Multiple Choice'),
        ('text', 'Text Response'),
        ('yesno', 'Yes/No'),
    ]
    
    CATEGORY_CHOICES = [
        ('depression', 'Depression'),
        ('anxiety', 'Anxiety'),
        ('sleep', 'Sleep'),
        ('appetite', 'Appetite'),
        ('energy', 'Energy'),
        ('mood', 'Mood'),
        ('social', 'Social Functioning'),
        ('general', 'General Health'),
    ]
    
    text = models.TextField()
    question_type = models.CharField(max_length=20, choices=QUESTION_TYPES)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    order = models.IntegerField(default=0)
    is_follow_up = models.BooleanField(default=False)
    depends_on_question = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True)
    trigger_value = models.CharField(max_length=50, blank=True)  # Value that triggers this question
    
    class Meta:
        ordering = ['order']
    
    def __str__(self):
        return f"Q{self.order}: {self.text[:50]}..."

class Assessment(models.Model):
    STATUS_CHOICES = [
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('crisis', 'Crisis Detected'),
        ('abandoned', 'Abandoned'),
    ]
    
    RISK_LEVELS = [
        ('low', 'Low Risk'),
        ('moderate', 'Moderate Risk'),
        ('high', 'High Risk'),
        ('crisis', 'Crisis - Immediate Attention Needed'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='in_progress')
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS, default='low')
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    current_question_index = models.IntegerField(default=0)
    
    # Assessment scores (will be calculated)
    depression_score = models.IntegerField(default=0)
    anxiety_score = models.IntegerField(default=0)
    overall_score = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Assessment #{self.id} - {self.patient}"

class Answer(models.Model):
    assessment = models.ForeignKey(Assessment, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    response = models.TextField()
    score = models.IntegerField(null=True, blank=True)  # For scaled questions (0-3)
    answered_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['assessment', 'question']
    
    def __str__(self):
        return f"Answer: {self.response[:20]} (Score: {self.score})"

class AssessmentReport(models.Model):
    assessment = models.OneToOneField(Assessment, on_delete=models.CASCADE)
    summary = models.TextField()
    risk_factors = models.TextField()
    recommendations = models.TextField()
    generated_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Report for Assessment #{self.assessment.id}"