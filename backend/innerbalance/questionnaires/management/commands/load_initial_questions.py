from django.core.management.base import BaseCommand
from questionnaires.models import Question

class Command(BaseCommand):
    help = 'Load final 10 refined mental health assessment questions'

    def handle(self, *args, **options):
        questions = [
        # Core Depression Symptoms (DSM-5 based)
            {
                'text': 'How often have you felt little interest or pleasure in doing things over the past two weeks?',
                'type': 'scale',
                'category': 'depression',
                'order': 1
            },
            {
                'text': 'How often have you felt down, depressed, or hopeless during the last two weeks?',
                'type': 'scale', 
                'category': 'depression',
                'order': 2
            },
            {
                'text': 'How often have you had trouble falling asleep, staying asleep, or sleeping too much?',
                'type': 'scale',
                'category': 'sleep',
                'order': 3
            },
            {
                'text': 'How often have you felt tired or had little energy, even after rest?',
                'type': 'scale',
                'category': 'energy',
                'order': 4
            },
            {
                'text': 'How often have you experienced poor appetite or overeating in the past two weeks?',
                'type': 'scale',
                'category': 'appetite', 
                'order': 5
            },
            
            # Anxiety & Mood
            {
                'text': 'How often have you felt nervous, anxious, or on edge?',
                'type': 'scale',
                'category': 'anxiety',
                'order': 6
            },
            {
                'text': 'How often have you been unable to stop or control worrying?',
                'type': 'scale',
                'category': 'anxiety',
                'order': 7
            },
            {
                'text': 'How often have you felt easily annoyed or irritable?',
                'type': 'scale',
                'category': 'mood',
                'order': 8
            },
            
            # Critical Safety Questions
            {
                'text': 'How often have you had thoughts that you would be better off dead or hurting yourself?',
                'type': 'scale', 
                'category': 'suicidality',
                'order': 9,
                'risk_trigger': True  # Flag for crisis protocol
            },
            
            # Social & Functional Impact
            {
                'text': 'How often have these difficulties affected your work, social life, or home responsibilities?',
                'type': 'scale',
                'category': 'functioning',
                'order': 10
            }
        ]

        for q_data in questions:
            question, created = Question.objects.get_or_create(
                text=q_data['text'],
                defaults={
                    'question_type': q_data['type'],
                    'category': q_data['category'],
                    'order': q_data['order']
                }
            )
            if created:
                self.stdout.write(f"Created: {q_data['text'][:50]}...")
            else:
                self.stdout.write(f"Already exists: {q_data['text'][:50]}...")

        self.stdout.write(
            self.style.SUCCESS('Successfully loaded initial questions.')
        )
