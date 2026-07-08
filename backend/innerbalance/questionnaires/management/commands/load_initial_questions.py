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
                'text': 'How often have you felt so restless that it is hard to sit still?',
                'type': 'scale',
                'category': 'anxiety',
                'order': 8
            },
            {
                'text': 'How often have you become easily annoyed or irritable?',
                'type': 'scale', 
                'category': 'anxiety',
                'order': 9
            },
            {
                'text': 'How often have you felt afraid as if something awful might happen?',
                'type': 'scale',
                'category': 'anxiety',
                'order': 10
            },

            # Additional clinical scale items (DSM-5 / GAD / PHQ)
            {
                'text': 'How often have you felt bad about yourself - or that you are a failure or have let yourself or your family down?',
                'type': 'scale',
                'category': 'depression',
                'order': 11
            },
            {
                'text': 'How often have you had trouble concentrating on things, such as reading the newspaper or watching television?',
                'type': 'scale',
                'category': 'depression',
                'order': 12
            },
            {
                'text': 'How often have you moved or spoken so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?',
                'type': 'scale',
                'category': 'depression',
                'order': 13
            },
            {
                'text': 'How often have you worried too much about different things?',
                'type': 'scale',
                'category': 'anxiety',
                'order': 14
            },
            {
                'text': 'How often have you had trouble relaxing?',
                'type': 'scale',
                'category': 'anxiety',
                'order': 15
            },

            # Distress Intensity Sliders (1.0 to 5.0 with mid-levels like PUSH-D)
            {
                'text': 'On a scale of 1.0 to 5.0, how intense has your anxiety or worry been when at its worst over the past two weeks?',
                'type': 'slider',
                'category': 'anxiety',
                'order': 16
            },
            {
                'text': 'On a scale of 1.0 to 5.0, how intense has your low mood or lack of interest been when at its worst over the past two weeks?',
                'type': 'slider',
                'category': 'depression',
                'order': 17
            },

            # Daily Functioning Scale
            {
                'text': 'If you checked off any problems, how difficult have these problems made it for you to do your work, take care of things at home, or get along with other people?',
                'type': 'scale',
                'category': 'functioning',
                'order': 18
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
