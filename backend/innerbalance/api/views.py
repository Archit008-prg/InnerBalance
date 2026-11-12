from rest_framework.decorators import api_view
from rest_framework.response import Response
from questionnaires.models import Question
import traceback 

@api_view(['GET'])
def test_api(request):
    return Response({
        'message': 'InnerBalance API is working!',
        'status': 'success',
        'endpoints': {
            'admin': '/admin/',
            'api_test': '/api/test/',
            'authentication': '/auth/'
        }
    })

# ADD THIS FUNCTION - it's referenced in urls.py but missing
@api_view(['GET'])
def get_questions(request):
    """Get the first 10 fixed initial assessment questions for standardized screening"""
    try:
        # Exactly the first 10 fixed standardized questions (PHQ/GAD core items)
        questions = Question.objects.filter(is_follow_up=False).order_by('order')[:10]
        question_data = []
        
        for question in questions:
            question_data.append({
                'id': question.id,
                'text': question.text,
                'type': question.question_type,
                'category': question.category,
                'order': question.order
            })
        
        print(f"[SUCCESS] Found {len(question_data)} standardized questions for initial assessment")
        
        return Response({
            'count': len(question_data),
            'questions': question_data
        })
        
    except Exception as e:
        print(f"[ERROR] Error in get_questions: {str(e)}")
        traceback.print_exc()
        
        return Response({
            'error': str(e),
            'count': 0,
            'questions': []
        }, status=500)




