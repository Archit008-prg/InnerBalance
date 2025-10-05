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
    """Get all assessment questions"""
    try:
        questions = Question.objects.all().order_by('order')
        question_data = []
        
        for question in questions:
            question_data.append({
                'id': question.id,
                'text': question.text,
                'type': question.question_type,
                'category': question.category,
                'order': question.order
            })
        
        print(f"✅ Found {len(question_data)} questions in database")  # Debug print
        
        return Response({
            'count': len(question_data),
            'questions': question_data
        })
        
    except Exception as e:
        # Print detailed error for debugging
        print(f"❌ Error in get_questions: {str(e)}")
        traceback.print_exc()
        
        # Return empty questions with error info
        return Response({
            'error': str(e),
            'count': 0,
            'questions': []
        }, status=500)




