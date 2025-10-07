from django.core.management.base import BaseCommand
from rag.meditron_rag import MeditronRAGSystem

class Command(BaseCommand):
    help = 'Test the RAG system functionality'
    
    def handle(self, *args, **options):
        self.stdout.write("🧪 Testing RAG System...")
        
        try:
            # Initialize RAG system
            rag = MeditronRAGSystem()
            self.stdout.write(self.style.SUCCESS("✅ RAG System initialized"))
            
            # Test analysis
            self.stdout.write("\n1. Testing Answer Analysis...")
            # Anxiety test case
            sample_answers = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0}
            
            analysis = rag.analyze_initial_answers(sample_answers)
            self.stdout.write(f"✅ Analysis: {analysis['primary_concerns']}")
            
            # Test question generation
            self.stdout.write("\n2. Testing Question Generation...")
            questions = rag.generate_follow_up_questions(analysis)
            self.stdout.write(f"✅ Generated {len(questions)} questions")
            for i, q in enumerate(questions, 1):
                self.stdout.write(f"   {i}. {q}")
                
            self.stdout.write(self.style.SUCCESS("\n🎉 RAG System Working!"))

            # Add this after the question generation test
            print("\n3. Testing Clinical Report Generation...")
            follow_up_responses = {
                "How long have you been experiencing these low mood symptoms?": "About 2 months",
                "What activities or interactions still bring you some sense of pleasure or accomplishment?": "Sometimes watching movies helps",
                "How would you describe your overall quality of life right now?": "Pretty low, hard to enjoy things",
                "What aspects of your life are going well despite these challenges?": "My job is stable",
                "What kind of support would be most helpful to you right now?": "Someone to talk to regularly"
            }

            report = rag.generate_comprehensive_report(sample_answers, follow_up_responses)
            print("✅ Clinical Report Generated:")
            print(f"   - Risk Level: {report['risk_level']}")
            print(f"   - Primary Concerns: {report.get('diagnostic_considerations', [])}")
            print(f"   - Severity: {report.get('symptom_severity', {})}")
            print(f"   - Key Recommendations: {report.get('recommendations', [])[:2]}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ RAG System Failed: {e}"))