from django.core.management.base import BaseCommand
from rag.meditron_rag import MeditronRAGSystem

class Command(BaseCommand):
    help = 'Test the RAG system functionality'
    
    def handle(self, *args, **options):
        self.stdout.write("[TEST] Testing RAG System...")
        
        try:
            # Initialize RAG system
            rag = MeditronRAGSystem()
            self.stdout.write("[TEST] Rebuilding Vector Store to capture new documents...")
            rag.rebuild_vector_store()
            self.stdout.write(self.style.SUCCESS("[OK] RAG System initialized and vector store rebuilt"))
            
            # Test analysis
            self.stdout.write("\n1. Testing Answer Analysis...")
            # Anxiety test case
            sample_answers = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0}
            
            analysis = rag.analyze_initial_answers(sample_answers)
            self.stdout.write(f"[OK] Analysis: {analysis['primary_concerns']}")
            
            # Test question generation
            self.stdout.write("\n2. Testing Question Generation...")
            questions = rag.generate_follow_up_questions(analysis)
            self.stdout.write(f"[OK] Generated {len(questions)} questions")
            for i, q in enumerate(questions, 1):
                self.stdout.write(f"   {i}. {q}")
                
            self.stdout.write(self.style.SUCCESS("\n[SUCCESS] RAG System Working!"))

            # Safe print helper to prevent CP1252 charmap encoding errors on Windows terminal
            def safe_print(msg):
                print(str(msg).encode('ascii', 'ignore').decode('ascii'))

            # Add this after the question generation test
            safe_print("\n3. Testing Clinical Report Generation...")
            follow_up_responses = {
                "How long have you been experiencing these low mood symptoms?": "About 2 months",
                "What activities or interactions still bring you some sense of pleasure or accomplishment?": "Sometimes watching movies helps",
                "How would you describe your overall quality of life right now?": "Pretty low, hard to enjoy things",
                "What aspects of your life are going well despite these challenges?": "My job is stable",
                "What kind of support would be most helpful to you right now?": "Someone to talk to regularly"
            }

            report = rag.generate_comprehensive_report(sample_answers, follow_up_responses)
            safe_print("[OK] Clinical Report Generated:")
            safe_print(f"   - Risk Level: {report['risk_level']}")
            safe_print(f"   - Primary Concerns: {report.get('diagnostic_considerations', [])}")
            safe_print(f"   - Severity: {report.get('symptom_severity', {})}")
            safe_print(f"   - Key Recommendations: {report.get('recommendations', [])[:2]}")
            
            # Test case for off-topic/invalid responses
            safe_print("\n4. Testing Off-Topic / Low-effort Response Validation...")
            off_topic_responses = {
                "How long have you been experiencing these low mood symptoms?": "lalu yadav",
                "What activities or interactions still bring you some sense of pleasure or accomplishment?": "paav bhaji",
                "How would you describe your overall quality of life right now?": "nothing",
                "What aspects of your life are going well despite these challenges?": "ok",
                "What kind of support would be most helpful to you right now?": "blah"
            }
            off_topic_report = rag.generate_comprehensive_report(sample_answers, off_topic_responses)
            safe_print("[OK] Off-Topic Report Generated:")
            safe_print(f"   - Risk Level: {off_topic_report['risk_level']}")
            safe_print(f"   - Primary Considerations: {off_topic_report.get('diagnostic_considerations', [])}")
            safe_print(f"   - Severity: {off_topic_report.get('symptom_severity', {})}")
            safe_print(f"   - Insights: {off_topic_report.get('clinical_insights', [])}")
            safe_print(f"   - Key Recommendations: {off_topic_report.get('recommendations', [])[:2]}")
            safe_print(f"   - Low effort detected? {off_topic_report.get('validation', {}).get('low_effort')}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"[FAIL] RAG System Failed: {e}"))