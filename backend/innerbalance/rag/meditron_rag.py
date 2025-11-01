import os
import json
import torch
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

class MeditronRAGSystem:
    def __init__(self):
        self.knowledge_base_path = "rag/knowledge_base/"
        self.vector_db_path = "rag/vector_store/chroma_db/"
        
        # Initialize components
        self.setup_embeddings()
        self.setup_vector_store()
        self.setup_meditron_llm()
        self.setup_prompts()
        
    def setup_embeddings(self):
        """Setup sentence embeddings for vector store"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def setup_vector_store(self):
        """Initialize or load vector store"""
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            print("Loaded existing vector store")
        else:
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            self.load_knowledge_base()
            print("Created new vector store")
    
    def setup_meditron_llm(self):
        """Setup a reliable medical LLM - using stable models"""
        try:
            # Primary: Microsoft Phi-3 Mini (stable, good context)
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            
            print(f"🔄 Loading model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Wrap in LangChain
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
            print(f"✅ {model_name} loaded successfully")
            print("🎯 Model features: 4K context, 3.8B parameters, instruction-tuned")
        
        except Exception as e:
            print(f"❌ Failed to load primary model: {e}")
            print("🔄 Trying fallback model...")
            self.setup_fallback_llm()
    
    def setup_fallback_llm(self):
        """Fallback to ultra-stable model"""
        try:
            # Ultra-stable fallback
            model_name = "distilgpt2"
            
            print(f"🔄 Loading fallback model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            print(f"✅ {model_name} loaded successfully (fallback mode)")
            
        except Exception as e:
            print(f"❌ All models failed: {e}")
            print("🔄 Proceeding without LLM - using rule-based system")
            self.llm = None
    
    def setup_prompts(self):
        """Setup specialized prompts for mental health assessment"""
        
        # Enhanced prompt for generating follow-up questions
        self.follow_up_prompt = PromptTemplate(
            input_variables=["analysis", "context"],
            template="""As a clinical psychologist, generate exactly 5 personalized follow-up questions based on this patient's specific symptoms:

PATIENT ASSESSMENT:
{analysis}

CLINICAL GUIDANCE:
{context}

Create questions that are:
1. HIGHLY SPECIFIC to their symptom severity and patterns
2. EMPATHETIC and supportive in tone  
3. OPEN-ENDED to encourage detailed sharing
4. CLINICALLY RELEVANT based on their primary concerns
5. PERSONALIZED to their unique experience

Focus on understanding their specific struggles with {primary_concerns}.

Return ONLY a JSON array of exactly 5 questions. No explanations.

["question1", "question2", "question3", "question4", "question5"]"""
        )

        # Simplified report prompt
        self.report_prompt = PromptTemplate(
            input_variables=["initial_answers", "follow_up_answers", "clinical_context"],
            template="""As a psychiatrist, create a brief clinical report:

ASSESSMENT DATA:
{initial_answers}

FOLLOW-UP RESPONSES:
{follow_up_answers}

GUIDELINES:
{clinical_context}

Generate concise JSON report with:
- risk_level (low/moderate/high)
- primary_concerns (list)
- symptom_severity (object)
- key_recommendations (3-4 items)
- crisis_indicators (list)

JSON only:"""
        )
    
    def load_knowledge_base(self):
        """Load all medical documents into the vector store"""
        documents = []
        
        # Load symptom patterns
        symptoms_path = f"{self.knowledge_base_path}/clinical_guidelines/symptom_patterns.json"
        if os.path.exists(symptoms_path):
            with open(symptoms_path, 'r') as f:
                symptom_data = json.load(f)
                symptom_text = json.dumps(symptom_data, indent=2)
                documents.append(Document(
                    page_content=symptom_text,
                    metadata={"source": "symptom_patterns", "type": "clinical_guidelines"}
                ))
        
        # Load assessment tools
        assessment_files = [
            "assessment_tools/phq9_questions.txt",
            "assessment_tools/gad7_questions.txt", 
            "diagnostic_criteria/depression_dsm5.txt",
            "diagnostic_criteria/anxiety_dsm5.txt"
        ]
        
        for file_path in assessment_files:
            full_path = f"{self.knowledge_base_path}/{file_path}"
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path, "type": "assessment_tool"}
                    ))
        
        # Load clinical guidelines
        guideline_files = [
            "clinical_guidelines/nice_depression.txt",
            "clinical_guidelines/who_mhgap.txt"
        ]
        
        for file_path in guideline_files:
            full_path = f"{self.knowledge_base_path}/{file_path}"
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path, "type": "clinical_guideline"}
                    ))
        
        # Split documents and add to vector store
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist()
            print(f"✅ Loaded {len(split_docs)} document chunks into vector store")
        else:
            print("⚠️ No documents found in knowledge base")
    
    def analyze_initial_answers(self, answers: Dict[int, int]) -> Dict[str, Any]:
        """Analyze first 10 answers using clinical scoring"""
        # PHQ-9 based scoring (questions 0-8)
        depression_score = sum(answers.get(i, 0) for i in range(9))
        
        # GAD-7 based scoring (questions 4-6 for anxiety core items)
        anxiety_score = sum(answers.get(i, 0) for i in [4, 5, 6])
        
        # Sleep disturbance
        sleep_score = answers.get(2, 0)
        
        # Suicide risk (question 8)
        suicide_risk = answers.get(8, 0)
        
        analysis = {
            "depression_severity": self._score_phq9(depression_score),
            "anxiety_severity": self._score_gad7(anxiety_score),
            "sleep_disturbance": "significant" if sleep_score >= 2 else "moderate" if sleep_score == 1 else "minimal",
            "suicide_risk": "high" if suicide_risk >= 2 else "moderate" if suicide_risk == 1 else "low",
            "primary_concerns": [],
            "follow_up_focus": []
        }
        
        # Determine primary concerns
        if depression_score >= 10:
            analysis["primary_concerns"].append("depression")
            analysis["follow_up_focus"].extend(["mood_patterns", "anhedonia", "cognitive_symptoms"])
        
        if anxiety_score >= 8:
            analysis["primary_concerns"].append("anxiety") 
            analysis["follow_up_focus"].extend(["worry_patterns", "physical_symptoms", "avoidance"])
        
        if sleep_score >= 2:
            analysis["primary_concerns"].append("sleep_disturbance")
            analysis["follow_up_focus"].append("sleep_quality")
        
        # If no clear primary concerns, focus on general wellbeing
        if not analysis["primary_concerns"]:
            analysis["primary_concerns"].append("general_wellbeing")
            analysis["follow_up_focus"].extend(["coping_strategies", "support_systems", "life_impact"])
        
        return analysis
    
    def _score_phq9(self, score: int) -> str:
        if score >= 20: return "severe"
        elif score >= 15: return "moderately_severe" 
        elif score >= 10: return "moderate"
        elif score >= 5: return "mild"
        else: return "minimal"
    
    def _score_gad7(self, score: int) -> str:
        if score >= 15: return "severe"
        elif score >= 10: return "moderate"
        elif score >= 5: return "mild"
        else: return "minimal"
    
    def retrieve_clinical_context(self, analysis: Dict[str, Any]) -> str:
        """Retrieve relevant clinical context based on analysis"""
        query_terms = analysis["primary_concerns"] + analysis["follow_up_focus"]
        query = " ".join(query_terms)
        
        # Retrieve most relevant documents (limit to 2 to reduce token usage)
        docs = self.vector_store.similarity_search(query, k=2)
        
        context = "CLINICAL CONTEXT:\n"
        for i, doc in enumerate(docs):
            context += f"\n--- {doc.metadata.get('type', 'guideline').upper()} ---\n"
            context += doc.page_content[:500] + "\n"  # Shorter content
        
        return context
    
    def generate_follow_up_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized follow-up questions using LLM"""
        if not self.llm:
            return self._get_fallback_questions(analysis)
        
        try:
            # Retrieve relevant clinical context
            context = self.retrieve_clinical_context(analysis)
            
            # Generate questions using LLM with shorter context
            chain = self.follow_up_prompt | self.llm
            response = chain.invoke({
                "analysis": json.dumps(analysis, indent=2),
                "context": context[:1500],  # Limit context length
                "primary_concerns": ", ".join(analysis["primary_concerns"])
            })
            
            # Parse JSON response
            questions_text = response.strip()
            if questions_text.startswith('[') and questions_text.endswith(']'):
                questions = json.loads(questions_text)
                if len(questions) == 5:
                    print("🎯 LLM-generated personalized questions")
                    return questions
            
            # If parsing fails, extract questions from text
            questions = self._extract_questions_from_text(questions_text)
            if len(questions) >= 3:
                return questions[:5]
                
        except Exception as e:
            print(f"❌ Error generating questions with LLM: {e}")
        
        # Final fallback
        print("🔄 Using enhanced fallback questions")
        return self._get_enhanced_fallback_questions(analysis)
    
    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extract questions from LLM response text"""
        questions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for lines that look like questions
            if (line.startswith('"') and line.endswith('"')) or \
               (line.startswith("'") and line.endswith("'")) or \
               (line.endswith('?') and len(line) > 10):
                # Clean the question
                question = line.strip('"\'').split('?')[0] + '?'
                if len(question) > 15:  # Reasonable question length
                    questions.append(question)
        
        return questions[:5]
    
    def _get_enhanced_fallback_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Provide more personalized fallback questions"""
        # More specific question banks
        severe_depression_questions = [
            "When the weight feels heaviest, what goes through your mind?",
            "What does a 'better day' look like for you right now, even if it feels far away?",
            "How has this affected your sense of who you are?",
            "What keeps you going when everything feels overwhelming?",
            "If your pain could speak, what would it want me to understand?"
        ]
        
        moderate_depression_questions = [
            "Can you describe what a typical day looks like for you now compared to before these feelings started?",
            "What moments, if any, bring you even temporary relief from the heavy feelings?",
            "How has this affected your relationships with people you care about?",
            "What would you most want to change about how you're feeling right now?",
            "When you look ahead, what feels most uncertain or concerning to you?"
        ]
        
        anxiety_focused_questions = [
            "When anxiety peaks, what physical sensations do you notice in your body?",
            "Are there specific thoughts that tend to trigger the anxious feelings?",
            "What situations have you started avoiding because of how they make you feel?",
            "How does anxiety affect your sleep and morning routine?",
            "What have you found that provides even brief moments of calm?"
        ]
        
        sleep_focused_questions = [
            "What's your mind like when you're trying to fall asleep?",
            "How do you feel when you wake up - rested or something else?",
            "What happens in the hours before bed that might affect your sleep?",
            "How does poor sleep impact the following day for you?",
            "What have you tried that has helped even a little with sleep?"
        ]
        
        # Select questions based on analysis
        questions = []
        
        # Depression severity-based selection
        if analysis["depression_severity"] in ["severe", "moderately_severe"]:
            questions.extend(severe_depression_questions[:2])
        elif analysis["depression_severity"] == "moderate":
            questions.extend(moderate_depression_questions[:2])
        
        # Add anxiety questions if relevant
        if analysis["anxiety_severity"] in ["moderate", "severe"] and len(questions) < 4:
            questions.extend(anxiety_focused_questions[:1])
        
        # Add sleep questions if relevant
        if analysis["sleep_disturbance"] in ["moderate", "significant"] and len(questions) < 4:
            questions.extend(sleep_focused_questions[:1])
        
        # Fill remaining slots with general but personalized questions
        general_fallbacks = [
            "What would someone who knows you well say has changed most about you?",
            "If you could wave a magic wand and change one thing, what would it be?",
            "What small thing still feels meaningful to you?",
            "How has this experience changed what's important to you?",
            "What do you wish people understood about what you're going through?"
        ]
        
        while len(questions) < 5:
            for q in general_fallbacks:
                if q not in questions and len(questions) < 5:
                    questions.append(q)
        
        return questions[:5]
    
    def _get_fallback_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Original fallback questions"""
        question_bank = {
            "depression": [
                "How long have you been experiencing these low mood symptoms?",
                "What activities or interactions still bring you some sense of pleasure or accomplishment?",
                "How would you describe your energy levels throughout the day?",
                "Have you noticed changes in your ability to concentrate or make decisions?",
                "What does your support system look like right now?"
            ],
            "anxiety": [
                "Can you describe what happens in your body when you feel most anxious?",
                "Are there specific situations or thoughts that trigger these feelings?",
                "How does anxiety affect your ability to complete daily tasks?",
                "What techniques have you tried to manage anxious feelings?",
                "Do you ever experience panic attacks or intense fear episodes?"
            ],
            "sleep_disturbance": [
                "What does your typical sleep routine look like from evening to morning?",
                "How do you feel when you wake up in the morning?",
                "Do you find your mind racing when you try to sleep?",
                "What have you tried to improve your sleep quality?",
                "How does poor sleep affect your next day?"
            ],
            "general_wellbeing": [
                "How would you describe your overall quality of life right now?",
                "What aspects of your life are going well despite these challenges?",
                "What kind of support would be most helpful to you right now?",
                "How have you coped with difficult times in the past?",
                "What would you most like to change about how you're feeling?"
            ]
        }
        
        questions = []
        for concern in analysis["primary_concerns"]:
            if concern in question_bank:
                questions.extend(question_bank[concern][:2])
        
        # Fill remaining slots
        general_questions = question_bank["general_wellbeing"]
        while len(questions) < 5:
            for q in general_questions:
                if q not in questions and len(questions) < 5:
                    questions.append(q)
        
        return questions[:5]
    
    def generate_comprehensive_report(self, initial_answers: Dict[int, int], 
                                    follow_up_responses: Dict[str, str]) -> Dict[str, Any]:
        """Generate detailed clinical report using LLM"""
        if not self.llm:
            return self._generate_basic_report(initial_answers, follow_up_responses)
        
        try:
            # Analyze initial answers
            analysis = self.analyze_initial_answers(initial_answers)
            clinical_context = self.retrieve_clinical_context(analysis)
            
            # Prepare concise data for LLM
            answers_summary = self._summarize_answers(initial_answers)
            follow_up_summary = self._summarize_follow_up(follow_up_responses)
            
            # Generate report with limited context
            chain = self.report_prompt | self.llm
            response = chain.invoke({
                "initial_answers": answers_summary,
                "follow_up_answers": follow_up_summary,
                "clinical_context": clinical_context[:800]  # Strict limit
            })
            
            # Parse JSON response
            report_text = response.strip()
            try:
                report = json.loads(report_text)
                return self._validate_report_structure(report)
            except json.JSONDecodeError:
                return self._generate_basic_report(initial_answers, follow_up_responses)
                
        except Exception as e:
            print(f"❌ Error generating report with LLM: {e}")
            return self._generate_basic_report(initial_answers, follow_up_responses)
    
    def _summarize_answers(self, answers: Dict[int, int]) -> str:
        """Create concise summary of initial answers"""
        depression_score = sum(answers.get(i, 0) for i in range(9))
        anxiety_score = sum(answers.get(i, 0) for i in [4, 5, 6])
        return f"Depression: {depression_score}/27, Anxiety: {anxiety_score}/21, Sleep: {answers.get(2,0)}/3"
    
    def _summarize_follow_up(self, responses: Dict[str, str]) -> str:
        """Create concise summary of follow-up responses"""
        summary = []
        for q, a in list(responses.items())[:3]:  # Only use first 3
            summary.append(f"Q: {q[:50]}... A: {a[:30]}...")
        return " | ".join(summary)
    
    def _validate_report_structure(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure report has all required fields"""
        required_fields = [
            "risk_level", "diagnostic_considerations", "symptom_severity",
            "clinical_insights", "functional_impact", "recommendations", "crisis_indicators"
        ]
        
        for field in required_fields:
            if field not in report:
                report[field] = "Not specified"
        
        return report
    
    def _generate_basic_report(self, initial_answers: Dict[int, int], 
                             follow_up_responses: Dict[str, str]) -> Dict[str, Any]:
        """Generate basic report without LLM"""
        analysis = self.analyze_initial_answers(initial_answers)
        
        return {
            "risk_level": analysis["suicide_risk"],
            "diagnostic_considerations": [
                f"Potential {concern.replace('_', ' ')}" for concern in analysis["primary_concerns"]
            ],
            "symptom_severity": {
                "depression": analysis["depression_severity"],
                "anxiety": analysis["anxiety_severity"], 
                "sleep": analysis["sleep_disturbance"],
                "overall": "severe" if analysis["suicide_risk"] == "high" else "moderate"
            },
            "clinical_insights": [
                f"Primary concerns: {', '.join(analysis['primary_concerns'])}",
                f"Follow-up focus areas: {', '.join(analysis['follow_up_focus'])}"
            ],
            "functional_impact": "Assessment indicates significant impact on daily functioning",
            "recommendations": self._generate_basic_recommendations(analysis),
            "crisis_indicators": ["Suicide risk present"] if analysis["suicide_risk"] == "high" else []
        }
    
    def _generate_basic_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate basic clinical recommendations"""
        recommendations = []
        
        if analysis["suicide_risk"] == "high":
            recommendations.append("🚨 IMMEDIATE: Crisis assessment and safety planning required")
            recommendations.append("Consider urgent psychiatric evaluation")
        
        if analysis["depression_severity"] in ["moderate", "moderately_severe", "severe"]:
            recommendations.append("Comprehensive depression assessment recommended")
            recommendations.append("Consider therapy or medication evaluation")
        
        if analysis["anxiety_severity"] in ["moderate", "severe"]:
            recommendations.append("Anxiety management strategies indicated")
            recommendations.append("Consider cognitive-behavioral therapy")
        
        recommendations.append("Review social support systems and coping strategies")
        recommendations.append("Schedule follow-up within 1-2 weeks")
        
        return recommendations