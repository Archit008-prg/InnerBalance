import os
os.environ["HF_HOME"] = "F:\\huggingface"
import json
import random
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MeditronRAGSystem:
    def __init__(self):
        self.knowledge_base_path = "rag/knowledge_base/"
        self.vector_db_path = "rag/vector_store/chroma_db/"
        
        # Initialize components
        self.setup_embeddings()
        self.setup_vector_store()
        self.setup_prompts()
        
    @property
    def llm(self):
        """Lazy-loaded LLM pipeline"""
        if not hasattr(self, '_llm_loaded'):
            self._llm_loaded = True
            self._llm = None
            self.setup_meditron_llm()
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    def setup_embeddings(self):
        """Setup sentence embeddings for vector store"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"[RAG WARNING] Could not initialize local embeddings (skipping local vector store): {e}")
            self.embeddings = None
    
    def setup_vector_store(self):
        """Initialize or load vector store"""
        if not self.embeddings:
            self.vector_store = None
            print("Vector store disabled (no embeddings available)")
            return
            
        try:
            from langchain_community.vectorstores import Chroma
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
        except Exception as e:
            print(f"[RAG WARNING] Could not initialize Chroma vector store: {e}")
            self.vector_store = None
            
    def rebuild_vector_store(self):
        """Reset the vector store collection and reload all knowledge base documents"""
        if not self.vector_store:
            print("[RAG WARNING] Rebuild requested but vector store is not initialized.")
            return
            
        try:
            # Delete all documents in the Chroma collection
            results = self.vector_store._collection.get()
            if results and results.get('ids'):
                self.vector_store._collection.delete(ids=results['ids'])
            print("[VECTOR STORE] Cleared all existing documents from ChromaDB")
        except Exception as e:
            print(f"[VECTOR STORE WARNING] Failed to clear collection via API: {e}")
            
        self.load_knowledge_base()
        print("[VECTOR STORE] Successfully rebuilt and re-embedded knowledge base documents")
    
    def setup_meditron_llm(self):
        """Setup a reliable medical LLM - using stable models or Serverless API"""
        try:
            # Primary: Microsoft Phi-3 Mini (stable, good context)
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            
            # Check if Hugging Face API token is provided in .env
            from decouple import config
            hf_token = config("HF_TOKEN", default=None)
            
            if hf_token:
                print(f"[LLM] Using Hugging Face Serverless Inference API for {model_name}")
                class HuggingFaceServerlessLLM:
                    def __init__(self, model_id, token):
                        self.model_id = model_id
                        self.token = token
                    
                    def invoke(self, prompt, *args, **kwargs):
                        prompt_str = str(prompt)
                        if hasattr(prompt, 'to_string'):
                            prompt_str = prompt.to_string()
                        
                        import requests
                        api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
                        headers = {"Authorization": f"Bearer {self.token}"}
                        payload = {
                            "inputs": prompt_str,
                            "parameters": {
                                "max_new_tokens": 512,
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "do_sample": True,
                                "return_full_text": False
                            }
                        }
                        
                        # Retries for model startup (loading state)
                        for attempt in range(5):
                            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                            if response.status_code == 200:
                                res_json = response.json()
                                if isinstance(res_json, list) and len(res_json) > 0:
                                    return res_json[0].get("generated_text", "")
                                elif isinstance(res_json, dict):
                                    return res_json.get("generated_text", "")
                            elif response.status_code == 503:
                                print(f"[LLM WARNING] HF Serverless model is loading. Retrying in 5 seconds (attempt {attempt + 1}/5)...")
                                import time
                                time.sleep(5)
                            else:
                                raise Exception(f"HF API returned status {response.status_code}: {response.text}")
                        raise Exception("HF API model failed to load within timeout limits.")
                        
                    def __call__(self, prompt, stop=None, **kwargs):
                        return self.invoke(prompt)

                self.llm = HuggingFaceServerlessLLM(model_name, hf_token)
                print("[LLM] Hugging Face Serverless Inference API client initialized successfully")
                return
            
            print(f"[LLM] Loading model locally: {model_name}")
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from langchain_community.llms import HuggingFacePipeline
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {
                "trust_remote_code": True
            }
            if device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32
                
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
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
            
            print(f"[LLM] {model_name} loaded successfully")
            print("[LLM] Model features: 4K context, 3.8B parameters, instruction-tuned")
        
        except Exception as e:
            print(f"[LLM ERROR] Failed to load primary model: {e}")
            print("[LLM] Trying fallback model...")
            self.setup_fallback_llm()
    
    def setup_fallback_llm(self):
        """Fallback to ultra-stable model"""
        try:
            # Ultra-stable fallback
            model_name = "distilgpt2"
            
            print(f"[LLM] Loading fallback model: {model_name}")
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from langchain_community.llms import HuggingFacePipeline
            
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
            print(f"[LLM] {model_name} loaded successfully (fallback mode)")
            
        except Exception as e:
            print(f"[LLM ERROR] All models failed: {e}")
            print("[LLM] Proceeding without LLM - using rule-based system")
            self.llm = None
    
    def setup_prompts(self):
        """Setup specialized prompts for mental health assessment"""
        
        # Enhanced prompt for generating follow-up questions
        self.follow_up_prompt = PromptTemplate(
            input_variables=["analysis", "context", "primary_concerns"],
            template="""As a clinical psychologist, generate exactly 2 personalized follow-up questions based on this patient's specific symptoms.
CRITICAL CONSTRAINT: You must avoid addressing, validating, or engaging with the emotional details/feelings of the user. Maintain a strictly objective, factual, and symptom-focused clinical inquiry. Avoid therapeutic validation of emotions; instead, focus on collecting objective data such as frequency, duration, physical/somatic symptoms, specific triggers, and daily functional impact.

PATIENT ASSESSMENT:
{analysis}

CLINICAL GUIDANCE:
{context}

Create questions that are:
1. HIGHLY SPECIFIC to their symptom severity and patterns
2. Objective, factual, and clinically focused
3. Probing for specific details on frequency, duration, triggers, or somatic impact
4. CLINICALLY RELEVANT based on their primary concerns

Focus on understanding their specific struggles with {primary_concerns}.

Return ONLY a JSON array of exactly 2 questions. No explanations.

["question1", "question2"]"""
        )

        # Simplified report prompt
        self.report_prompt = PromptTemplate(
            input_variables=["initial_answers", "follow_up_answers", "clinical_context"],
            template="""As a psychiatrist, write a brief clinical report.
You must refer specifically to the patient's text answers under FOLLOW-UP RESPONSES.
If the patient's responses are vague, repetitive, or low-effort, explicitly note this under response_validity.
For every recommendation, mention the specific symptom or answer that triggered it.

ASSESSMENT DATA:
{initial_answers}

FOLLOW-UP RESPONSES:
{follow_up_answers}

GUIDELINES:
{clinical_context}

Generate a JSON report with:
- risk_level: (low/moderate/high)
- diagnostic_considerations: list of potential conditions
- symptom_severity: object mapping depression, anxiety, sleep, and overall
- clinical_insights: list explaining the reasons for diagnostic concerns, referencing the patient's specific inputs and context.
- functional_impact: string describing the impact on daily activities
- recommendations: list of actionable recommendations for the clinician, including WHY each is recommended based on the patient's data.
- crisis_indicators: list of acute warnings or response validation alerts.
- chief_complaint: a single concise sentence summarizing the core reason/primary underlying concern extracted from the patient's follow-up written answers.
- response_validity: object containing 'is_valid' (boolean, set to false ONLY if the patient typed complete non-compliant gibberish like 'asdf', 'blahh', or totally irrelevant off-topic text) and 'reason' (string explaining the decision).

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
        if documents and self.vector_store:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            self.vector_store.persist()
            print(f"[VECTOR STORE] Loaded {len(split_docs)} document chunks into vector store")
        elif not self.vector_store:
            print("[VECTOR STORE] Skipping database loading (vector store is disabled/inactive)")
        else:
            print("[VECTOR STORE WARNING] No documents found in knowledge base")
    
    def analyze_initial_answers(self, answers: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze first 18 answers using clinical scoring"""
        # Depression scoring based on questions 1, 2, 3, 4, 5, 11, 12, 13 (PHQ-8)
        depression_score = sum(int(answers.get(o, 0)) for o in [1, 2, 3, 4, 5, 11, 12, 13])
        
        # Anxiety scoring based on questions 6, 7, 8, 9, 10, 14, 15 (GAD-7)
        anxiety_score = sum(int(answers.get(o, 0)) for o in [6, 7, 8, 9, 10, 14, 15])
        
        # Sleep disturbance (question 3)
        sleep_score = int(answers.get(3, 0))
        
        # Sliders (1.0 to 5.0 intensity)
        anxiety_intensity = float(answers.get(16, 1.0))
        depression_intensity = float(answers.get(17, 1.0))
        
        # Functioning difficulty (question 18)
        functioning_score = int(answers.get(18, 0))
        
        analysis = {
            "depression_score": depression_score,
            "anxiety_score": anxiety_score,
            "depression_severity": self._score_phq9(depression_score * 1.125), # scale up to 27 max for normal PHQ-9 alignment
            "anxiety_severity": self._score_gad7(anxiety_score), # GAD-7 alignment (max 21)
            "sleep_disturbance": "significant" if sleep_score >= 2 else "moderate" if sleep_score == 1 else "minimal",
            "suicide_risk": "low", # Suicide risk is now handled via adaptive text assessment
            "anxiety_intensity": anxiety_intensity,
            "depression_intensity": depression_intensity,
            "functioning_difficulty": self._score_functioning(functioning_score),
            "primary_concerns": [],
            "follow_up_focus": []
        }
        
        # Determine primary concerns
        if depression_score >= 8 or depression_intensity >= 3.0:
            analysis["primary_concerns"].append("depression")
            analysis["follow_up_focus"].extend(["mood_patterns", "anhedonia", "cognitive_symptoms"])
        
        if anxiety_score >= 7 or anxiety_intensity >= 3.0:
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
    
    def _score_phq9(self, score: float) -> str:
        if score >= 20: return "severe"
        elif score >= 15: return "moderately_severe" 
        elif score >= 10: return "moderate"
        elif score >= 5: return "mild"
        else: return "minimal"
    
    def _score_gad7(self, score: float) -> str:
        if score >= 15: return "severe"
        elif score >= 10: return "moderate"
        elif score >= 5: return "mild"
        else: return "minimal"

    def _score_functioning(self, score: int) -> str:
        if score >= 3: return "extremely difficult"
        elif score == 2: return "very difficult"
        elif score == 1: return "somewhat difficult"
        else: return "not difficult at all"
    
    def retrieve_clinical_context(self, analysis: Dict[str, Any]) -> tuple:
        """Retrieve relevant clinical context based on analysis"""
        query_terms = analysis["primary_concerns"] + analysis["follow_up_focus"]
        query = " ".join(query_terms)
        
        # Retrieve most relevant documents (k=4 for richer context)
        docs = []
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=4)
            except Exception as e:
                print(f"[RAG WARNING] Similarity search failed: {e}")
        
        context = "CLINICAL CONTEXT:\n"
        referred_sources = []
        
        if docs:
            for i, doc in enumerate(docs):
                source_name = doc.metadata.get('source', 'Clinical Guideline')
                # Clean name for readable display (e.g. clinical_guidelines/nice_depression.txt -> Nice Depression)
                display_name = source_name.split("/")[-1].replace(".txt", "").replace(".json", "").replace("_", " ").title()
                
                if display_name not in referred_sources:
                    referred_sources.append(display_name)
                    
                context += f"\n--- {display_name.upper()} ---\n"
                context += doc.page_content[:1200] + "\n"  # Shorter content
        else:
            # Fallback clinical context: read directly from files!
            guideline_files = [
                "clinical_guidelines/nice_depression.txt",
                "clinical_guidelines/who_mhgap.txt",
                "diagnostic_criteria/anxiety_dsm5.txt"
            ]
            for file_path in guideline_files:
                full_path = os.path.join(self.knowledge_base_path, file_path)
                if os.path.exists(full_path):
                    display_name = file_path.split("/")[-1].replace(".txt", "").replace("_", " ").title()
                    referred_sources.append(display_name)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            context += f"\n--- {display_name.upper()} ---\n"
                            context += content[:1000] + "\n"
                    except Exception as e:
                        print(f"[RAG WARNING] Failed to read fallback file {file_path}: {e}")
        
        return context, referred_sources
    
    def generate_follow_up_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized follow-up questions using LLM"""
        if not self.llm:
            return self._get_fallback_questions(analysis)
        
        try:
            # Retrieve relevant clinical context
            context, _ = self.retrieve_clinical_context(analysis)
            
            # Generate questions using LLM with shorter context
            chain = self.follow_up_prompt | self.llm
            response = chain.invoke({
                "analysis": json.dumps(analysis, indent=2),
                "context": context[:4000],  # Limit context length to 4K chars
                "primary_concerns": ", ".join(analysis["primary_concerns"])
            })
            
            # Parse JSON response
            questions_text = response.strip()
            questions = []
            if questions_text.startswith('[') and questions_text.endswith(']'):
                try:
                    questions = json.loads(questions_text)
                except Exception:
                    pass
            
            # If parsing fails, extract questions from text
            if not isinstance(questions, list) or not questions:
                questions = self._extract_questions_from_text(questions_text)
            
            # Filter and sanitize questions
            questions = [str(q).strip() for q in questions if q and len(str(q).strip()) > 10]
            
            # Ensure we have exactly 2 questions
            if len(questions) < 2:
                questions = self._pad_questions(questions, analysis, 2)
            else:
                questions = questions[:2]
            print(f"[LLM] LLM-generated personalized questions (count: {len(questions)})")
            return questions
                
        except Exception as e:
            print(f"[LLM ERROR] Error generating questions with LLM: {e}")
        
        # Final fallback
        print("[LLM] Using enhanced fallback questions")
        return self._get_enhanced_fallback_questions(analysis)[:2]
    
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
        
        return questions
        
    def _pad_questions(self, questions: List[str], analysis: Dict[str, Any], target_count: int) -> List[str]:
        """Pad the question list with high-quality fallback questions to reach target_count"""
        fallback_pool = self._get_enhanced_fallback_questions(analysis)
        for q in fallback_pool:
            if q not in questions and len(questions) < target_count:
                questions.append(q)
        return questions
    
    def _get_enhanced_fallback_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Provide more personalized fallback questions"""
        # More specific question banks
        severe_depression_questions = [
            "When the weight feels heaviest, what goes through your mind?",
            "What does a 'better day' look like for you right now, even if it feels far away?",
            "How has this affected your sense of who you are?",
            "What keeps you going when everything feels overwhelming?",
            "If your pain could speak, what would it want me to understand?",
            "How do these low feelings impact your self-esteem and confidence?",
            "Do you find yourself withdrawing from activities that used to bring you joy?",
            "Have you noticed changes in your concentration or ability to make decisions?",
            "How does this current depressive state affect your energy levels throughout the day?",
            "What kind of support from family or friends has felt most helpful during this time?"
        ]
        
        moderate_depression_questions = [
            "Can you describe what a typical day looks like for you now compared to before these feelings started?",
            "What moments, if any, bring you even temporary relief from the heavy feelings?",
            "How has this affected your relationships with people you care about?",
            "What would you most want to change about how you're feeling right now?",
            "When you look ahead, what feels most uncertain or concerning to you?",
            "How do these mood swings affect your focus at work or daily responsibilities?",
            "Do you find your appetite or eating habits have shifted significantly?",
            "What coping mechanisms have you tried recently, and did they make any difference?",
            "How do you feel in the mornings compared to the evenings?",
            "If you could talk to someone about one specific concern right now, what would it be?"
        ]
        
        anxiety_focused_questions = [
            "When anxiety peaks, what physical sensations do you notice in your body?",
            "Are there specific thoughts that tend to trigger the anxious feelings?",
            "What situations have you started avoiding because of how they make you feel?",
            "How does anxiety affect your sleep and morning routine?",
            "What have you found that provides even brief moments of calm?",
            "Do you experience sudden moments of panic or heart racing for no clear reason?",
            "How does anxiety affect your communication with others?",
            "Do you feel a constant sense of restlessness or being on edge?",
            "How do you try to calm your racing thoughts when they feel out of control?",
            "How has anxiety impacted your decision-making or daily plans?"
        ]
        
        sleep_focused_questions = [
            "What's your mind like when you're trying to fall asleep?",
            "How do you feel when you wake up - rested or something else?",
            "What happens in the hours before bed that might affect your sleep?",
            "How does poor sleep impact the following day for you?",
            "What have you tried that has helped even a little with sleep?",
            "Do you find yourself waking up in the middle of the night and struggling to return to sleep?",
            "How long has this current pattern of sleep disruption been going on?",
            "How does lack of sleep affect your emotional resilience during the day?",
            "What is the ideal sleep environment for you, and how does it compare to your current setup?",
            "Do thoughts about sleep itself cause you stress or anxiety?"
        ]
        
        # Select questions based on analysis with randomization to ensure variety
        questions = []
        
        general_fallbacks = [
            "What would someone who knows you well say has changed most about you?",
            "If you could wave a magic wand and change one thing, what would it be?",
            "What small thing still feels meaningful to you?",
            "How has this experience changed what's important to you?",
            "What do you wish people understood about what you're going through?",
            "What has helped you get through difficult phases in the past?",
            "How do you typically express or release heavy emotions?",
            "What kind of setting or environment makes you feel safest and most at ease?",
            "How has your self-care routine changed since you started feeling this way?",
            "What is one goal, no matter how small, that you'd like to work towards?"
        ]

        severe_dep_pool = list(severe_depression_questions)
        mod_dep_pool = list(moderate_depression_questions)
        anxiety_pool = list(anxiety_focused_questions)
        sleep_pool = list(sleep_focused_questions)
        fallback_pool = list(general_fallbacks)
        
        random.shuffle(severe_dep_pool)
        random.shuffle(mod_dep_pool)
        random.shuffle(anxiety_pool)
        random.shuffle(sleep_pool)
        random.shuffle(fallback_pool)
        
        # Depression severity-based selection
        if analysis["depression_severity"] in ["severe", "moderately_severe"]:
            questions.extend(severe_dep_pool[:8])
        elif analysis["depression_severity"] == "moderate":
            questions.extend(mod_dep_pool[:8])
        else:
            questions.extend(mod_dep_pool[:4])
        
        # Add anxiety questions if relevant
        if analysis["anxiety_severity"] in ["moderate", "severe"]:
            questions.extend(anxiety_pool[:6])
        else:
            questions.extend(anxiety_pool[:3])
        
        # Add sleep questions if relevant
        if analysis["sleep_disturbance"] in ["moderate", "significant"]:
            questions.extend(sleep_pool[:4])
        else:
            questions.extend(sleep_pool[:2])
        
        # Unique list
        unique_questions = []
        for q in questions:
            if q not in unique_questions:
                unique_questions.append(q)
                
        for q in fallback_pool:
            if q not in unique_questions and len(unique_questions) < 10:
                unique_questions.append(q)
                
        # Hard fallback check
        all_shuffled = severe_dep_pool + mod_dep_pool + anxiety_pool + sleep_pool
        random.shuffle(all_shuffled)
        while len(unique_questions) < 2:
            for q in all_shuffled:
                if q not in unique_questions and len(unique_questions) < 2:
                    unique_questions.append(q)
                    
        return unique_questions[:2]
    
    def _get_fallback_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Original fallback questions wrapper - updated to return 2 questions"""
        return self._get_enhanced_fallback_questions(analysis)
    
    def generate_comprehensive_report(self, initial_answers: Dict[int, int], 
                                    follow_up_responses: Dict[str, str]) -> Dict[str, Any]:
        """Generate detailed clinical report using LLM with response effort verification"""
        report = {}
        # Always run analysis and clinical context search to identify referenced sources
        analysis = self.analyze_initial_answers(initial_answers)
        clinical_context, referred_sources = self.retrieve_clinical_context(analysis)
        
        if not self.llm:
            report = self._generate_basic_report(initial_answers, follow_up_responses)
        else:
            try:
                # Prepare concise data for LLM
                answers_summary = self._summarize_answers(initial_answers)
                follow_up_summary = self._summarize_follow_up(follow_up_responses)
                
                # Generate report with limited context
                chain = self.report_prompt | self.llm
                response = chain.invoke({
                    "initial_answers": answers_summary,
                    "follow_up_answers": follow_up_summary,
                    "clinical_context": clinical_context[:4000]  # Increased context window characters
                })
                
                # Parse JSON response
                report_text = response.strip()
                try:
                    # Robust extraction of JSON substring
                    start_idx = report_text.find('{')
                    end_idx = report_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = report_text[start_idx:end_idx+1]
                        report = json.loads(json_str)
                    else:
                        report = json.loads(report_text)
                    
                    report = self._validate_report_structure(report)
                except Exception as json_err:
                    print(f"[JSON ERROR] Failed to extract/parse LLM JSON: {json_err}. Raw response snippet: {report_text[:250]}")
                    report = self._generate_basic_report(initial_answers, follow_up_responses)
            except Exception as e:
                print(f"[LLM ERROR] Error generating report with LLM: {e}")
                report = self._generate_basic_report(initial_answers, follow_up_responses)
        
        # Check LLM response validity flag first
        llm_valid = True
        is_llm_report = "response_validity" in report
        if is_llm_report and isinstance(report["response_validity"], dict):
            llm_valid = report["response_validity"].get("is_valid", True)
            
        # Fallback validity: our simplified hardcoded checks
        validity = self._check_response_validity(follow_up_responses)
        report["validation"] = validity
        
        # Inject patient's actual text responses so they are always visible in the report copy
        report["patient_responses"] = [
            {"question": q, "answer": a} for q, a in follow_up_responses.items()
        ]
        
        # Inject referred source documents list
        report["referred_sources"] = referred_sources
        
        # Override clinical metrics if LLM dynamically flags as invalid, or if using fallback and rules flag it
        is_invalid = (not llm_valid) if is_llm_report else validity["low_effort"]
        
        if is_invalid:
            report["risk_level"] = "unreliable"
            report["diagnostic_considerations"] = ["Assessment Unreliable: Vague/off-topic response effort detected"]
            report["symptom_severity"] = {
                "depression": "unreliable",
                "anxiety": "unreliable",
                "sleep": "unreliable",
                "overall": "unreliable"
            }
            # List actual responses to show the physician what was typed
            actual_responses_str = ", ".join(f"'{a}'" for a in follow_up_responses.values() if str(a).strip())
            report["clinical_insights"] = [
                "The patient's textual answers do not contain relevant clinical info.",
                f"Patient's actual input text: {actual_responses_str}"
            ]
            report["functional_impact"] = "Cannot be assessed due to non-compliant response effort."
            report["recommendations"] = [
                "⚠️ Clinical attention required: Patient text responses were flagged as repetitive, off-topic, or clinically uninformative.",
                "Conduct a face-to-face or direct verbal clinical intake to reconstruct patient history.",
                "Discard the automated symptom severity scales as the responses are invalid."
            ]
            report["chief_complaint"] = "Vague/off-topic response effort detected."
            if "crisis_indicators" not in report or not isinstance(report["crisis_indicators"], list):
                report["crisis_indicators"] = []
            report["crisis_indicators"].extend(validity["alerts"])
            
        return report
    
    def _check_response_validity(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate if patient's textual responses are clinical, meaningful, or low-effort/vague/off-topic.
        Returns a dictionary with status, warnings, and messages.
        """
        if not responses:
            return {
                "is_valid": True,
                "low_effort": False,
                "alerts": []
            }
            
        total_answers = 0
        vague_answers = 0
        unique_answers = set()
        gibberish_words = {
            "blah", "blahh", "blahhh", "asdf", "xyz", "abc", "qwerty"
        }
        
        for q, a in responses.items():
            a_clean = str(a).strip().lower()
            if not a_clean:
                continue
            total_answers += 1
            unique_answers.add(a_clean)
            
            # An answer is vague in rules ONLY if it is empty, tiny (< 3 chars), in gibberish words, or repeated characters
            is_vague = False
            if len(a_clean) < 3:
                is_vague = True
            elif a_clean in gibberish_words:
                is_vague = True
            elif all(c == a_clean[0] for c in a_clean) and len(a_clean) > 2:
                is_vague = True
                
            if is_vague:
                vague_answers += 1
                
        # If the user typed the exact same answer for multiple questions
        is_repetitive = total_answers > 2 and len(unique_answers) <= 1
        
        # Flag if ANY answer is vague, off-topic, or gibberish, or if answers are highly repetitive
        is_low_effort = vague_answers > 0 or is_repetitive
        
        alerts = []
        if is_low_effort:
            alerts.append(
                "⚠️ RESPONSE VALIDATION ALERT: Patient input was flagged as repetitive, off-topic, or clinically uninformative. Clinical intake validity is compromised."
            )
            
        return {
            "is_valid": not is_low_effort,
            "low_effort": is_low_effort,
            "alerts": alerts,
            "vague_count": vague_answers,
            "total_count": total_answers
        }

    def _summarize_answers(self, answers: Dict[int, int]) -> str:
        """Create concise summary of initial answers"""
        depression_score = sum(int(answers.get(o, 0)) for o in [1, 2, 3, 4, 5, 11, 12, 13])
        anxiety_score = sum(int(answers.get(o, 0)) for o in [6, 7, 8, 9, 10, 14, 15])
        return f"Depression: {depression_score}/24, Anxiety: {anxiety_score}/21"

    def _summarize_follow_up(self, responses: Dict[str, str]) -> str:
        """Create concise summary of follow-up responses"""
        summary = []
        for q, a in list(responses.items()):
            summary.append(f"Q: {q[:100]} A: {a[:2000]}")
        return " | ".join(summary)
    
    def _validate_report_structure(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure report has all required fields"""
        required_fields = [
            "risk_level", "diagnostic_considerations", "symptom_severity",
            "clinical_insights", "functional_impact", "recommendations", 
            "crisis_indicators", "chief_complaint", "response_validity"
        ]
        
        for field in required_fields:
            if field not in report:
                if field == "response_validity":
                    report[field] = {"is_valid": True, "reason": "Defaulting to valid"}
                else:
                    report[field] = "Not specified"
        
        return report
    
    def _generate_basic_report(self, initial_answers: Dict[int, int], 
                              follow_up_responses: Dict[str, str]) -> Dict[str, Any]:
        """Generate basic report without LLM"""
        analysis = self.analyze_initial_answers(initial_answers)
        
        # Rule-based crisis keywords detection in patient's textual responses
        crisis_keywords = [
            "suicide", "kill myself", "end my life", "hurt myself", "cutting", 
            "overdose", "want to die", "better off dead", "self-harm", "hanging", 
            "poisoning", "jump from", "suicidal"
        ]
        
        has_crisis_text = False
        if follow_up_responses:
            for val in follow_up_responses.values():
                val_lower = str(val).lower()
                if any(kw in val_lower for kw in crisis_keywords):
                    has_crisis_text = True
                    break
        
        # Dynamic risk level determination
        risk_level = "low"
        if has_crisis_text:
            risk_level = "crisis"
        elif analysis["depression_severity"] == "severe" or analysis["anxiety_severity"] == "severe":
            risk_level = "high"
        elif analysis["depression_severity"] in ["moderate", "moderately_severe"] or analysis["anxiety_severity"] == "moderate":
            risk_level = "moderate"
            
        # Update analysis risk
        analysis["suicide_risk"] = "high" if risk_level in ["high", "crisis"] else risk_level
        
        # Extract a single-sentence core complaint from patient responses for the clinician
        core_reason = ""
        if follow_up_responses:
            combined_responses = " ".join(val for val in follow_up_responses.values() if val).strip()
            if combined_responses:
                # Take the first sentence or first 120 characters
                first_sentence = combined_responses.split(".")[0].strip()
                if len(first_sentence) > 120:
                    core_reason = first_sentence[:117] + "..."
                else:
                    core_reason = first_sentence + "."
        
        if not core_reason:
            core_reason = "Initial screening for mental health evaluation."

        # Dynamically build highly detailed clinical insights
        clinical_insights = []
        
        # Depressive symptoms insights
        dep_symptoms = []
        if int(initial_answers.get(1, 0)) >= 2:
            dep_symptoms.append("significant loss of interest/pleasure (anhedonia)")
        if int(initial_answers.get(2, 0)) >= 2:
            dep_symptoms.append("persistent low or hopeless mood")
        if int(initial_answers.get(4, 0)) >= 2:
            dep_symptoms.append("pronounced fatigue or energy depletion")
        if int(initial_answers.get(11, 0)) >= 2:
            dep_symptoms.append("feelings of worthlessness or self-blame")
        if int(initial_answers.get(12, 0)) >= 2:
            dep_symptoms.append("cognitive focus deficits and difficulty concentrating")
        if int(initial_answers.get(13, 0)) >= 2:
            dep_symptoms.append("psychomotor deceleration or agitation")
            
        if dep_symptoms:
            clinical_insights.append(f"Depression Profile: Patient reports {', '.join(dep_symptoms)} over the past two weeks.")
            
        # Anxiety symptoms insights
        anx_symptoms = []
        if int(initial_answers.get(6, 0)) >= 2:
            anx_symptoms.append("frequent states of being on edge or highly nervous")
        if int(initial_answers.get(7, 0)) >= 2:
            anx_symptoms.append("uncontrollable or intrusive worry loops")
        if int(initial_answers.get(15, 0)) >= 2:
            anx_symptoms.append("difficulty achieving relaxation")
        if int(initial_answers.get(10, 0)) >= 2:
            anx_symptoms.append("somatic dread or acute fear indices")
            
        if anx_symptoms:
            clinical_insights.append(f"Anxiety Profile: Patient report reveals {', '.join(anx_symptoms)} indicating generalized distress patterns.")
            
        # Intensity insights
        dep_intensity = float(initial_answers.get(17, 1.0))
        anx_intensity = float(initial_answers.get(16, 1.0))
        clinical_insights.append(f"Distress Severity: Peak self-rated distress intensity registered at {dep_intensity}/5.0 for low mood, and {anx_intensity}/5.0 for anxiety symptoms.")
        
        # Sleep & Somatic patterns
        sleep_score = int(initial_answers.get(3, 0))
        appetite_score = int(initial_answers.get(5, 0))
        somatic_details = []
        if sleep_score >= 2:
            somatic_details.append("significant sleep disturbance (latency/maintenance issues)")
        if appetite_score >= 2:
            somatic_details.append("appetite shifts (overeating or restriction)")
            
        if somatic_details:
            clinical_insights.append(f"Vegetative Indicators: Somatic evaluations detect {' and '.join(somatic_details)} directly complicating daily functioning.")

        if has_crisis_text:
            clinical_insights.append("CRITICAL: Essay responses contain keyword patterns indicating possible active self-harm ideation or severe panic triggers.")
        else:
            clinical_insights.append("Crisis Safety: No acute crisis keywords detected in patient's open-ended responses.")

        # Summarize follow-up responses for context
        if follow_up_responses:
            essay_insights = []
            for question_text, answer_text in follow_up_responses.items():
                if answer_text and len(answer_text.strip()) > 5:
                    short_q = question_text.split("?")[0][:50]
                    essay_insights.append(f"Patient reported for '{short_q}': '{answer_text}'")
            if essay_insights:
                clinical_insights.extend(essay_insights[:2])

        # Dynamic functioning description
        func_diff = analysis.get("functioning_difficulty", "not difficult at all")
        if func_diff == "extremely difficult":
            functional_impact = "Assessment indicates extreme impairment: the patient finds managing daily activities (work, relationships, home life) extremely difficult."
        elif func_diff == "very difficult":
            functional_impact = "Assessment indicates significant impairment: the patient experiences major barriers to functioning at work, home, or in social settings."
        elif func_diff == "somewhat difficult":
            functional_impact = "Assessment indicates mild impairment: symptoms pose some day-to-day challenges, but functioning is largely preserved."
        else:
            functional_impact = "Assessment indicates minimal functional impairment: symptoms do not currently disrupt work, home, or social responsibilities."
            
        return {
            "risk_level": risk_level,
            "diagnostic_considerations": [
                f"Potential {concern.replace('_', ' ')}" for concern in analysis["primary_concerns"]
            ],
            "symptom_severity": {
                "depression": analysis["depression_severity"],
                "anxiety": analysis["anxiety_severity"], 
                "sleep": analysis["sleep_disturbance"],
                "sleep_disturbance": analysis["sleep_disturbance"],
                "overall": "severe" if risk_level in ["high", "crisis"] else "moderate" if risk_level == "moderate" else "minimal"
            },
            "anxiety_intensity": anx_intensity,
            "depression_intensity": dep_intensity,
            "functioning_difficulty": analysis["functioning_difficulty"],
            "clinical_insights": clinical_insights,
            "functional_impact": functional_impact,
            "recommendations": self._generate_basic_recommendations(analysis),
            "crisis_indicators": ["Suicide risk present / Crisis alerts triggered by patient response"] if has_crisis_text else [],
            "chief_complaint": core_reason
        }
    
    def _generate_basic_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate basic clinical recommendations based on actual score thresholds"""
        recommendations = []
        
        if analysis["suicide_risk"] == "high":
            recommendations.append("🚨 IMMEDIATE: Crisis assessment and safety planning required due to elevated suicidal ideation indices (score threshold reached).")
        
        # Depression indicators
        dep_severity = analysis["depression_severity"]
        dep_score = analysis.get("depression_score", 0)
        if dep_severity in ["severe", "moderately_severe"]:
            recommendations.append(f"Initiate major mood disorder treatment protocol. Standard clinical guidelines dictate full diagnostic workup for depression because the patient's raw PHQ-8 score ({dep_score}/24) indicates a {dep_severity.replace('_', ' ')} presentation.")
        elif dep_severity == "moderate":
            recommendations.append(f"Schedule diagnostic follow-up for moderate depressive symptoms (PHQ-8 score: {dep_score}/24) to rule out adjustment disorder or persistent depressive disorder.")
            
        # Anxiety indicators
        anx_severity = analysis["anxiety_severity"]
        anx_score = analysis.get("anxiety_score", 0)
        if anx_severity in ["severe", "moderate"]:
            recommendations.append(f"Conduct full generalized anxiety evaluation. GAD guideline protocols specify somatic and worry monitoring for {anx_severity} anxiety (GAD-7 score: {anx_score}/21).")
            
        # Sleep indicators
        sleep = analysis["sleep_disturbance"]
        if sleep in ["significant", "severe"]:
            recommendations.append("Recommend sleep study or sleep architecture questionnaire due to reports of significant sleep maintenance/latency disruption.")
            
        # General wellbeing
        recommendations.append("Initiate baseline mood and cognitive symptom charting to track severity fluctuations over the next 14 days.")
        recommendations.append("Schedule professional clinical intake interview within 1-2 weeks for diagnostic verification.")
        
        return recommendations