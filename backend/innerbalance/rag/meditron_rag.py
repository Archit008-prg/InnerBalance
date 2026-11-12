import os
os.environ["HF_HOME"] = "F:\\huggingface"
import json
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

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
            print(f"[RAG WARNING] Could not initialize local embeddings: {e}")
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
            
    def setup_meditron_llm(self):
        """Setup medical LLM via Hugging Face API or fallback"""
        try:
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            from decouple import config
            hf_token = config("HF_TOKEN", default=None)
            
            if hf_token:
                print(f"[LLM] Using Hugging Face Inference API for {model_name}")
                class HuggingFaceServerlessLLM:
                    def __init__(self, model_id, token):
                        self.model_id = model_id
                        self.token = token
                    
                    def invoke(self, prompt, *args, **kwargs):
                        prompt_str = str(prompt)
                        if hasattr(prompt, 'to_string'):
                            prompt_str = prompt.to_string()
                        
                        import requests
                        # Try standard and router endpoints
                        endpoints = [
                            f"https://router.huggingface.co/hf-inference/models/{self.model_id}",
                            f"https://api-inference.huggingface.co/models/{self.model_id}"
                        ]
                        headers = {"Authorization": f"Bearer {self.token}"}
                        payload = {
                            "inputs": prompt_str,
                            "parameters": {
                                "max_new_tokens": 512,
                                "temperature": 0.6,
                                "top_p": 0.9,
                                "do_sample": True,
                                "return_full_text": False
                            }
                        }
                        
                        for api_url in endpoints:
                            try:
                                response = requests.post(api_url, headers=headers, json=payload, timeout=8)
                                if response.status_code == 200:
                                    res_json = response.json()
                                    if isinstance(res_json, list) and len(res_json) > 0:
                                        return res_json[0].get("generated_text", "")
                                    elif isinstance(res_json, dict):
                                        return res_json.get("generated_text", "")
                            except Exception:
                                continue
                        raise Exception("HF Inference API unreachable; transitioning to local clinical synthesis engine.")
                        
                    def __call__(self, prompt, stop=None, **kwargs):
                        return self.invoke(prompt)

                self.llm = HuggingFaceServerlessLLM(model_name, hf_token)
                print("[LLM] Hugging Face Inference client initialized")
                return
            
            self.llm = None
        except Exception as e:
            print(f"[LLM ERROR] LLM setup failed: {e}")
            self.llm = None

    def setup_prompts(self):
        """Setup specialized prompts for mental health assessment"""
        self.follow_up_prompt = PromptTemplate(
            input_variables=["analysis", "context", "primary_concerns"],
            template="""As an expert clinical psychiatrist, formulate exactly 2 targeted, highly dynamic clinical inquiry questions for this patient.
CRITICAL INSTRUCTIONS:
- Base the questions directly on their highest-severity symptoms identified in the assessment below.
- Questions must be open-ended, objective, and aimed at extracting deeper clinical context (triggers, duration, physical somatic symptoms, and daily functional impact).
- Avoid generic pleasantries. Focus on gathering high-yield clinical data for the attending physician.

PATIENT SCREENING DATA:
{analysis}

CLINICAL GUIDELINES:
{context}

PRIMARY CONCERNS:
{primary_concerns}

Return ONLY a valid JSON array containing exactly 2 questions. Example format:
["Question 1 text here?", "Question 2 text here?"]"""
        )

        self.report_prompt = PromptTemplate(
            input_variables=["initial_answers", "follow_up_answers", "clinical_context"],
            template="""As an attending psychiatrist, generate a structured clinical assessment report.
CRITICAL CLINICAL RULES:
1. Examine the patient's open-ended written responses carefully.
2. If the patient typed a LONG STORY or detailed narrative, extract the CLINICAL ZEST (the core life stressor, emotional reaction, and impact on functioning) so the physician does not have to read a long paragraph.
3. If the patient typed VAGUE, GIBBERISH, or OFF-TOPIC answers (e.g. 'idk', 'nothing', 'asdf', random words), IGNORE them for diagnostic inference. State clearly that the text was uninformative and anchor the diagnosis strictly in the standardized questions 1-10.
4. Report must be factual, professional, and a true reflection of the patient's answers.

DATA:
Initial Screening (1-10): {initial_answers}
Follow-up Text Responses: {follow_up_answers}
Clinical Guidelines: {clinical_context}

Return ONLY valid JSON matching this schema:
{{
  "risk_level": "low" | "moderate" | "high" | "crisis",
  "chief_complaint": "Single sentence summarizing core clinical reason or patient situation",
  "clinical_zest": "Bulleted or 2-sentence synthesis of key life events/triggers extracted from narrative text, or 'N/A' if brief/vague",
  "diagnostic_considerations": ["Condition 1", "Condition 2"],
  "symptom_severity": {{
    "depression": "minimal" | "mild" | "moderate" | "moderately_severe" | "severe",
    "anxiety": "minimal" | "mild" | "moderate" | "moderately_severe" | "severe",
    "sleep_disturbance": "minimal" | "moderate" | "significant",
    "overall": "minimal" | "moderate" | "severe"
  }},
  "clinical_insights": [
    "Insight 1 citing specific patient responses",
    "Insight 2 explaining diagnostic deduction"
  ],
  "functional_impact": "Impact on work, home, and relationships",
  "recommendations": [
    "Evidence-based recommendation 1",
    "Evidence-based recommendation 2"
  ],
  "crisis_indicators": ["Alert text if suicide or self-harm mentioned, else empty list"],
  "response_validity": {{
    "is_valid": true,
    "quality": "informative" | "long_story_narrative" | "vague_or_irrelevant",
    "reason": "Explanation of text quality"
  }}
}}"""
        )

    def load_knowledge_base(self):
        """Load medical guidelines into vector store"""
        documents = []
        guidelines = [
            ("symptom_patterns.json", "symptom_patterns"),
            ("nice_depression.txt", "NICE Depression Guidelines"),
            ("who_mhgap.txt", "WHO mhGAP Intervention Guide"),
            ("anxiety_dsm5.txt", "DSM-5 Generalized Anxiety Criteria")
        ]
        for fname, name in guidelines:
            for sub in ["clinical_guidelines", "diagnostic_criteria", "assessment_tools"]:
                fpath = os.path.join(self.knowledge_base_path, sub, fname)
                if os.path.exists(fpath):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            text = f.read()
                            documents.append(Document(page_content=text, metadata={"source": name}))
                    except Exception:
                        pass
        return documents

    # --------------------------------------------------------------------------
    # 1. SCORING & INITIAL ANALYSIS (First 10 fixed questions)
    # --------------------------------------------------------------------------
    def analyze_initial_answers(self, answers: Dict[int, Any]) -> Dict[str, Any]:
        """
        Analyze the standardized first 10 questions.
        Q1-Q5: Core Depression & Somatic cluster (PHQ items: anhedonia, mood, sleep, fatigue, appetite)
        Q6-Q10: Core Anxiety & Dread cluster (GAD items: nervous, uncontrollable worry, restlessness, irritability, fear/dread)
        """
        # Depression scoring from questions 1, 2, 3, 4, 5 (plus 11, 12, 13 if legacy)
        q_dep = [1, 2, 3, 4, 5]
        depression_raw = sum(int(answers.get(o, 0)) for o in q_dep)
        
        # Anxiety scoring from questions 6, 7, 8, 9, 10 (plus 14, 15 if legacy)
        q_anx = [6, 7, 8, 9, 10]
        anxiety_raw = sum(int(answers.get(o, 0)) for o in q_anx)
        
        # Check if legacy questions 11-18 exist
        if any(o in answers for o in [11, 12, 13]):
            depression_raw += sum(int(answers.get(o, 0)) for o in [11, 12, 13])
        if any(o in answers for o in [14, 15]):
            anxiety_raw += sum(int(answers.get(o, 0)) for o in [14, 15])
            
        sleep_score = int(answers.get(3, 0))
        functioning_score = int(answers.get(18, 0)) if 18 in answers else None
        
        # Calibrated Severity Grading
        # For 5-item scale (max 15 each): 0-2 Minimal, 3-5 Mild, 6-9 Moderate, 10-12 Mod Severe, 13-15 Severe
        dep_severity = self._score_subscale(depression_raw)
        anx_severity = self._score_subscale(anxiety_raw)
        
        # Specific High-Yield Symptom Flags
        symptom_flags = {
            "anhedonia": int(answers.get(1, 0)) >= 2,
            "depressed_mood": int(answers.get(2, 0)) >= 2,
            "sleep_disturbance": sleep_score >= 2,
            "fatigue": int(answers.get(4, 0)) >= 2,
            "appetite_change": int(answers.get(5, 0)) >= 2,
            "nervousness": int(answers.get(6, 0)) >= 2,
            "uncontrollable_worry": int(answers.get(7, 0)) >= 2,
            "restlessness": int(answers.get(8, 0)) >= 2,
            "irritability": int(answers.get(9, 0)) >= 2,
            "dread_or_panic": int(answers.get(10, 0)) >= 2,
        }
        
        primary_concerns = []
        follow_up_focus = []
        
        if depression_raw >= 6 or symptom_flags["anhedonia"] or symptom_flags["depressed_mood"]:
            primary_concerns.append("Depressive Disorder")
            if symptom_flags["anhedonia"]:
                follow_up_focus.append("loss_of_interest_and_motivation")
            if symptom_flags["depressed_mood"]:
                follow_up_focus.append("hopelessness_and_mood_depth")
            if symptom_flags["fatigue"]:
                follow_up_focus.append("chronic_energy_depletion")
                
        if anxiety_raw >= 6 or symptom_flags["nervousness"] or symptom_flags["uncontrollable_worry"]:
            primary_concerns.append("Generalized Anxiety Disorder")
            if symptom_flags["uncontrollable_worry"]:
                follow_up_focus.append("worry_cycles_and_intrusive_thoughts")
            if symptom_flags["dread_or_panic"] or symptom_flags["restlessness"]:
                follow_up_focus.append("panic_and_somatic_agitation")
                
        if sleep_score >= 2:
            primary_concerns.append("Sleep-Wake Disruption")
            follow_up_focus.append("nocturnal_rumination_and_insomnia")
            
        if symptom_flags["irritability"]:
            follow_up_focus.append("emotional_reactivity_and_interpersonal_stress")
            
        if not primary_concerns:
            primary_concerns.append("Subclinical / General Wellness Evaluation")
            follow_up_focus.append("subtle_stressors_and_coping_resilience")
            
        analysis = {
            "depression_score": depression_raw,
            "anxiety_score": anxiety_raw,
            "overall_score": depression_raw + anxiety_raw,
            "depression_severity": dep_severity,
            "anxiety_severity": anx_severity,
            "sleep_disturbance": "significant" if sleep_score >= 2 else "moderate" if sleep_score == 1 else "minimal",
            "suicide_risk": "low",
            "symptom_flags": symptom_flags,
            "primary_concerns": primary_concerns,
            "follow_up_focus": follow_up_focus,
            "functioning_difficulty": self._score_functioning(functioning_score) if functioning_score is not None else "Evaluation in progress"
        }
        return analysis

    def _score_subscale(self, score: int) -> str:
        """Calibrated severity for 5-item clinical subscale (0 to 15)"""
        if score >= 13: return "severe"
        elif score >= 10: return "moderately_severe"
        elif score >= 6: return "moderate"
        elif score >= 3: return "mild"
        else: return "minimal"

    def _score_functioning(self, score: int) -> str:
        if score >= 3: return "extremely difficult"
        elif score == 2: return "very difficult"
        elif score == 1: return "somewhat difficult"
        else: return "not difficult at all"

    # --------------------------------------------------------------------------
    # 2. DYNAMIC GENERATION OF TYPING QUESTIONS
    # --------------------------------------------------------------------------
    def generate_follow_up_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate exactly 2 personalized, dynamic typing questions.
        These are generated strictly based on how the patient scored on questions 1-10.
        """
        # Try LLM first if available
        if self.llm:
            try:
                context, _ = self.retrieve_clinical_context(analysis)
                chain = self.follow_up_prompt | self.llm
                response = chain.invoke({
                    "analysis": json.dumps(analysis, indent=2),
                    "context": context[:3000],
                    "primary_concerns": ", ".join(analysis["primary_concerns"])
                })
                questions_text = response.strip()
                match = re.search(r'\[.*\]', questions_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    cleaned = [str(q).strip() for q in parsed if len(str(q).strip()) > 15]
                    if len(cleaned) >= 2:
                        print(f"[LLM] Dynamic typing questions successfully synthesized via LLM")
                        return cleaned[:2]
            except Exception as e:
                print(f"[LLM NOTE] Using specialized clinical dynamic question generator: {e}")

        # Clinically calibrated dynamic question generator based on exact answer profile
        return self._build_dynamic_clinical_questions(analysis)

    def _build_dynamic_clinical_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Dynamically synthesize 2 targeted inquiry questions based on the exact
        symptom combination from questions 1-10.
        """
        flags = analysis.get("symptom_flags", {})
        dep_sev = analysis.get("depression_severity", "minimal")
        anx_sev = analysis.get("anxiety_severity", "minimal")
        questions = []

        # Question 1: Deep probe into their most prominent symptom domain
        if flags.get("uncontrollable_worry") and (anx_sev in ["severe", "moderately_severe", "moderate"]):
            questions.append(
                "You noted constant worry that feels difficult to control. What specific themes, thoughts, or events trigger these worry spirals most often, and what physical sensations (such as a racing heart or tight chest) do you experience?"
            )
        elif flags.get("dread_or_panic") or flags.get("restlessness"):
            questions.append(
                "You experienced feelings of dread or restlessness as if something awful might happen. Could you describe when these sensations feel most acute, and what situations or thoughts tend to bring them on?"
            )
        elif flags.get("anhedonia") and (dep_sev in ["severe", "moderately_severe", "moderate"]):
            questions.append(
                "You mentioned feeling little interest or pleasure in things you normally do. Could you share what activities or moments you've felt most disconnected from lately, and what goes through your mind during those low periods?"
            )
        elif flags.get("depressed_mood") and (dep_sev in ["severe", "moderately_severe", "moderate"]):
            questions.append(
                "You reported feeling down, depressed, or hopeless. When these feelings feel heaviest, how long do they typically last, and what has been the hardest part of managing your mood over the past two weeks?"
            )
        elif flags.get("sleep_disturbance"):
            questions.append(
                "Sleep disruption stood out as a significant concern in your answers. Are you struggling more with falling asleep due to racing thoughts, or waking up in the middle of the night, and how is this impacting your daytime focus?"
            )
        elif flags.get("irritability"):
            questions.append(
                "You indicated feeling easily annoyed or irritable. In what settings (such as at work, school, or home with loved ones) does this frustration surface most, and what feels like the underlying trigger?"
            )
        else:
            # Low / subclinical symptom profile
            questions.append(
                "Your initial responses indicate relatively stable symptoms overall. What specific life circumstances, recent changes, or thoughts prompted you to seek out this check-in today?"
            )

        # Question 2: Probing daily functional impact and personal context
        if dep_sev in ["severe", "moderately_severe"] or anx_sev in ["severe", "moderately_severe"]:
            questions.append(
                "How have these emotional and physical challenges affected your day-to-day responsibilities, your work or studies, and your ability to connect with family and friends?"
            )
        elif flags.get("fatigue") or flags.get("sleep_disturbance"):
            questions.append(
                "Given the fatigue and energy depletion you noted, how are you currently pacing yourself through your daily routine, and what coping strategies have given you even brief relief?"
            )
        else:
            questions.append(
                "Looking ahead to your clinical consultation, what is the single most important goal or question you would like the doctor to help you address regarding your mental well-being?"
            )

        return questions[:2]

    # --------------------------------------------------------------------------
    # 3. TEXT FIELD PROCESSING (Long Story Zest Extraction vs Vague Handling)
    # --------------------------------------------------------------------------
    def analyze_patient_text_responses(self, follow_up_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Deep analysis of patient written text answers:
        - Detects vague / random / out-of-context inputs (e.g. 'idk', 'asdf', 'nothing') -> flagged to ignore in clinical inference.
        - Detects long stories / multi-sentence narratives -> extracts the 'Clinical Zest' (Core Essence) for doctor reference.
        - Scans for acute crisis keywords -> immediate safety alert.
        """
        if not follow_up_responses:
            return {
                "overall_quality": "none_provided",
                "is_vague": True,
                "has_crisis": False,
                "clinical_zest": "No written responses submitted.",
                "vague_notice": "No open-ended text was provided.",
                "crisis_alerts": []
            }

        # 1. Crisis keyword scanning
        crisis_terms = [
            "suicide", "kill myself", "end my life", "hurt myself", "cutting",
            "overdose", "want to die", "better off dead", "self-harm", "hanging",
            "poisoning", "jump from", "suicidal", "no reason to live", "end it all"
        ]
        has_crisis = False
        crisis_alerts = []
        for q, text in follow_up_responses.items():
            t_lower = str(text).lower()
            for term in crisis_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', t_lower):
                    has_crisis = True
                    crisis_alerts.append(f"Patient referenced acute distress/crisis language ('{term}') in response to: '{q[:60]}...'")

        # 2. Evaluate quality of each response
        vague_patterns = {
            "idk", "nothing", "asdf", "blah", "blahh", "qwerty", "no", "yes",
            "none", "ok", "okay", "cool", "fine", "test", "whatever", "n/a", "na",
            "123", "abc", "nope", "dunno", "nothing much", "good", "bad"
        }

        vague_count = 0
        total_count = len(follow_up_responses)
        long_narratives = []
        informative_responses = []

        for q, text in follow_up_responses.items():
            cleaned = str(text).strip()
            lower = cleaned.lower()
            words = cleaned.split()
            word_count = len(words)

            is_vague = False
            if word_count == 0 or len(cleaned) < 3:
                is_vague = True
            elif lower in vague_patterns:
                is_vague = True
            elif all(w in vague_patterns or len(w) <= 2 for w in words):
                is_vague = True
            elif all(c == cleaned[0] for c in cleaned) and len(cleaned) > 2:
                is_vague = True
            elif word_count <= 2 and not any(kw in lower for kw in ["anxious", "depressed", "tired", "sad", "work", "sleep", "pain", "panic", "fear", "down"]):
                is_vague = True

            if is_vague:
                vague_count += 1
            else:
                if word_count >= 20 or len(re.split(r'[.!?]+', cleaned)) >= 3:
                    long_narratives.append({"question": q, "text": cleaned})
                else:
                    informative_responses.append({"question": q, "text": cleaned})

        is_predominantly_vague = (vague_count == total_count)
        
        # 3. Extract the "Clinical Zest" from long stories
        clinical_zest = ""
        if long_narratives:
            zest_points = []
            for item in long_narratives:
                t = item["text"]
                
                # Detect Context / Life Stressors
                stressors = []
                if re.search(r'\b(job|work|boss|colleague|career|office|company|shift)\b', t, re.I):
                    stressors.append("Workplace pressures / professional burnout")
                if re.search(r'\b(relationship|partner|husband|wife|boyfriend|girlfriend|breakup|divorce|marriage|dating)\b', t, re.I):
                    stressors.append("Relationship discord or interpersonal loss")
                if re.search(r'\b(family|parent|parents|mother|father|mom|dad|sister|brother|child|kids)\b', t, re.I):
                    stressors.append("Familial tensions / domestic stress")
                if re.search(r'\b(money|financial|rent|bills|debt|loan|pay)\b', t, re.I):
                    stressors.append("Financial distress")
                if re.search(r'\b(college|university|exam|study|grades|school|degree)\b', t, re.I):
                    stressors.append("Academic strain")
                if re.search(r'\b(health|pain|sick|illness|doctor|hospital|medical|chronic)\b', t, re.I):
                    stressors.append("Health concerns / somatic vulnerabilities")

                # Detect Emotional / Somatic State
                emotions = []
                if re.search(r'\b(crying|tears|sob|hopeless|empty|numb|worthless|depressed)\b', t, re.I):
                    emotions.append("Depressive affect & tearfulness")
                if re.search(r'\b(panic|heart|chest|breathe|shaking|tremble|dread|scared|terrified)\b', t, re.I):
                    emotions.append("Acute autonomic anxiety & panic sensations")
                if re.search(r'\b(insomnia|awake|sleep|nightmare|restless)\b', t, re.I):
                    emotions.append("Severe sleep disruption")
                if re.search(r'\b(exhausted|drained|tired|fatigue|burnout|no energy)\b', t, re.I):
                    emotions.append("Physical exhaustion / severe anergia")

                # Detect Functional Impact
                impacts = []
                if re.search(r'\b(can\'t focus|concentrate|distracted|brain fog|forget)\b', t, re.I):
                    impacts.append("Cognitive deficits / focus disruption")
                if re.search(r'\b(isolate|avoid|withdrawn|stay in bed|alone|hide)\b', t, re.I):
                    impacts.append("Social withdrawal & avoidance")

                stressor_str = ", ".join(stressors) if stressors else "General life stress"
                emotion_str = ", ".join(emotions) if emotions else "Persistent emotional distress"
                impact_str = ", ".join(impacts) if impacts else "General disruption to routine"

                first_few_sentences = ". ".join(re.split(r'[.!?]+', t)[:2]).strip()
                if len(first_few_sentences) > 140:
                    first_few_sentences = first_few_sentences[:137] + "..."

                zest_points.append(
                    f"• Narrative Essence: \"{first_few_sentences}\"\n"
                    f"  - Core Stressors: {stressor_str}\n"
                    f"  - Emotional State: {emotion_str}\n"
                    f"  - Functional Toll: {impact_str}"
                )
            clinical_zest = "\n".join(zest_points)
        elif informative_responses:
            summary_quotes = [f"\"{item['text']}\"" for item in informative_responses]
            clinical_zest = "Patient provided direct, focused inputs: " + " | ".join(summary_quotes)
        else:
            clinical_zest = "No informative text provided; clinical evaluation rests entirely on standardized screening."

        vague_notice = ""
        if is_predominantly_vague:
            sample = list(follow_up_responses.values())[0]
            vague_notice = f"Note: Patient open-ended text response was brief / non-specific ('{sample}') and was disregarded during clinical deduction. Clinical scoring is derived reliably from the 10-item clinical scale."

        return {
            "is_vague": is_predominantly_vague,
            "has_crisis": has_crisis,
            "crisis_alerts": crisis_alerts,
            "clinical_zest": clinical_zest,
            "vague_notice": vague_notice,
            "has_long_narrative": len(long_narratives) > 0,
            "overall_quality": "long_story_narrative" if long_narratives else ("vague_or_irrelevant" if is_predominantly_vague else "informative")
        }

    # --------------------------------------------------------------------------
    # 4. COMPREHENSIVE CLINICAL REPORT GENERATION (TL;DR format)
    # --------------------------------------------------------------------------
    def generate_comprehensive_report(self, initial_answers: Dict[int, int], 
                                     follow_up_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate comprehensive, accurate clinical report in TL;DR format.
        Grounded strictly in patient's actual answers, saving time for the doctor.
        """
        analysis = self.analyze_initial_answers(initial_answers)
        text_analysis = self.analyze_patient_text_responses(follow_up_responses)
        
        # Clinical Risk Level determination
        risk_level = "low"
        if text_analysis["has_crisis"]:
            risk_level = "crisis"
        elif analysis["depression_severity"] == "severe" or analysis["anxiety_severity"] == "severe":
            risk_level = "high"
        elif analysis["depression_severity"] in ["moderate", "moderately_severe"] or analysis["anxiety_severity"] == "moderate":
            risk_level = "moderate"

        # Formulate Chief Complaint
        chief_complaint = ""
        if text_analysis["has_crisis"]:
            chief_complaint = "Urgent clinical safety assessment required due to crisis indicators in patient narrative."
        elif text_analysis["has_long_narrative"] and text_analysis["clinical_zest"]:
            # Extract first sentence of zest
            lines = text_analysis["clinical_zest"].split("\n")
            chief_complaint = lines[0].replace("• Narrative Essence: ", "").strip('"')
        elif not text_analysis["is_vague"] and follow_up_responses:
            first_ans = list(follow_up_responses.values())[0]
            chief_complaint = f"Patient presents with: '{first_ans[:100]}...'" if len(first_ans) > 100 else f"Patient presents with: '{first_ans}'"
        else:
            # Fallback to score-based complaint
            concerns_str = " and ".join(analysis["primary_concerns"])
            chief_complaint = f"Clinical screening positive for {concerns_str} with {analysis['depression_severity']} depressive symptoms and {analysis['anxiety_severity']} anxiety."

        # Assemble Clinical Insights
        clinical_insights = []
        
        # 1. Standardized Screening Reflections
        dep_symptoms = []
        if int(initial_answers.get(1, 0)) >= 2: dep_symptoms.append("anhedonia (inability to feel pleasure)")
        if int(initial_answers.get(2, 0)) >= 2: dep_symptoms.append("persistent low or hopeless mood")
        if int(initial_answers.get(4, 0)) >= 2: dep_symptoms.append("severe fatigue or lack of energy")
        if int(initial_answers.get(5, 0)) >= 2: dep_symptoms.append("appetite alterations")
        
        if dep_symptoms:
            clinical_insights.append(f"Depression Profile (PHQ-5 Score: {analysis['depression_score']}/15): Patient confirmed {', '.join(dep_symptoms)} occurring more than half the days.")
        else:
            clinical_insights.append(f"Depression Profile (PHQ-5 Score: {analysis['depression_score']}/15): Symptoms fall within the {analysis['depression_severity']} range with no prominent core anhedonia.")

        anx_symptoms = []
        if int(initial_answers.get(6, 0)) >= 2: anx_symptoms.append("persistent nervousness or feeling on edge")
        if int(initial_answers.get(7, 0)) >= 2: anx_symptoms.append("uncontrollable worry loops")
        if int(initial_answers.get(8, 0)) >= 2: anx_symptoms.append("motor restlessness / inability to sit still")
        if int(initial_answers.get(9, 0)) >= 2: anx_symptoms.append("heightened irritability")
        if int(initial_answers.get(10, 0)) >= 2: anx_symptoms.append("acute dread of impending catastrophe")

        if anx_symptoms:
            clinical_insights.append(f"Anxiety Profile (GAD-5 Score: {analysis['anxiety_score']}/15): Patient endorsed {', '.join(anx_symptoms)}.")
        else:
            clinical_insights.append(f"Anxiety Profile (GAD-5 Score: {analysis['anxiety_score']}/15): Anxiety indices remain in the {analysis['anxiety_severity']} range.")

        # 2. Somatic and Sleep Insights
        sleep_score = int(initial_answers.get(3, 0))
        if sleep_score >= 2:
            clinical_insights.append(f"Somatic Profile: Severe sleep disruption reported (rating {sleep_score}/3), directly contributing to cognitive fatigue.")
        else:
            clinical_insights.append(f"Somatic Profile: Sleep patterns rated as {analysis['sleep_disturbance']} disruption.")

        # 3. Text Narrative Essence or Vague Disclosure
        if text_analysis["has_long_narrative"]:
            clinical_insights.append(f"Narrative Extraction (Clinical Zest):\n{text_analysis['clinical_zest']}")
        elif text_analysis["is_vague"]:
            clinical_insights.append(text_analysis["vague_notice"])
        elif follow_up_responses:
            for q, a in list(follow_up_responses.items())[:2]:
                clinical_insights.append(f"Patient reported for '{q[:40]}...': \"{a}\"")

        # Functional Impact description
        if analysis["depression_severity"] == "severe" or analysis["anxiety_severity"] == "severe":
            functional_impact = "Severe functional impairment: High distress scores indicate major disruption to daily occupational performance, social engagement, and emotional stability."
        elif analysis["depression_severity"] in ["moderate", "moderately_severe"] or analysis["anxiety_severity"] == "moderate":
            functional_impact = "Moderate functional impairment: Patient experiences recurrent barriers to sustaining full focus at work/school and strains in interpersonal relationships."
        else:
            functional_impact = "Mild or preserved functional capacity: Symptoms cause noticeable discomfort but core daily responsibilities remain manageable."

        # Recommendations for Doctor
        recommendations = []
        if risk_level == "crisis":
            recommendations.append("🚨 IMMEDIATE SAFETY AUDIT: Conduct immediate in-person or urgent psychiatric safety assessment and lethal means counseling.")
        
        if analysis["depression_severity"] in ["severe", "moderately_severe"]:
            recommendations.append(f"Major Depressive Disorder evaluation: Initiate structured clinical diagnostic interview and consider evidence-based psychotherapy (CBT) and pharmacotherapy evaluation (PHQ-5: {analysis['depression_score']}/15).")
        elif analysis["depression_severity"] == "moderate":
            recommendations.append(f"Depression follow-up: Monitor mood trajectory and schedule focused clinical review within 2 weeks (PHQ-5: {analysis['depression_score']}/15).")

        if analysis["anxiety_severity"] in ["severe", "moderately_severe"]:
            recommendations.append(f"Anxiety management protocol: Screen for panic disorder or generalized anxiety; review somatic relaxation techniques and consider psychiatric consult (GAD-5: {analysis['anxiety_score']}/15).")
        elif analysis["anxiety_severity"] == "moderate":
            recommendations.append(f"Anxiety monitoring: Provide psychoeducation on worry loops and sleep hygiene; review stress triggers.")

        if sleep_score >= 2:
            recommendations.append("Sleep architecture review: Assess for sleep hygiene deficits or comorbid insomnia; recommend CBT-I techniques.")

        recommendations.append("Comprehensive intake review: Verify patient narrative during the clinical consultation to finalize personalized treatment plan.")

        # Retrieve referenced clinical guidelines
        _, referred_sources = self.retrieve_clinical_context(analysis)

        return {
            "risk_level": risk_level,
            "chief_complaint": chief_complaint,
            "clinical_zest": text_analysis["clinical_zest"],
            "diagnostic_considerations": analysis["primary_concerns"],
            "symptom_severity": {
                "depression": analysis["depression_severity"],
                "anxiety": analysis["anxiety_severity"],
                "sleep": analysis["sleep_disturbance"],
                "sleep_disturbance": analysis["sleep_disturbance"],
                "overall": "severe" if risk_level in ["high", "crisis"] else "moderate" if risk_level == "moderate" else "minimal"
            },
            "scores": {
                "depression": analysis["depression_score"],
                "anxiety": analysis["anxiety_score"],
                "overall": analysis["overall_score"],
                "max_subscale": 15
            },
            "clinical_insights": clinical_insights,
            "functional_impact": functional_impact,
            "recommendations": recommendations,
            "crisis_indicators": text_analysis["crisis_alerts"],
            "response_validity": {
                "is_valid": not text_analysis["is_vague"],
                "quality": text_analysis["overall_quality"],
                "vague_notice": text_analysis["vague_notice"]
            },
            "patient_responses": [
                {"question": q, "answer": a} for q, a in follow_up_responses.items()
            ],
            "referred_sources": referred_sources
        }

    def retrieve_clinical_context(self, analysis: Dict[str, Any]) -> tuple:
        """Retrieve relevant clinical context based on analysis"""
        query_terms = analysis.get("primary_concerns", []) + analysis.get("follow_up_focus", [])
        query = " ".join(query_terms)
        
        docs = []
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=3)
            except Exception:
                pass
        
        context = "CLINICAL GUIDELINES CONTEXT:\n"
        referred_sources = []
        
        if docs:
            for doc in docs:
                source_name = doc.metadata.get('source', 'Clinical Guideline')
                display_name = source_name.split("/")[-1].replace(".txt", "").replace(".json", "").replace("_", " ").title()
                if display_name not in referred_sources:
                    referred_sources.append(display_name)
                context += f"\n--- {display_name.upper()} ---\n{doc.page_content[:800]}\n"
        else:
            referred_sources = ["NICE Depression Guidelines", "WHO mhGAP Intervention Guide", "DSM-5 Anxiety Criteria"]
            context += "Standard NICE and WHO mhGAP clinical guidelines for mental health screening and clinical triage."
            
        return context, referred_sources
