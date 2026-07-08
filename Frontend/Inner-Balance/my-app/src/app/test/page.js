"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import {
  getQuestions,
  analyzeInitialAssessment,
  generateClinicalReport,
} from "../../lib/api";
import {
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Heart,
} from "lucide-react";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

export default function TestPage() {
  const router = useRouter();

  const [questions, setQuestions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isFollowUpPhase, setIsFollowUpPhase] = useState(false);
  const [initialAnswers, setInitialAnswers] = useState({});
  const [assessmentId, setAssessmentId] = useState(null);
  const [initialAnalysis, setInitialAnalysis] = useState(null);
  const [userRole, setUserRole] = useState(null);

  const questionRef = useRef(null);
  const progressRef = useRef(null);
  const resultRef = useRef(null);
  const pageRef = useRef(null);
  const headerRef = useRef(null);

  useEffect(() => {
    const role = localStorage.getItem("user_role");
    setUserRole(role);
    if (!role) {
      router.push("/login");
      return;
    }
  }, [router]);

  useEffect(() => {
    if (headerRef.current) {
      gsap.fromTo(
        headerRef.current,
        { opacity: 0, y: -30 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

  useEffect(() => {
    const fetchQuestions = async () => {
      setLoading(true);
      setError(null);
      
      // getQuestions() never throws - it always returns fallback questions if API fails
      const data = await getQuestions();
      
      if (data?.questions && Array.isArray(data.questions) && data.questions.length > 0) {
        setQuestions(data.questions);
        
        // Show informational message if using fallback questions (not an error)
        if (data.fallback || data.offline) {
          setError(
            "ℹ️ Offline mode: Backend server is not available. " +
            "Using default questions. Start the Django server for full functionality."
          );
          // Clear message after 8 seconds since it's just informational
          setTimeout(() => setError(null), 8000);
        }
      } else {
        // This should never happen since getQuestions always returns fallback
        setError("No questions available. Please refresh the page.");
      }
      
      setLoading(false);
    };
    
    fetchQuestions();
  }, []);

  useEffect(() => {
    if (questionRef.current && questions.length) {
      gsap.fromTo(
        questionRef.current,
        { opacity: 0, x: 30, scale: 0.95 },
        { opacity: 1, x: 0, scale: 1, duration: 0.5, ease: "power2.out" }
      );
    }
  }, [currentIndex, questions]);

  useEffect(() => {
    if (progressRef.current && questions.length) {
      const progress = ((currentIndex + 1) / questions.length) * 100;
      gsap.to(progressRef.current, {
        width: `${progress}%`,
        duration: 0.6,
        ease: "power2.out",
      });
    }
  }, [currentIndex, questions.length]);

  useEffect(() => {
    if (result && resultRef.current) {
      gsap.fromTo(
        resultRef.current,
        { opacity: 0, y: 50, scale: 0.9 },
        { opacity: 1, y: 0, scale: 1, duration: 0.8, ease: "back.out(1.7)" }
      );
    }
  }, [result]);

  const handleChange = (value) => {
    const q = questions[currentIndex];
    if (!q) return;
    let v = value;
    if (q.question_type === "scale" || q.type === "scale") v = parseInt(value, 10);
    if (q.question_type === "slider" || q.type === "slider") v = parseFloat(value);
    setAnswers((prev) => ({ ...prev, [q.id]: v }));
  };

  const handleNext = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex((i) => i + 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handlePrev = () => {
    if (currentIndex > 0) {
      setCurrentIndex((i) => i - 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handleSubmit = async () => {
    try {
      setSubmitting(true);
      setError(null);

      const formatted = {};
      Object.entries(answers).forEach(([k, v]) => {
        formatted[k] = v;
      });

      if (!isFollowUpPhase) {
        const data = await analyzeInitialAssessment(formatted, assessmentId);
        setInitialAnswers(formatted);
        setAssessmentId(data.assessment_id || Date.now().toString());

        if (data.follow_up_questions?.length) {
          const followUps = data.follow_up_questions.map((q, i) => ({
            id: `followup_${i}_${Date.now()}`,
            text: q,
            question_type: "text",
            type: "text",
            is_follow_up: true,
          }));
          setInitialAnalysis(data);
          setQuestions((prev) => [...prev, ...followUps]);
          setIsFollowUpPhase(true);
          setCurrentIndex(questions.length);
        } else {
          setResult(data);
        }
      } else {
        const followUpResponses = {};
        questions.forEach((q) => {
          if (q.is_follow_up) followUpResponses[q.text] = answers[q.id];
        });

        try {
          const report = await generateClinicalReport(
            assessmentId,
            initialAnswers,
            followUpResponses
          );

          setResult({
            ...report,
            initial_analysis: initialAnalysis,
            assessment_complete: true,
          });
        } catch (reportErr) {
          console.error("Report generation failed:", reportErr);
          setError("Could not generate the full report. Showing offline summary.");

          // Minimal offline summary so the flow completes
          setResult({
            assessment_complete: true,
            initial_analysis: initialAnalysis,
            report: {
              summary:
                "Offline mode: report not generated. Please retry when backend is online.",
              follow_up_responses: followUpResponses,
            },
            fallback: true,
          });
        }
      }
    } catch (err) {
      setError("Submission failed.");
    } finally {
      setSubmitting(false);
    }
  };

  const currentQuestion = questions[currentIndex];
  const progress =
    questions.length > 0
      ? ((currentIndex + 1) / questions.length) * 100
      : 0;
  const currentQuestionAnswered =
    currentQuestion &&
    ((answers[currentQuestion.id] !== undefined && answers[currentQuestion.id] !== "") ||
     currentQuestion.question_type === "slider" ||
     currentQuestion.type === "slider");

  if (loading) {
    return (
      <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
        <Navbar />
        <div className="flex-grow flex flex-col items-center justify-center">
          <Loader2 size={40} className="animate-spin text-primary-accent mb-4" />
          <p className="text-sm text-text-muted">Loading assessment questions...</p>
        </div>
        <Footer />
      </main>
    );
  }

  return (
    <main
      ref={pageRef}
      className="bg-background text-foreground min-h-screen flex flex-col pt-20 relative overflow-hidden"
    >
      <Navbar />

      {/* Decorative ambient background glows */}
      <div className="absolute top-1/4 left-1/4 w-80 h-80 bg-primary-accent/5 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-teal-500/5 rounded-full blur-[120px] pointer-events-none" />

      {/* Dim glowing green ambience behind the MCQ card, visible only for MCQ questions */}
      <div 
        className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[550px] bg-primary-accent/10 rounded-full blur-[130px] pointer-events-none z-0 transition-opacity duration-700 ease-in-out ${
          currentQuestion && (currentQuestion.question_type === "scale" || currentQuestion.type === "scale") && !result
            ? "opacity-100" 
            : "opacity-0"
        }`} 
      />

      <div className="flex-grow py-10 px-4 z-10 relative">
        <div className="max-w-3xl mx-auto">
          
          {!result && (
            <>
              <div ref={headerRef} className="text-center mb-8">
                <div className="flex justify-center items-center gap-2 mb-3">
                  <span className="bg-primary-accent/10 text-primary-accent p-1.5 rounded-xl">
                    <Heart size={24} className="animate-pulse" />
                  </span>
                  <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight">
                    Clinical Intake Assessment
                  </h1>
                </div>
                <p className="text-xs text-text-muted">Pre-consultation screening evaluation</p>
              </div>

              <div className="mb-6">
                <div className="flex justify-between text-xs mb-2 font-medium">
                  <span>
                    Question {currentIndex + 1} of {questions.length}
                  </span>
                  <span>{Math.round(progress)}% Complete</span>
                </div>
                <div className="h-2.5 bg-slate-900/10 dark:bg-white/10 rounded-full overflow-hidden">
                  <div
                    ref={progressRef}
                    className="h-full bg-primary-accent rounded-full transition-all duration-300"
                    style={{ width: "0%" }}
                  />
                </div>
              </div>

              {currentQuestion && (
                <div
                  ref={questionRef}
                  className="bg-card-bg/60 backdrop-blur-xl border border-card-border rounded-3xl shadow-[var(--card-shadow)] p-6 md:p-8 mb-6 relative overflow-hidden"
                >
                  <div className="mb-4">
                    <h2 className="text-base md:text-lg font-bold text-foreground mb-6">
                      {currentQuestion.text}
                    </h2>

                    {(currentQuestion.question_type === "scale" ||
                      currentQuestion.type === "scale") && (
                      <div className="space-y-3">
                        {[0, 1, 2, 3].map((v) => {
                          const labels = currentQuestion.category === "functioning" || currentQuestion.order === 18
                            ? [
                                "Not difficult at all",
                                "Somewhat difficult",
                                "Very difficult",
                                "Extremely difficult"
                              ]
                            : [
                                "Not at all",
                                "Several days",
                                "More than half the days",
                                "Nearly everyday"
                              ];
                          return (
                            <label
                              key={v}
                              className={`flex items-center gap-3.5 p-4 border rounded-2xl cursor-pointer transition text-xs md:text-sm font-semibold select-none ${
                                answers[currentQuestion.id] === v
                                  ? "bg-primary-accent/10 border-primary-accent/40 text-primary-accent"
                                  : "bg-background/40 border-card-border hover:bg-slate-900/5 text-foreground"
                              }`}
                            >
                              <input
                                type="radio"
                                name={`q_${currentQuestion.id}`}
                                checked={answers[currentQuestion.id] === v}
                                onChange={() => handleChange(v)}
                                className="w-4 h-4 accent-primary-accent"
                              />
                              <span>{labels[v]}</span>
                            </label>
                          );
                        })}
                      </div>
                    )}

                    {(currentQuestion.question_type === "slider" ||
                      currentQuestion.type === "slider") && (
                      <div className="space-y-12 py-8 select-none">
                        <div className="relative pt-6 pb-8">
                          {/* Track Background */}
                          <div className="absolute top-1/2 left-0 right-0 h-1 bg-slate-900/10 dark:bg-white/10 -translate-y-1/2 rounded-full pointer-events-none" />
                          
                          {/* Track Active Fill */}
                          <div 
                            className="absolute top-1/2 left-0 h-1 bg-primary-accent -translate-y-1/2 rounded-full pointer-events-none transition-all duration-300"
                            style={{ 
                              width: `${(( (answers[currentQuestion.id] !== undefined ? parseFloat(answers[currentQuestion.id]) : 3.0) - 1) / 4) * 100}%` 
                            }}
                          />

                          {/* Track Tick Circles */}
                          <div className="relative flex justify-between">
                            {[1, 2, 3, 4, 5].map((step) => {
                              const val = answers[currentQuestion.id] !== undefined ? parseFloat(answers[currentQuestion.id]) : 3.0;
                              const isSelected = Math.abs(val - step) < 0.1;
                              const isPassed = step <= val;
                              
                              const labels = {
                                1: "Minimal",
                                2: "Mild",
                                3: "Moderate",
                                4: "Severe",
                                5: "Extreme"
                              };

                              return (
                                <button
                                  key={step}
                                  type="button"
                                  onClick={() => handleChange(step)}
                                  className="relative z-10 flex flex-col items-center focus:outline-none cursor-pointer group"
                                >
                                  {/* Step Circle */}
                                  <div 
                                    className={`w-6 h-6 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                                      isSelected 
                                        ? "bg-slate-900 dark:bg-black border-primary-accent scale-110 shadow-[0_0_12px_rgba(20,250,150,0.5)]" 
                                        : isPassed
                                          ? "bg-primary-accent border-primary-accent"
                                          : "bg-background border-card-border hover:border-primary-accent/60"
                                    }`}
                                  >
                                    {isSelected && (
                                      <div className="w-2.5 h-2.5 bg-primary-accent rounded-full animate-pulse" />
                                    )}
                                  </div>

                                  {/* Label Text */}
                                  <span 
                                    className={`absolute top-9 text-[10px] md:text-xs font-bold tracking-tight text-center whitespace-nowrap transition-colors duration-300 ${
                                      isSelected 
                                        ? "text-primary-accent font-extrabold" 
                                        : "text-text-muted group-hover:text-foreground"
                                    }`}
                                  >
                                    {labels[step]}
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    )}

                    {(currentQuestion.question_type === "yesno" ||
                      currentQuestion.type === "yesno") && (
                      <div className="flex gap-4">
                        {["Yes", "No"].map((o) => (
                          <label
                            key={o}
                            className={`flex-1 flex items-center gap-3 p-4 border rounded-2xl cursor-pointer transition text-xs md:text-sm font-semibold justify-center ${
                              answers[currentQuestion.id] === o
                                ? "bg-primary-accent/10 border-primary-accent/40 text-primary-accent"
                                : "bg-background/40 border-card-border hover:bg-slate-900/5 text-foreground"
                            }`}
                          >
                            <input
                              type="radio"
                              name={`q_${currentQuestion.id}`}
                              checked={answers[currentQuestion.id] === o}
                              onChange={() => handleChange(o)}
                              className="w-4 h-4 accent-primary-accent"
                            />
                            <span>{o}</span>
                          </label>
                        ))}
                      </div>
                    )}

                    {(currentQuestion.question_type === "text" ||
                      currentQuestion.type === "text") && (
                      <textarea
                        rows="5"
                        className="w-full p-4 border border-card-border bg-card-bg text-foreground rounded-2xl focus:border-primary-accent focus:ring-1 focus:ring-primary-accent outline-none transition-all duration-200 placeholder-text-muted text-xs md:text-sm font-sans"
                        value={answers[currentQuestion.id] || ""}
                        onChange={(e) => handleChange(e.target.value)}
                        placeholder="Please type your response here..."
                      />
                    )}
                  </div>
                </div>
              )}

              <div className="flex justify-between items-center">
                <button
                  onClick={handlePrev}
                  disabled={currentIndex === 0}
                  className="px-6 py-2.5 bg-slate-900/5 dark:bg-white/5 border border-card-border hover:bg-slate-900/10 text-foreground disabled:opacity-40 rounded-xl transition text-xs font-semibold flex items-center gap-1 cursor-pointer"
                >
                  <ChevronLeft size={16} />
                  <span>Back</span>
                </button>

                {currentIndex < questions.length - 1 ? (
                  <button
                    onClick={handleNext}
                    disabled={!currentQuestionAnswered}
                    className="px-6 py-2.5 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold rounded-xl transition flex items-center gap-1 cursor-pointer text-xs disabled:opacity-50"
                  >
                    <span>Next</span>
                    <ChevronRight size={16} />
                  </button>
                ) : (
                  <button
                    onClick={handleSubmit}
                    disabled={submitting}
                    className="px-6 py-2.5 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold rounded-xl transition flex items-center gap-1.5 cursor-pointer text-xs disabled:opacity-50"
                  >
                    {submitting ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Submitting...</span>
                      </>
                    ) : (
                      <span>Submit Assessment</span>
                    )}
                  </button>
                )}
              </div>
            </>
          )}

          {/* Info/Warning Message */}
          {error && (
            <div className={`mt-6 p-4 border rounded-2xl flex items-start gap-3 transition-all ${
              error.includes('ℹ️') || error.includes('Offline mode') || error.includes('offline mode')
                ? 'bg-blue-500/10 border-blue-500/20 text-blue-600 dark:text-blue-400'
                : error.includes('⚠️')
                ? 'bg-primary-accent/10 border-primary-accent/20 text-primary-accent'
                : 'bg-red-500/10 border-red-500/20 text-red-600 dark:text-red-400 animate-shake'
            }`}>
              <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                error.includes('ℹ️') || error.includes('Offline mode') || error.includes('offline mode')
                  ? 'text-blue-500'
                  : error.includes('⚠️')
                  ? 'text-primary-accent'
                  : 'text-red-500'
              }`} />
              <div className="flex-1">
                <p className="text-xs font-semibold">
                  {error}
                </p>
                {(error.includes('Django server') || error.includes('Backend server')) && (
                  <div className="mt-3 text-xs space-y-1">
                    <p className="font-semibold text-foreground/80">To start the backend server:</p>
                    <code className="bg-card-bg border border-card-border px-3 py-2 rounded-lg block text-foreground font-mono">
                      cd backend/innerbalance && python manage.py runserver
                    </code>
                    <p className="text-text-muted mt-2">
                      The app will continue working with default questions until the server is available.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {result && (
            <div className="mt-4 space-y-6">
              {userRole === "patient" ? (
                /* Patient Completion View: Only success tick and answer copy */
                <div className="bg-card-bg/60 backdrop-blur-xl border border-card-border rounded-3xl shadow-[var(--card-shadow)] p-8 md:p-10 text-foreground transition-all duration-300 text-center">
                  <div className="flex flex-col items-center pb-6 border-b border-card-border">
                    <div className="inline-flex p-4 bg-primary-accent/10 text-primary-accent rounded-full mb-4 animate-pulse">
                      <CheckCircle2 size={48} />
                    </div>
                    <h2 className="text-2xl font-extrabold tracking-tight">Assessment Submitted Successfully!</h2>
                    <p className="text-xs text-text-muted mt-2 max-w-md">
                      Thank you for completing the clinical screening. Your responses have been sent directly to your clinician for evaluation during your appointment.
                    </p>
                  </div>

                  {/* Copy of responses */}
                  <div className="py-6 text-left">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-text-muted mb-4 border-b border-card-border pb-2">
                      Copy of Your Submitted Responses
                    </h3>
                    <div className="space-y-4">
                      {questions.map((q, idx) => {
                        const answerVal = answers[q.id];
                        let readableAnswer = answerVal;
                        
                        // Map scale options to names if scale type
                        if (q.question_type === 'scale' || q.type === 'scale') {
                          const scaleLabels = q.category === 'functioning' || q.order === 18
                            ? {
                                "0": "Not difficult at all",
                                "1": "Somewhat difficult",
                                "2": "Very difficult",
                                "3": "Extremely difficult"
                              }
                            : {
                                "0": "Not at all",
                                "1": "Several days",
                                "2": "More than half the days",
                                "3": "Nearly everyday"
                              };
                          readableAnswer = scaleLabels[String(answerVal)] || answerVal;
                        } else if (q.question_type === 'slider' || q.type === 'slider') {
                          const val = parseFloat(answerVal !== undefined ? answerVal : 3.0);
                          let label = "Moderate";
                          if (val <= 1.0) label = "Minimal";
                          else if (val < 2.0) label = "Minimal-Mild";
                          else if (val <= 2.0) label = "Mild";
                          else if (val < 3.0) label = "Mild-Moderate";
                          else if (val <= 3.0) label = "Moderate";
                          else if (val < 4.0) label = "Moderate-Severe";
                          else if (val <= 4.0) label = "Severe";
                          else if (val < 5.0) label = "Severe-Extreme";
                          else label = "Extreme";
                          readableAnswer = `${val.toFixed(1)} — ${label}`;
                        }

                        return (
                          <div key={q.id} className="bg-background border border-card-border p-4.5 rounded-2xl text-xs leading-relaxed">
                            <p className="font-bold text-foreground mb-1.5 flex items-start gap-2">
                              <span className="inline-flex w-5 h-5 items-center justify-center rounded-full bg-primary-accent/10 text-primary-accent font-extrabold text-[10px] flex-shrink-0 mt-0.5">
                                {idx + 1}
                              </span>
                              <span>{q.text}</span>
                            </p>
                            <p className="text-text-muted bg-card-bg/40 p-3 rounded-xl border border-card-border italic mt-1 ml-7">
                              &ldquo;{readableAnswer || 'No response provided.'}&rdquo;
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Return Button */}
                  <div className="flex justify-center pt-4">
                    <button
                      onClick={() => router.push("/patient")}
                      className="px-6 py-2.5 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold rounded-xl transition-all shadow-md cursor-pointer text-xs"
                    >
                      Return to Care Dashboard
                    </button>
                  </div>
                </div>
              ) : (
                /* Doctor View: Full Clinical Intake Record */
                <div className="bg-card-bg/60 backdrop-blur-xl border border-card-border rounded-3xl shadow-[var(--card-shadow)] p-8 md:p-10 text-foreground transition-all duration-300">
                  <div className="border-b border-card-border pb-5 mb-5">
                    <h2 className="text-lg font-bold tracking-tight text-foreground">Clinical Intake Summary</h2>
                    <p className="text-xs text-text-muted mt-1 font-semibold">Physician Reference Copy</p>
                  </div>
                  <pre className="text-xs whitespace-pre-wrap bg-background p-5 rounded-2xl border border-card-border text-foreground font-mono leading-relaxed max-h-[400px] overflow-y-auto">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                  {/* Return Button */}
                  <div className="flex justify-center pt-6">
                    <button
                      onClick={() => router.push("/patient")}
                      className="px-6 py-2.5 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold rounded-xl transition cursor-pointer text-xs shadow-md"
                    >
                      Return to Dashboard
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <Footer />
    </main>
  );
}
