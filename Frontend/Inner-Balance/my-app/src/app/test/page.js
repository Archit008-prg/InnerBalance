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

  const questionRef = useRef(null);
  const progressRef = useRef(null);
  const resultRef = useRef(null);
  const pageRef = useRef(null);
  const headerRef = useRef(null);

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
    if (q.question_type === "scale") v = parseInt(value, 10);
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
    answers[currentQuestion.id] !== undefined &&
    answers[currentQuestion.id] !== "";

  if (loading) {
    return (
      <main className="min-h-screen flex flex-col">
        <Navbar />
        <div className="flex-grow flex items-center justify-center">
          <Loader2 className="w-12 h-12 animate-spin text-yellow-500" />
        </div>
        <Footer />
      </main>
    );
  }

  return (
    <main
      ref={pageRef}
      className="bg-gradient-to-b from-white via-yellow-50 to-white min-h-screen flex flex-col"
    >
      <Navbar />

      <div className="flex-grow py-10 px-4">
        <div className="max-w-4xl mx-auto">
          <div ref={headerRef} className="text-center mb-10">
            <div className="flex justify-center items-center gap-3 mb-4">
              <Heart className="w-8 h-8 text-yellow-500" />
              <h1 className="text-4xl font-bold">
                AI-Driven Mental Health Assessment
              </h1>
            </div>
          </div>

          <div className="mb-8">
            <div className="flex justify-between text-sm mb-2">
              <span>
                Question {currentIndex + 1} of {questions.length}
              </span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
              <div
                ref={progressRef}
                className="h-full bg-yellow-500"
                style={{ width: "0%" }}
              />
            </div>
          </div>

          {currentQuestion && (
            <div
              ref={questionRef}
              className="bg-white border-2 border-yellow-200 rounded-3xl shadow-xl p-8 mb-8"
            >
              <div className="mb-6">
                <h2 className="text-xl font-bold mb-4">
                  {currentQuestion.text}
                </h2>

                {(currentQuestion.question_type === "scale" ||
                  currentQuestion.type === "scale") && (
                  <div className="space-y-3">
                    {[0, 1, 2, 3].map((v) => (
                      <label
                        key={v}
                        className="flex items-center gap-3 p-4 border rounded-xl cursor-pointer"
                      >
                        <input
                          type="radio"
                          checked={answers[currentQuestion.id] === v}
                          onChange={() => handleChange(v)}
                        />
                        <span>{v}</span>
                      </label>
                    ))}
                  </div>
                )}

                {(currentQuestion.question_type === "yesno" ||
                  currentQuestion.type === "yesno") && (
                  <div className="flex gap-4">
                    {["Yes", "No"].map((o) => (
                      <label
                        key={o}
                        className="flex items-center gap-3 p-4 border rounded-xl cursor-pointer"
                      >
                        <input
                          type="radio"
                          checked={answers[currentQuestion.id] === o}
                          onChange={() => handleChange(o)}
                        />
                        {o}
                      </label>
                    ))}
                  </div>
                )}

                {(currentQuestion.question_type === "text" ||
                  currentQuestion.type === "text") && (
                  <textarea
                    rows="5"
                    className="w-full p-4 border rounded-xl"
                    value={answers[currentQuestion.id] || ""}
                    onChange={(e) => handleChange(e.target.value)}
                  />
                )}
              </div>
            </div>
          )}

          <div className="flex justify-between items-center">
            <button
              onClick={handlePrev}
              disabled={currentIndex === 0}
              className="px-6 py-3 bg-gray-200 rounded-full"
            >
              <ChevronLeft />
            </button>

            {currentIndex < questions.length - 1 ? (
              <button
                onClick={handleNext}
                disabled={!currentQuestionAnswered}
                className="px-6 py-3 bg-yellow-400 rounded-full"
              >
                <ChevronRight />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={submitting}
                className="px-6 py-3 bg-green-500 text-white rounded-full"
              >
                {submitting ? "Submitting..." : "Submit"}
              </button>
            )}
          </div>

          {/* Info/Warning Message */}
          {error && (
            <div className={`mt-6 p-4 border-2 rounded-xl flex items-start gap-3 transition-all ${
              error.includes('ℹ️') || error.includes('Offline mode') || error.includes('offline mode')
                ? 'bg-blue-50 border-blue-300'
                : error.includes('⚠️')
                ? 'bg-yellow-50 border-yellow-300'
                : 'bg-red-50 border-red-200 animate-shake'
            }`}>
              <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                error.includes('ℹ️') || error.includes('Offline mode') || error.includes('offline mode')
                  ? 'text-blue-600'
                  : error.includes('⚠️')
                  ? 'text-yellow-600'
                  : 'text-red-500'
              }`} />
              <div className="flex-1">
                <p className={`text-sm font-medium ${
                  error.includes('ℹ️') || error.includes('Offline mode') || error.includes('offline mode')
                    ? 'text-blue-800'
                    : error.includes('⚠️')
                    ? 'text-yellow-800'
                    : 'text-red-700'
                }`}>
                  {error}
                </p>
                {(error.includes('Django server') || error.includes('Backend server')) && (
                  <div className="mt-3 text-xs text-gray-600 space-y-1">
                    <p className="font-semibold text-gray-700">To start the backend server:</p>
                    <code className="bg-gray-100 px-3 py-2 rounded-lg block text-gray-800 font-mono">
                      cd backend/InnerBalance/backend && python manage.py runserver
                    </code>
                    <p className="text-gray-500 mt-2">
                      The app will continue working with default questions until the server is available.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {result && (
            <div
              ref={resultRef}
              className="mt-10 p-8 bg-yellow-50 border border-yellow-200 rounded-3xl"
            >
              <h2 className="text-2xl font-bold mb-4">
                Assessment Complete
              </h2>
              <pre className="text-sm whitespace-pre-wrap">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>

      <Footer />
    </main>
  );
}
