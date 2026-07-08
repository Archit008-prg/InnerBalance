"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { 
  Loader2, User, FileText, Activity, BookOpen, AlertCircle, 
  PlusCircle, LogOut, CheckCircle2, ChevronRight, Calendar, Heart
} from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { getPatientDashboard, logout } from "../../lib/api";

export default function PatientDashboard() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [error, setError] = useState("");

  // Role validation and fetch data
  useEffect(() => {
    const role = localStorage.getItem("user_role");
    if (role !== "patient") {
      router.push("/login");
      return;
    }

    const loadData = async () => {
      try {
        setLoading(true);
        const res = await getPatientDashboard();
        setData(res);
        if (res.assessments && res.assessments.length > 0) {
          setSelectedId(res.assessments[0].assessment_id);
        }
      } catch (err) {
        setError(err.message || "Failed to load dashboard. Please try logging in again.");
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [router]);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  const getRiskBadge = (level) => {
    switch (level?.toLowerCase()) {
      case "crisis":
        return <span className="bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20 text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider">Crisis Alarm</span>;
      case "high":
        return <span className="bg-orange-500/10 text-orange-600 dark:text-orange-400 border border-orange-500/20 text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider">High Severity</span>;
      case "moderate":
        return <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider">Moderate</span>;
      case "unreliable":
        return <span className="bg-slate-500/10 text-slate-600 dark:text-slate-400 border border-slate-500/20 text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider">Invalid Effort</span>;
      default:
        return <span className="bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider">Low Risk</span>;
    }
  };

  if (loading) {
    return (
      <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
        <Navbar />
        <div className="flex-grow flex flex-col items-center justify-center">
          <Loader2 size={40} className="animate-spin text-primary-accent mb-4" />
          <p className="text-sm text-text-muted">Loading Patient dashboard...</p>
        </div>
        <Footer />
      </main>
    );
  }

  if (error) {
    return (
      <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
        <Navbar />
        <div className="flex-grow flex flex-col items-center justify-center p-6 text-center">
          <AlertCircle size={40} className="text-red-500 mb-4" />
          <p className="text-base font-bold mb-4">{error}</p>
          <button
            onClick={() => router.push("/login")}
            className="px-6 py-2.5 bg-primary-accent text-slate-900 font-extrabold rounded-xl cursor-pointer"
          >
            Go to Login
          </button>
        </div>
        <Footer />
      </main>
    );
  }

  const assessments = data?.assessments || [];
  const selectedItem = assessments.find(item => item.assessment_id === selectedId);
  const report = selectedItem?.report;

  return (
    <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
      <Navbar />

      <div className="flex-grow max-w-6xl w-full mx-auto p-4 md:p-6 flex flex-col gap-6 relative overflow-hidden">
        {/* Background glow effects */}
        <div className="absolute top-1/4 left-1/4 w-80 h-80 bg-primary-accent/5 rounded-full blur-[100px] pointer-events-none" />
        
        {/* Portal Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-card-bg/60 backdrop-blur-xl border border-card-border p-6 rounded-3xl z-10">
          <div>
            <div className="flex items-center gap-2">
              <span className="bg-primary-accent/10 text-primary-accent p-1.5 rounded-xl">
                <Heart size={20} />
              </span>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight">Patient Care Portal</h2>
            </div>
            <p className="text-xs text-text-muted mt-1">Welcome back, {data?.patient?.name} | Profile ID: {data?.patient?.id}</p>
          </div>
          <div className="flex gap-3 w-full sm:w-auto">
            <button
              onClick={() => router.push("/test")}
              className="flex-grow sm:flex-grow-0 px-4 py-2.5 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold rounded-xl text-sm flex items-center justify-center gap-1.5 cursor-pointer shadow-md"
            >
              <PlusCircle size={16} />
              Start New Screening
            </button>
            <button
              onClick={handleLogout}
              className="flex-grow sm:flex-grow-0 px-4 py-2.5 bg-red-500/10 hover:bg-red-500/15 text-red-500 border border-red-500/20 rounded-xl text-sm font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <LogOut size={16} />
              Logout
            </button>
          </div>
        </div>

        {/* Dash Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch z-10">
          
          {/* Historical list */}
          <div className="lg:col-span-1 bg-card-bg/60 backdrop-blur-xl border border-card-border rounded-3xl p-5 flex flex-col gap-4 max-h-[600px] overflow-y-auto">
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-muted flex items-center gap-1.5 border-b border-card-border pb-3">
              <Calendar size={16} className="text-primary-accent" />
              Completed Screenings ({assessments.length})
            </h3>
            
            <div className="space-y-3">
              {assessments.length === 0 ? (
                <div className="text-center py-10 text-xs text-text-muted">
                  <p className="mb-4">You have not completed any health screenings yet.</p>
                  <button
                    onClick={() => router.push("/test")}
                    className="px-4 py-2 bg-primary-accent text-slate-900 font-extrabold rounded-xl text-xs cursor-pointer"
                  >
                    Take First Assessment
                  </button>
                </div>
              ) : (
                assessments.map((item) => (
                  <button
                    key={item.assessment_id}
                    onClick={() => setSelectedId(item.assessment_id)}
                    className={`w-full p-4 text-left border rounded-2xl transition flex justify-between items-center cursor-pointer ${
                      selectedId === item.assessment_id
                        ? "bg-primary-accent/10 border-primary-accent/30"
                        : "bg-background/40 border-card-border hover:bg-slate-900/10"
                    }`}
                  >
                    <div className="space-y-1.5">
                      <div className="text-xs font-bold text-foreground">
                        Screening Record
                      </div>
                      <div className="text-[10px] text-text-muted flex items-center gap-1">
                        <Calendar size={10} />
                        {new Date(item.started_at).toLocaleDateString("en-US", {
                          month: "long",
                          day: "numeric",
                          year: "numeric",
                          hour: "2-digit",
                          minute: "2-digit"
                        })}
                      </div>
                      <div className="flex gap-1.5">
                        <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider">Completed</span>
                      </div>
                    </div>
                    <ChevronRight size={16} className="text-text-muted" />
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Details/Report Panel */}
          <div className="lg:col-span-2 bg-card-bg/60 backdrop-blur-xl border border-card-border rounded-3xl p-6 max-h-[600px] overflow-y-auto">
            {!selectedItem ? (
              <div className="h-full flex flex-col items-center justify-center text-center py-10">
                <FileText size={48} className="text-text-muted/40 mb-3" />
                <h4 className="text-base font-semibold">No Screening Selected</h4>
                <p className="text-xs text-text-muted mt-1">Select a screening record on the left to view your submitted response copy.</p>
              </div>
            ) : (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-bold text-foreground">Screening Intake Summary</h3>
                  <p className="text-xs text-text-muted mt-1">
                    Submitted: {new Date(selectedItem.completed_at || selectedItem.started_at).toLocaleString()}
                  </p>
                </div>

                <div className="border-t border-card-border pt-4">
                  <div className="bg-primary-accent/10 border border-primary-accent/20 text-primary-accent p-5 rounded-2xl text-xs leading-relaxed flex items-start gap-3 mb-6">
                    <CheckCircle2 size={24} className="flex-shrink-0 text-primary-accent" />
                    <div>
                      <h4 className="font-extrabold text-sm text-foreground mb-1">Assessment Submitted Successfully</h4>
                      <p className="text-text-muted">
                        Your screening responses have been successfully compiled and securely transferred to your consulting doctor&apos;s portal for clinical review.
                      </p>
                    </div>
                  </div>
                </div>

                {selectedItem.answers && selectedItem.answers.length > 0 && (
                  <div>
                    <h4 className="text-sm font-bold uppercase tracking-wider text-text-muted mb-3 border-b border-card-border pb-2">Your Responses (Copy)</h4>
                    <div className="space-y-4">
                      {selectedItem.answers.map((resp, idx) => (
                        <div key={idx} className="bg-background border border-card-border p-4 rounded-2xl text-xs leading-relaxed">
                          <p className="font-bold text-foreground mb-1.5 flex items-center gap-1.5">
                            <span className="inline-flex w-5 h-5 items-center justify-center rounded-full bg-primary-accent/10 text-primary-accent font-extrabold text-[10px]">
                              {idx + 1}
                            </span>
                            {resp.question_text}
                          </p>
                          <p className="text-text-muted bg-card-bg/40 p-3 rounded-xl border border-card-border italic mt-1 ml-6">
                            &ldquo;{resp.response || 'No response provided.'}&rdquo;
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                <div className="bg-primary-accent/10 border border-primary-accent/20 text-primary-accent p-4.5 rounded-2xl text-xs leading-relaxed flex items-start gap-2.5">
                  <CheckCircle2 size={18} className="flex-shrink-0 mt-0.5" />
                  <div>
                    <strong>Note on Clinical Copies:</strong> Your doctor has access to the full, comprehensive assessment report in their portal, which incorporates diagnostic criteria (DSM-5 / NICE) for your session review.
                  </div>
                </div>

              </div>
            )}
          </div>

        </div>

      </div>

      <Footer />
    </main>
  );
}
