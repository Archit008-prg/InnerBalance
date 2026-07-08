"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { 
  Loader2, User, FileText, Activity, BookOpen, AlertCircle, 
  Search, ShieldAlert, LogOut, CheckCircle2, ChevronRight, 
  Printer, Calendar, Mail, Phone, MapPin, Heart
} from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { getDoctorDashboard, logout, deleteAssessment } from "../../lib/api";

export default function DoctorDashboard() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [error, setError] = useState("");
  const [expandedSections, setExpandedSections] = useState({
    severity: true,
    actionPlan: false,
    transcript: false,
    citations: false,
    screening: false
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const handleDelete = async (assessmentId) => {
    if (!assessmentId) return;
    if (!window.confirm("⚠️ WARNING: Are you sure you want to permanently delete this patient assessment record? This will delete all answers and reports associated with this intake session and cannot be undone.")) {
      return;
    }
    
    try {
      setError("");
      await deleteAssessment(assessmentId);
      
      // Update state data to remove the deleted assessment from queue
      const updatedAssessments = assessments.filter(item => item.assessment_id !== assessmentId);
      setData(prev => ({
        ...prev,
        assessments: updatedAssessments,
        count: updatedAssessments.length
      }));
      
      // Select the first remaining intake if any, otherwise select null
      if (selectedId === assessmentId) {
        if (updatedAssessments.length > 0) {
          setSelectedId(updatedAssessments[0].assessment_id);
        } else {
          setSelectedId(null);
        }
      }
    } catch (err) {
      setError(err.message || "Failed to delete assessment record.");
    }
  };

  // Role validation and fetch data
  useEffect(() => {
    const role = localStorage.getItem("user_role");
    if (role !== "doctor") {
      router.push("/login");
      return;
    }

    const loadData = async () => {
      try {
        setLoading(true);
        const res = await getDoctorDashboard();
        setData(res);
        if (res.assessments && res.assessments.length > 0) {
          setSelectedId(res.assessments[0].assessment_id);
        }
      } catch (err) {
        setError(err.message || "Failed to fetch dashboard data. Please log in again.");
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

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return (
      <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
        <Navbar />
        <div className="flex-grow flex flex-col items-center justify-center">
          <Loader2 size={40} className="animate-spin text-primary-accent mb-4" />
          <p className="text-sm text-text-muted">Loading Clinical Intake records...</p>
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
            className="px-6 py-2.5 bg-primary-accent text-slate-900 font-bold rounded-xl cursor-pointer"
          >
            Go to Login
          </button>
        </div>
        <Footer />
      </main>
    );
  }

  const assessments = data?.assessments || [];
  
  // Filter assessments based on search string
  const filteredAssessments = assessments.filter(item => {
    const searchStr = search.toLowerCase();
    return (
      item.patient.name.toLowerCase().includes(searchStr) ||
      item.patient.id.toLowerCase().includes(searchStr) ||
      item.patient.email.toLowerCase().includes(searchStr) ||
      (item.patient.phone && item.patient.phone.toLowerCase().includes(searchStr))
    );
  });

  const selectedItem = assessments.find(item => item.assessment_id === selectedId);
  const report = selectedItem?.report;

  // Helpers for displaying badges
  const getRiskBadge = (level) => {
    switch (level?.toLowerCase()) {
      case "crisis":
        return <span className="bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">Crisis</span>;
      case "high":
        return <span className="bg-orange-500/10 text-orange-600 dark:text-orange-400 border border-orange-500/20 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">High</span>;
      case "moderate":
        return <span className="bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border border-yellow-500/20 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">Moderate</span>;
      case "unreliable":
        return <span className="bg-slate-500/10 text-slate-600 dark:text-slate-400 border border-slate-500/20 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">Unreliable</span>;
      default:
        return <span className="bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">Low</span>;
    }
  };

  const getScoreWidth = (score, max = 27) => {
    const percentage = Math.min(Math.max((score / max) * 100, 0), 100);
    return `${percentage}%`;
  };

  const getSeverityBadgeColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case "severe":
        return "bg-red-500/15 text-red-500 border-red-500/20";
      case "moderately severe":
      case "moderately_severe":
        return "bg-orange-500/15 text-orange-500 border-orange-500/20";
      case "moderate":
        return "bg-yellow-500/15 text-yellow-500 border-yellow-500/20";
      case "mild":
        return "bg-teal-500/15 text-teal-500 border-teal-500/20";
      case "unreliable":
        return "bg-slate-500/15 text-slate-500 border-slate-500/20";
      default:
        return "bg-green-500/15 text-green-500 border-green-500/20";
    }
  };

  return (
    <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
      <Navbar />

      <div className="flex-grow max-w-7xl w-full mx-auto p-4 md:p-6 flex flex-col gap-6">
        
        {/* Dashboard Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-card-bg/60 backdrop-blur-xl border border-card-border p-6 rounded-3xl">
          <div>
            <div className="flex items-center gap-2">
              <span className="bg-primary-accent/10 text-primary-accent p-1.5 rounded-xl">
                <ShieldAlert size={20} />
              </span>
              <h2 className="text-xl md:text-2xl font-bold tracking-tight">Clinician Intake Dashboard</h2>
            </div>
            <p className="text-xs text-text-muted mt-1">Logged in as {data?.doctor?.name} | {data?.doctor?.specialization}</p>
          </div>
          <div className="flex gap-3 w-full md:w-auto">
            <button
              onClick={handlePrint}
              disabled={!selectedId}
              className="flex-grow md:flex-grow-0 px-4 py-2.5 bg-slate-900/5 hover:bg-slate-900/15 border border-card-border rounded-xl text-sm font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <Printer size={16} />
              Print Report
            </button>
            {selectedId && (
              <button
                onClick={() => handleDelete(selectedId)}
                className="flex-grow md:flex-grow-0 px-4 py-2.5 bg-red-500/10 hover:bg-red-500/15 text-red-500 border border-red-500/20 rounded-xl text-sm font-semibold flex items-center justify-center gap-1.5 cursor-pointer no-print"
              >
                <ShieldAlert size={16} />
                Delete Record
              </button>
            )}
            <button
              onClick={handleLogout}
              className="flex-grow md:flex-grow-0 px-4 py-2.5 bg-[#E53E3E]/10 hover:bg-[#E53E3E]/15 text-[#E53E3E] border border-[#E53E3E]/20 rounded-xl text-sm font-semibold flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <LogOut size={16} />
              Logout
            </button>
          </div>
        </div>

        {/* Dashboard Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-grow items-stretch">
          
          {/* Patient Queue Sidebar */}
          <div className="lg:col-span-1 bg-card-bg/60 backdrop-blur-xl border border-card-border rounded-3xl p-5 flex flex-col gap-4 overflow-hidden h-[750px]">
            <h3 className="text-sm font-bold uppercase tracking-wider text-text-muted flex items-center gap-1.5">
              <User size={16} className="text-primary-accent" />
              Patient Intake Queue ({filteredAssessments.length})
            </h3>
            
            {/* Search inputs */}
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-text-muted">
                <Search size={16} />
              </span>
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by name or email..."
                className="w-full pl-9 pr-4 py-2.5 bg-slate-900/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-xs transition"
              />
            </div>

            {/* Assessment list */}
            <div className="flex-grow overflow-y-auto space-y-3 pr-1">
              {filteredAssessments.length === 0 ? (
                <div className="text-center py-10 text-xs text-text-muted">
                  No patient assessments found matching query.
                </div>
              ) : (
                filteredAssessments.map((item) => (
                  <button
                    key={item.assessment_id}
                    onClick={() => setSelectedId(item.assessment_id)}
                    className={`w-full p-4 text-left border rounded-2xl transition flex justify-between items-center cursor-pointer ${
                      selectedId === item.assessment_id
                        ? "bg-primary-accent/10 border-primary-accent/50"
                        : "bg-background/40 border-card-border hover:bg-slate-900/10"
                    }`}
                  >
                    <div className="space-y-1.5">
                      <div className="text-xs font-bold text-foreground">
                        {item.patient.name}
                      </div>
                      <div className="text-[10px] text-text-muted flex items-center gap-1">
                        <Calendar size={10} />
                        {new Date(item.started_at).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric"
                        })}
                      </div>
                      <div className="flex gap-1.5 flex-wrap">
                        {getRiskBadge(item.risk_level)}
                        <span className="bg-slate-900/10 border border-card-border text-[9px] font-semibold px-2 py-0.5 rounded-full">
                          PHQ-9: {item.scores.depression}
                        </span>
                      </div>
                    </div>
                    <ChevronRight size={16} className="text-text-muted" />
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Clinical Report Panel */}
          <div className="lg:col-span-2 bg-card-bg/60 backdrop-blur-xl border border-card-border rounded-3xl p-6 overflow-y-auto h-[750px] printable-section">
            {!selectedItem ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-6">
                <FileText size={48} className="text-text-muted/40 mb-3" />
                <h4 className="text-base font-semibold">No Patient Selected</h4>
                <p className="text-xs text-text-muted mt-1">Select an intake from the queue on the left to audit results.</p>
              </div>
            ) : (
              <div>
                
                {/* Header Profile Copy */}
                <div className="border-b border-card-border pb-5 mb-6">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-2 flex-wrap">
                        <h3 className="text-xl font-bold text-foreground">{selectedItem.patient.name}</h3>
                        {getRiskBadge(selectedItem.risk_level)}
                      </div>
                      <p className="text-xs text-text-muted mt-1">
                        Patient ID: <span className="font-mono">{selectedItem.patient.id}</span> | Completed: {new Date(selectedItem.completed_at || selectedItem.started_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  
                  {/* Patient Demographics Info grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 bg-background/40 border border-card-border rounded-2xl p-4 text-xs">
                    <div className="flex items-center gap-2">
                      <User size={14} className="text-primary-accent" />
                      <div>
                        <div className="text-[10px] text-text-muted uppercase">Gender</div>
                        <div className="font-semibold">{selectedItem.patient.gender === "M" ? "Male" : selectedItem.patient.gender === "F" ? "Female" : "Other"}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Mail size={14} className="text-primary-accent" />
                      <div>
                        <div className="text-[10px] text-text-muted uppercase">Email</div>
                        <div className="font-semibold truncate max-w-[130px]">{selectedItem.patient.email}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Phone size={14} className="text-primary-accent" />
                      <div>
                        <div className="text-[10px] text-text-muted uppercase">Phone</div>
                        <div className="font-semibold">{selectedItem.patient.phone || "N/A"}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <MapPin size={14} className="text-primary-accent" />
                      <div>
                        <div className="text-[10px] text-text-muted uppercase">Address</div>
                        <div className="font-semibold truncate max-w-[130px]">{selectedItem.patient.address || "N/A"}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Validation Warnings (Off-Topic/Garbage Checks) */}
                {report?.validation && report.validation.low_effort && (
                  <div className="bg-red-500/10 border border-red-500/25 rounded-2xl p-5 mb-6 animate-pulse">
                    <h4 className="text-xs font-bold text-red-600 dark:text-red-400 uppercase tracking-wider flex items-center gap-2 font-bold">
                      <ShieldAlert size={16} />
                      Response Effort Validation Error
                    </h4>
                    <div className="mt-2 text-xs leading-relaxed space-y-2">
                      {report.validation.alerts.map((alert, i) => (
                        <p key={i}>{alert}</p>
                      ))}
                      <p className="text-[10px] text-text-muted italic">
                        The metrics below have been flagged as unreliable. Manual clinical audit is required.
                      </p>
                    </div>
                  </div>
                )}

                {/* Suicide Risk Alerts */}
                {report?.suicide_risk && report.suicide_risk.toLowerCase() !== "low" && (
                  <div className="bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 p-5 rounded-2xl mb-6 flex items-start gap-3">
                    <ShieldAlert size={20} className="flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-wider">Suicidality Warning</h4>
                      <p className="text-xs mt-1 leading-relaxed">
                        Patient reports thoughts of suicide or self-harm. Conduct immediate clinical safety assessment.
                      </p>
                    </div>
                  </div>
                )}

                {/* TL;DR Summary Card */}
                <div className="bg-primary-accent/5 border border-primary-accent/20 rounded-3xl p-5 mb-6 z-10 relative">
                  <h4 className="text-xs font-bold text-primary-accent uppercase tracking-wider mb-3">
                    Intake TL;DR Summary
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                    {/* Chief Complaint */}
                    <div className="md:col-span-1 bg-card-bg/40 border border-card-border p-4 rounded-2xl flex flex-col justify-between">
                      <span className="text-[10px] font-bold text-text-muted uppercase">Chief Complaint</span>
                      <p className="text-xs text-foreground font-semibold italic mt-1.5 leading-relaxed">
                        &ldquo;{report?.chief_complaint || report?.patient_responses?.[0]?.answer || "Initial screening for mental health evaluation."}&rdquo;
                      </p>
                    </div>

                    {/* Risk Level */}
                    <div className="bg-card-bg/40 border border-card-border p-4 rounded-2xl flex flex-col justify-between">
                      <span className="text-[10px] font-bold text-text-muted uppercase">Clinical Risk Level</span>
                      <div className="mt-1.5 flex items-center gap-1.5">
                        {getRiskBadge(selectedItem.risk_level)}
                      </div>
                    </div>

                    {/* Score Overview */}
                    <div className="bg-card-bg/40 border border-card-border p-4 rounded-2xl flex flex-col justify-between text-xs leading-relaxed space-y-1">
                      <span className="text-[10px] font-bold text-text-muted uppercase mb-1">Score Overview</span>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Depression:</span>
                        <span className="font-bold text-foreground">{selectedItem.scores.depression} / 24</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">Anxiety:</span>
                        <span className="font-bold text-foreground">{selectedItem.scores.anxiety} / 21</span>
                      </div>
                      {report?.anxiety_intensity !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-text-muted">Max Distress:</span>
                          <span className="font-bold text-primary-accent">
                            {Math.max(parseFloat(report.anxiety_intensity || 1.0), parseFloat(report.depression_intensity || 1.0)).toFixed(1)} / 5.0
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Collapsible Section 1: Clinical Severity & Distress Breakdown */}
                <div className="border border-card-border rounded-2xl overflow-hidden mb-4 bg-background/40">
                  <button
                    onClick={() => toggleSection("severity")}
                    className="w-full px-5 py-4 flex justify-between items-center bg-card-bg/25 hover:bg-card-bg/50 transition cursor-pointer text-left"
                  >
                    <div className="flex items-center gap-2">
                      <Activity size={16} className="text-primary-accent" />
                      <span className="text-sm font-bold text-foreground">1. Clinical Severity & Distress Breakdown</span>
                    </div>
                    <span className="text-text-muted text-xs">
                      {expandedSections.severity ? "Collapse" : "Expand"}
                    </span>
                  </button>
                  {expandedSections.severity && (
                    <div className="p-5 border-t border-card-border space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Depression */}
                        <div className="space-y-1">
                          <div className="flex justify-between items-center text-xs">
                            <span className="font-semibold text-foreground">Depression Score (PHQ-8)</span>
                            <span className={`px-2.5 py-0.5 rounded-full text-[10px] border font-bold ${getSeverityBadgeColor(report?.symptom_severity?.depression)}`}>
                              {report?.symptom_severity?.depression || "Low"}
                            </span>
                          </div>
                          <div className="w-full bg-slate-900/10 dark:bg-white/10 border border-card-border h-2.5 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                report?.validation?.low_effort ? "bg-slate-400" : "bg-primary-accent"
                              }`}
                              style={{ width: report?.validation?.low_effort ? "0%" : `${(selectedItem.scores.depression / 24) * 100}%` }}
                            />
                          </div>
                          <div className="text-[10px] text-text-muted text-right">Raw Score: {selectedItem.scores.depression} / 24</div>
                        </div>

                        {/* Anxiety */}
                        <div className="space-y-1">
                          <div className="flex justify-between items-center text-xs">
                            <span className="font-semibold text-foreground">Anxiety Score (GAD-7)</span>
                            <span className={`px-2.5 py-0.5 rounded-full text-[10px] border font-bold ${getSeverityBadgeColor(report?.symptom_severity?.anxiety)}`}>
                              {report?.symptom_severity?.anxiety || "Low"}
                            </span>
                          </div>
                          <div className="w-full bg-slate-900/10 dark:bg-white/10 border border-card-border h-2.5 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                report?.validation?.low_effort ? "bg-slate-400" : "bg-teal-500"
                              }`}
                              style={{ width: report?.validation?.low_effort ? "0%" : `${(selectedItem.scores.anxiety / 21) * 100}%` }}
                            />
                          </div>
                          <div className="text-[10px] text-text-muted text-right">Raw Score: {selectedItem.scores.anxiety} / 21</div>
                        </div>
                      </div>

                      {/* Sliders & Functional Impact Grid */}
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-card-border/50 text-xs">
                        <div className="bg-background p-3 rounded-xl border border-card-border">
                          <div className="text-[10px] text-text-muted uppercase mb-1">Sleep Disturbance</div>
                          <div className="font-bold capitalize text-foreground">{report?.symptom_severity?.sleep_disturbance || "Minimal"}</div>
                        </div>
                        <div className="bg-background p-3 rounded-xl border border-card-border">
                          <div className="text-[10px] text-text-muted uppercase mb-1">Max Distress Intensity</div>
                          <div className="font-bold text-foreground">
                            Anxiety: {report?.anxiety_intensity !== undefined ? parseFloat(report.anxiety_intensity).toFixed(1) : "3.0"}/5.0<br/>
                            Mood: {report?.depression_intensity !== undefined ? parseFloat(report.depression_intensity).toFixed(1) : "3.0"}/5.0
                          </div>
                        </div>
                        <div className="bg-background p-3 rounded-xl border border-card-border">
                          <div className="text-[10px] text-text-muted uppercase mb-1">Daily Functioning Impact</div>
                          <div className="font-bold capitalize text-foreground">{report?.functioning_difficulty || "Not difficult at all"}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Collapsible Section 2: Physician Action Plan & Recommendations */}
                <div className="border border-card-border rounded-2xl overflow-hidden mb-4 bg-background/40">
                  <button
                    onClick={() => toggleSection("actionPlan")}
                    className="w-full px-5 py-4 flex justify-between items-center bg-card-bg/25 hover:bg-card-bg/50 transition cursor-pointer text-left"
                  >
                    <div className="flex items-center gap-2">
                      <CheckCircle2 size={16} className="text-primary-accent" />
                      <span className="text-sm font-bold text-foreground">2. Physician Action Plan & Recommendations</span>
                    </div>
                    <span className="text-text-muted text-xs">
                      {expandedSections.actionPlan ? "Collapse" : "Expand"}
                    </span>
                  </button>
                  {expandedSections.actionPlan && (
                    <div className="p-5 border-t border-card-border text-xs leading-relaxed space-y-4">
                      {/* Clinical Insights */}
                      <div>
                        <h5 className="font-bold text-foreground uppercase tracking-wider text-[10px] mb-2">Clinical Insights</h5>
                        <div className="bg-background border border-card-border rounded-2xl p-4 space-y-2">
                          <p><strong>Functional Impact:</strong> {report?.functional_impact || 'No major impairment indicated.'}</p>
                          {report?.clinical_insights && report.clinical_insights.length > 0 && (
                            <ul className="list-disc pl-4 space-y-1.5">
                              {report.clinical_insights.map((insight, idx) => (
                                <li key={idx}>{insight}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      </div>

                      {/* Recommendations */}
                      <div>
                        <h5 className="font-bold text-foreground uppercase tracking-wider text-[10px] mb-2">Action Plan</h5>
                        <div className="bg-background border border-card-border rounded-2xl p-4">
                          {report?.recommendations && report.recommendations.length > 0 ? (
                            <ul className="list-disc pl-4 space-y-1.5">
                              {report.recommendations.map((rec, idx) => (
                                <li key={idx}>{rec}</li>
                              ))}
                            </ul>
                          ) : (
                            <p className="text-text-muted">No recommendations generated.</p>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Collapsible Section 3: Patient Dialogue Transcripts */}
                <div className="border border-card-border rounded-2xl overflow-hidden mb-4 bg-background/40">
                  <button
                    onClick={() => toggleSection("transcript")}
                    className="w-full px-5 py-4 flex justify-between items-center bg-card-bg/25 hover:bg-card-bg/50 transition cursor-pointer text-left"
                  >
                    <div className="flex items-center gap-2">
                      <FileText size={16} className="text-primary-accent" />
                      <span className="text-sm font-bold text-foreground">3. Patient Written Responses (Adaptive Dialogue)</span>
                    </div>
                    <span className="text-text-muted text-xs">
                      {expandedSections.transcript ? "Collapse" : "Expand"}
                    </span>
                  </button>
                  {expandedSections.transcript && (
                    <div className="p-5 border-t border-card-border space-y-4">
                      {report?.patient_responses && report.patient_responses.length > 0 ? (
                        <div className="space-y-4">
                          {report.patient_responses.map((resp, idx) => (
                            <div key={idx} className="border-b border-card-border last:border-0 pb-3 last:pb-0 text-xs">
                              <p className="font-bold text-foreground mb-1">
                                Q: {resp.question}
                              </p>
                              <p className="text-text-muted italic bg-card-bg/35 p-2.5 rounded-lg border border-card-border leading-relaxed">
                                &ldquo;{resp.answer || 'No response provided.'}&rdquo;
                              </p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-text-muted text-xs">No written dialogue responses recorded.</p>
                      )}
                    </div>
                  )}
                </div>

                {/* Collapsible Section 4: Referred Clinical Guidelines & Citations */}
                <div className="border border-card-border rounded-2xl overflow-hidden bg-background/40">
                  <button
                    onClick={() => toggleSection("citations")}
                    className="w-full px-5 py-4 flex justify-between items-center bg-card-bg/25 hover:bg-card-bg/50 transition cursor-pointer text-left"
                  >
                    <div className="flex items-center gap-2">
                      <BookOpen size={16} className="text-primary-accent" />
                      <span className="text-sm font-bold text-foreground">4. Referred Clinical Guidelines & Citations</span>
                    </div>
                    <span className="text-text-muted text-xs">
                      {expandedSections.citations ? "Collapse" : "Expand"}
                    </span>
                  </button>
                  {expandedSections.citations && (
                    <div className="p-5 border-t border-card-border text-xs leading-relaxed space-y-2">
                      {report?.referred_sources && report.referred_sources.length > 0 ? (
                        <>
                          <p className="text-text-muted mb-3">
                            The clinical intelligence engine queried and referenced the following medical resources and guidelines to analyze this patient&apos;s symptoms:
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {report.referred_sources.map((source, idx) => (
                              <span key={idx} className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 text-[11px] font-semibold px-3 py-1.5 rounded-xl flex items-center gap-1.5">
                                <FileText size={12} />
                                {source}
                              </span>
                            ))}
                          </div>
                        </>
                      ) : (
                        <p className="text-text-muted">No reference materials cited.</p>
                      )}
                    </div>
                  )}
                </div>

                {/* Collapsible Section 5: Patient Screening Responses */}
                <div className="border border-card-border rounded-2xl overflow-hidden mt-4 bg-background/40">
                  <button
                    onClick={() => toggleSection("screening")}
                    className="w-full px-5 py-4 flex justify-between items-center bg-card-bg/25 hover:bg-card-bg/50 transition cursor-pointer text-left"
                  >
                    <div className="flex items-center gap-2">
                      <FileText size={16} className="text-primary-accent" />
                      <span className="text-sm font-bold text-foreground">5. Patient Screening Responses (Initial 18 Questions)</span>
                    </div>
                    <span className="text-text-muted text-xs">
                      {expandedSections.screening ? "Collapse" : "Expand"}
                    </span>
                  </button>
                  {expandedSections.screening && (
                    <div className="p-5 border-t border-card-border space-y-4">
                      {selectedItem.answers && selectedItem.answers.filter(a => !a.is_follow_up).length > 0 ? (
                        <div className="space-y-3">
                          {selectedItem.answers.filter(a => !a.is_follow_up).map((ans, idx) => {
                            let readableAnswer = ans.response;
                            
                            // Map scale choices
                            if (ans.question_text.includes("how difficult have these problems made it") || ans.question_text.includes("If you checked off any problems")) {
                              const labels = ["Not difficult at all", "Somewhat difficult", "Very difficult", "Extremely difficult"];
                              readableAnswer = labels[parseInt(ans.response)] || ans.response;
                            } else if (!ans.question_text.includes("scale of 1.0 to 5.0")) {
                              const labels = ["Not at all", "Several days", "More than half the days", "Nearly everyday"];
                              readableAnswer = labels[parseInt(ans.response)] || ans.response;
                            } else {
                              // Slider response
                              const val = parseFloat(ans.response);
                              let label = "Moderate";
                              if (val <= 1.0) label = "Minimal";
                              else if (val <= 2.0) label = "Mild";
                              else if (val <= 3.0) label = "Moderate";
                              else if (val <= 4.0) label = "Severe";
                              else label = "Extreme";
                              readableAnswer = `${val.toFixed(1)} — ${label}`;
                            }

                            return (
                              <div key={idx} className="border-b border-card-border last:border-0 pb-3 last:pb-0 text-xs">
                                <p className="font-bold text-foreground mb-1 flex items-start gap-2">
                                  <span className="inline-flex w-5 h-5 items-center justify-center rounded-full bg-primary-accent/10 text-primary-accent font-extrabold text-[10px] flex-shrink-0 mt-0.5">
                                    {idx + 1}
                                  </span>
                                  <span>{ans.question_text}</span>
                                </p>
                                <p className="text-text-muted italic bg-card-bg/35 p-2 px-3 rounded-lg border border-card-border leading-relaxed ml-7">
                                  &ldquo;{readableAnswer}&rdquo;
                                </p>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="text-text-muted text-xs">No screening answers recorded.</p>
                      )}
                    </div>
                  )}
                </div>

                {/* AI Warning Disclaimer */}
                <div className="mt-8 border-t border-card-border pt-4 flex items-center justify-center gap-2 text-[#E53E3E] text-xs font-bold leading-relaxed no-print">
                  <AlertCircle size={16} className="flex-shrink-0 animate-pulse text-[#E53E3E]" />
                  <span>AI generated report can make mistakes.</span>
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
