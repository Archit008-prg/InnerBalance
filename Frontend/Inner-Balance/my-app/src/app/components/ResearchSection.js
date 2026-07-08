"use client";
import React from "react";
import { motion } from "framer-motion";
import { FileText, Brain, LineChart, CheckCircle2 } from "lucide-react";

const ResearchSection = () => {
  const sections = [
    {
      title: "Clinical Intake Workflow (CDDS)",
      icon: <FileText className="w-6 h-6 text-primary-accent" />,
      content: (
        <>
          Inner Balance operates as a dedicated <b>Clinical Decision Support System (CDDS)</b>. 
          Patients undergo a standardized pre-screening process consisting of 10 baseline MCQs, 
          followed by 10 dynamically adaptive, LLM-generated screening questions. This structure 
          is supervised to guarantee clinical relevance.
        </>
      ),
    },
    {
      title: "Preventing LLM Hallucinations",
      icon: <Brain className="w-6 h-6 text-primary-accent" />,
      content: (
        <>
          Unlike generic conversational AI, our engine constrains responses within verified 
          medical knowledge databases (using GAD-7 and PHQ-9 diagnostic criteria). This 
          <b>Clinical Decision Support</b> database mapping prevents clinical hallucinations, 
          ensuring intake metrics are accurate and secure.
        </>
      ),
    },
    {
      title: "Saving Clinician Intake Time",
      icon: <LineChart className="w-6 h-6 text-primary-accent" />,
      content: (
        <>
          Intake summaries are compiled in seconds. Doctors receive immediate structured reports 
          outlining functional impairment metrics, warning flags (such as suicidality indicators), 
          and validation alerts. This reduces administrative overhead and saves valuable consultation time.
        </>
      ),
    },
    {
      title: "Secure Decision Platform",
      icon: <CheckCircle2 className="w-6 h-6 text-primary-accent" />,
      content: (
        <>
          Patient intake records are stored securely in a relational PostgreSQL database. This allows 
          clinicians to recall historical intakes, track treatment progression over time, and ensure 
          compliance with data privacy standards.
        </>
      ),
    },
  ];

  return (
    <section className="relative py-16 px-6 sm:px-16 text-foreground bg-background transition-colors duration-300 overflow-hidden">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="text-center mb-12"
      >
        <motion.h2
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-3xl sm:text-4xl font-bold text-foreground tracking-tight"
        >
          Clinical Decision <span className="bg-primary-accent/15 border border-primary-accent/20 text-primary-accent px-2 py-0.5 rounded-full inline-block">Support</span>
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-text-muted mt-4 text-sm sm:text-base max-w-3xl mx-auto leading-relaxed"
        >
          Guiding patient evaluations with genuine, evidence-based CDDS intelligence. Built to help doctors optimize intake diagnostic tasks safely and efficiently.
        </motion.p>
      </motion.div>

      {/* Sections */}
      <div className="max-w-4xl mx-auto space-y-8">
        {sections.map((section, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: index * 0.2 }}
            viewport={{ once: true }}
            className="flex items-start gap-4"
          >
            <div className="bg-primary-accent/10 p-2 rounded-xl flex-shrink-0">{section.icon}</div>
            <div className="text-sm sm:text-base">
              <h3 className="text-primary-accent font-semibold mb-1">{section.title}</h3>
              <p className="text-text-muted">{section.content}</p>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Action Button */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.8 }}
        viewport={{ once: true }}
        className="mt-12 text-center"
      >
        <a
          href="/features"
          className="inline-flex items-center gap-2 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold px-6 py-3 rounded-full shadow transition-transform hover:scale-105 cursor-pointer"
        >
          <FileText className="w-4 h-4" />
          Explore System Features
        </a>
      </motion.div>
    </section>
  );
};

export default ResearchSection;


