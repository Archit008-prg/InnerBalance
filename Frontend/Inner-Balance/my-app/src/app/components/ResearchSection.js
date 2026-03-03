"use client";
import React from "react";
import { motion } from "framer-motion";
import { FileText, Brain, LineChart, CheckCircle2 } from "lucide-react";

const ResearchSection = () => {
  const sections = [
    {
      title: "Abstract",
      icon: <FileText className="w-6 h-6 text-yellow-500" />,
      content: (
        <>
          This paper presents the design and development of an{" "}
          <b>Automated Number Plate Recognition (ANPR)</b> system for
          automating vehicle entry and exit operations. It uses <b>YOLOv8</b>{" "}
          for object detection and <b>EasyOCR</b> for optical character
          recognition, providing robust performance under diverse conditions.
        </>
      ),
    },
    {
      title: "Methodology",
      icon: <Brain className="w-6 h-6 text-yellow-500" />,
      content: (
        <>
          The system starts with live video input. Detected number plates are
          localized using <b>YOLOv8</b>, cropped, and passed to{" "}
          <b>EasyOCR</b> for character extraction. Data is verified and stored
          in <b>MongoDB</b>, while a <b>Streamlit</b> dashboard allows real-time
          monitoring.
        </>
      ),
    },
    {
      title: "Results and Discussion",
      icon: <LineChart className="w-6 h-6 text-yellow-500" />,
      content: (
        <>
          The framework achieves an average detection accuracy of <b>96.7%</b>{" "}
          and recognition accuracy of <b>94.3%</b>. It is resilient against
          motion blur, angular distortion, and low-light conditions, and shows
          a <b>15–20%</b> improvement over conventional ANPR systems.
        </>
      ),
    },
    {
      title: "Conclusion and Future Work",
      icon: <CheckCircle2 className="w-6 h-6 text-yellow-500" />,
      content: (
        <>
          The ANPR system automates vehicle identification and database
          management with high accuracy. Future work includes deployment on{" "}
          <b>edge devices</b> and integration with <b>IoT-based gate control</b>{" "}
          for fully autonomous entry.
        </>
      ),
    },
  ];

  return (
    <section className="relative py-16 px-6 sm:px-16 text-gray-900 overflow-hidden">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="text-center mb-12"
      >
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-3xl sm:text-4xl font-bold text-[#000] tracking-tight"
        >
          Research on <span className="bg-yellow-300 px-1 rounded">Topic</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-gray-800 mt-4 text-sm sm:text-base max-w-3xl mx-auto leading-relaxed"
        >
          IEEE-formatted study on integrating YOLOv8 and EasyOCR for real-time
          number plate detection and recognition to improve efficiency,
          security, and automation in vehicle management systems.
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
            <div>{section.icon}</div>
            <div className="text-sm sm:text-base">
              <h3 className="text-yellow-600 font-semibold mb-1">{section.title}</h3>
              <p className="text-gray-700">{section.content}</p>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Download Button */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.8 }}
        viewport={{ once: true }}
        className="mt-12 text-center"
      >
        <a
          href="/research-paper.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 bg-yellow-400 hover:bg-yellow-500 text-black font-semibold px-6 py-3 rounded-full shadow transition-transform hover:scale-105"
        >
          <FileText className="w-4 h-4" />
          View Full Research Paper
        </a>
      </motion.div>
    </section>
  );
};

export default ResearchSection;


