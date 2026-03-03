"use client";
import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Plus, Minus } from "lucide-react";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const faqs = [
  {
    question: "What is Inner Balance?",
    answer:
      "Inner Balance is an intelligent digital mental health assessment platform that transforms the pre-consultation stage of psychological evaluation. It combines AI-powered adaptive questioning with RAG-enhanced clinical reasoning, grounded in medical evidence (DSM-5, NICE, WHO mhGAP).",
  },
  {
    question: "How does the two-stage adaptive assessment work?",
    answer:
      "Stage 1 involves completing a short set of validated clinical questions to establish a baseline. Stage 2 uses AI-powered personalization where the system dynamically generates follow-up questions using LLM reasoning, probing deeper into symptoms and risk indicators based on your initial responses.",
  },
  {
    question: "What is RAG-enhanced clinical intelligence?",
    answer:
      "RAG (Retrieval-Augmented Generation) ensures our AI system is grounded in medical knowledge. A medical knowledge base stored in ChromaDB provides factual and clinical grounding. Our specialized medical LLM (Meditron-7B) retrieves contextually relevant knowledge for safe, evidence-based, and ethical adaptive questioning.",
  },
  {
    question: "How accurate are the assessments?",
    answer:
      "Our assessments are grounded in validated clinical tools and medical guidelines. The RAG system ensures all AI-generated questions and insights are based on evidence from DSM-5, NICE guidelines, and WHO mhGAP protocols. However, these assessments are for informational purposes and should complement, not replace, professional clinical evaluation.",
  },
  {
    question: "What kind of output do I receive?",
    answer:
      "You receive a comprehensive, structured report integrating symptom summary, risk factors, and clinical insights. The report is designed to be seamlessly integrable with Electronic Health Records (EHR) and clinical workflows, providing clinicians with valuable pre-consultation data.",
  },
  {
    question: "Is my data secure and private?",
    answer:
      "Yes, we take your privacy seriously. All data is encrypted and stored securely. We comply with healthcare privacy regulations (HIPAA-compliant practices) and never share your information without your explicit consent. Your assessment data is used solely for generating your personalized report.",
  },
];

const FAQItem = ({ faq, isOpen, onToggle }) => {
  const itemRef = useRef(null);

  useEffect(() => {
    if (itemRef.current) {
      if (isOpen) {
        gsap.to(itemRef.current, {
          height: "auto",
          opacity: 1,
          duration: 0.3,
          ease: "power2.out",
        });
      } else {
        gsap.to(itemRef.current, {
          height: 0,
          opacity: 0,
          duration: 0.3,
          ease: "power2.in",
        });
      }
    }
  }, [isOpen]);

  return (
    <div className="border-b-2 border-gray-200 pb-3 sm:pb-4">
      <button
        onClick={onToggle}
        className="w-full flex justify-between items-center text-left text-base sm:text-lg font-medium hover:text-yellow-500 transition-colors gap-4"
      >
        <span className="flex-1">{faq.question}</span>
        {isOpen ? (
          <Minus size={20} className="flex-shrink-0" />
        ) : (
          <Plus size={20} className="flex-shrink-0" />
        )}
      </button>

      <div
        ref={itemRef}
        className="overflow-hidden"
        style={{ height: 0, opacity: 0 }}
      >
        <p className="text-gray-700 mt-2 sm:mt-3 text-sm sm:text-base leading-relaxed">
          {faq.answer}
        </p>
      </div>
    </div>
  );
};

const FAQ = () => {
  const [openIndex, setOpenIndex] = useState(null);
  const sectionRef = useRef(null);
  const leftRef = useRef(null);
  const rightRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate left section
      gsap.fromTo(
        leftRef.current,
        { opacity: 0, x: -30 },
        {
          opacity: 1,
          x: 0,
          duration: 0.6,
          scrollTrigger: {
            trigger: leftRef.current,
            start: "top 80%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate right section
      gsap.fromTo(
        rightRef.current,
        { opacity: 0, x: 30 },
        {
          opacity: 1,
          x: 0,
          duration: 0.6,
          scrollTrigger: {
            trigger: rightRef.current,
            start: "top 80%",
            toggleActions: "play none none none",
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  const handleToggle = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section
      ref={sectionRef}
      className="bg-white py-12 sm:py-16 lg:py-20 px-4 sm:px-6 lg:px-16 text-black"
    >
      <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-8 sm:gap-12 items-start">
        {/* Left Side */}
        <div ref={leftRef}>
          {/* Small Animated Icon */}
          <div className="flex items-center gap-2 mb-3">
            <Image
              src="https://media.tenor.com/r2l6ol9HRqIAAAAi/question-mark-question.gif"
              alt="Question Icon"
              width={55}
              height={55}
              className="rounded-full"
              unoptimized
            />
          </div>

          {/* Heading */}
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold leading-tight mb-4">
            Frequently Asked <br />
            Questions{" "}
            <span className="bg-yellow-300 px-2 py-1 rounded-md">(FAQs)</span>
          </h2>

          {/* Description */}
          <p className="text-gray-700 mt-4 max-w-md text-sm sm:text-base">
            Find everything you need to know about Inner Balance, its features,
            and how it helps you on your wellness journey.
          </p>

          {/* Search box */}
          <div className="mt-6">
            <input
              type="text"
              placeholder="Type your question here..."
              className="w-full sm:w-80 border-2 border-gray-300 rounded-full px-4 sm:px-5 py-2.5 sm:py-3 text-gray-600 focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-yellow-400 transition-all text-sm sm:text-base"
            />
          </div>
        </div>

        {/* Right Side - FAQ Accordion */}
        <div ref={rightRef} className="space-y-3 sm:space-y-4">
          {faqs.map((faq, index) => {
            const isOpen = openIndex === index;

            return (
              <FAQItem
                key={index}
                faq={faq}
                isOpen={isOpen}
                onToggle={() => handleToggle(index)}
              />
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default FAQ;
