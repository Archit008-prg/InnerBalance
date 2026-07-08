"use client";

import React, { useEffect, useRef } from "react";
import Image from "next/image";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const testimonials = [
  {
    text: "The adaptive questioning system truly understood my situation. The AI-generated follow-up questions were incredibly relevant and helped me articulate feelings I couldn't express before. The clinical grounding gave me confidence.",
    name: "Sarah Johnson",
    role: "Patient User",
    img: "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
    color: "#4BB04F",
  },
  {
    text: "As someone who struggled with traditional static assessments, Inner Balance's evidence-grounded system provided personalized insights that felt truly tailored. The two-stage approach made the process feel less overwhelming.",
    name: "Michael Chen",
    role: "Mental Health Patient",
    img: "https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
    color: "#48407D",
  },
  {
    text: "The comprehensive report I received was incredibly detailed and helped my therapist understand my situation better before our first session. It saved valuable time and made the consultation more productive.",
    name: "Emily Rodriguez",
    role: "Therapy Patient",
    img: "https://images.pexels.com/photos/733872/pexels-photo-733872.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
    color: "#F98531",
  },
  {
    text: "Inner Balance bridges the gap between technology and clinical care. The evidence-based approach, grounded in medical guidelines, made me trust the assessment process. Highly recommend for anyone seeking mental health evaluation.",
    name: "David Park",
    role: "Healthcare Professional",
    img: "https://images.pexels.com/photos/2379005/pexels-photo-2379005.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
    color: "#FFB800",
  },
];

// Reusable testimonial card
const TestimonialCard = ({ t }) => {
  const cardRef = useRef(null);

  useEffect(() => {
    if (cardRef.current) {
      gsap.fromTo(
        cardRef.current,
        { opacity: 0, y: 20 },
        {
          opacity: 1,
          y: 0,
          duration: 0.6,
          scrollTrigger: {
            trigger: cardRef.current,
            start: "top 85%",
            toggleActions: "play none none none",
          },
        }
      );
    }
  }, []);

  return (
    <div
      ref={cardRef}
      className="min-w-[280px] sm:min-w-[320px] max-w-[350px] bg-card-bg border border-card-border rounded-2xl shadow-sm p-5 sm:p-6 flex flex-col justify-between mx-3 sm:mx-4 transition-all duration-300 hover:scale-105 hover:shadow-md"
    >
      {/* Quote icon */}
      <div className="text-4xl mb-4" style={{ color: t.color }}>
        &ldquo;
      </div>

      {/* Testimonial text */}
      <p className="text-text-muted text-xs sm:text-sm leading-relaxed mb-6 break-words whitespace-normal flex-grow">
        {t.text}
      </p>

      {/* User info */}
      <div className="flex items-center gap-3 sm:gap-4 mt-auto">
        <Image
          src={t.img}
          alt={t.name}
          width={48}
          height={48}
          className="w-10 h-10 sm:w-12 sm:h-12 rounded-full object-cover flex-shrink-0"
        />
        <div>
          <p className="text-sm sm:text-base font-semibold text-foreground">
            {t.name}
          </p>
          <p className="text-xs sm:text-sm text-text-muted/80">{t.role}</p>
        </div>
      </div>
    </div>
  );
};

const Testimonials = () => {
  const sectionRef = useRef(null);
  const headingRef = useRef(null);
  const subtitleRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate heading
      gsap.fromTo(
        headingRef.current,
        { opacity: 0, y: -30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          scrollTrigger: {
            trigger: headingRef.current,
            start: "top 80%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate subtitle
      gsap.fromTo(
        subtitleRef.current,
        { opacity: 0 },
        {
          opacity: 1,
          duration: 0.8,
          delay: 0.2,
          scrollTrigger: {
            trigger: subtitleRef.current,
            start: "top 80%",
            toggleActions: "play none none none",
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="testimonials"
      className="font-[Rubik] bg-transparent text-foreground px-4 sm:px-6 py-12 sm:py-16 lg:py-20 overflow-hidden transition-colors duration-300"
    >
      {/* Heading */}
      <h2
        ref={headingRef}
        className="text-3xl sm:text-4xl md:text-5xl font-bold text-center mb-4 sm:mb-6 px-4 text-foreground"
      >
        What Our Users{" "}
        <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 px-3 py-1 rounded-full text-xl sm:text-2xl align-middle inline-block">Say</span>
      </h2>

      <p
        ref={subtitleRef}
        className="max-w-2xl mx-auto text-center text-text-muted mb-12 sm:mb-16 text-sm sm:text-base lg:text-lg leading-relaxed px-4"
      >
        Real feedback from patients and healthcare professionals who have experienced the 
        transformative power of our clinical-guideline guided adaptive assessment system.
      </p>

      {/* Row 1 → left to right */}
      <div className="flex mb-6 sm:mb-10 overflow-hidden">
        <div className="flex animate-scroll">
          {[...testimonials, ...testimonials].map((t, i) => (
            <TestimonialCard key={`row1-${i}`} t={t} />
          ))}
        </div>
      </div>

      {/* Row 2 → right to left */}
      <div className="flex overflow-hidden">
        <div className="flex animate-scroll-reverse">
          {[...testimonials, ...testimonials].map((t, i) => (
            <TestimonialCard key={`row2-${i}`} t={t} />
          ))}
        </div>
      </div>

      <style jsx>{`
        @keyframes scroll {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-50%);
          }
        }

        @keyframes scroll-reverse {
          0% {
            transform: translateX(-50%);
          }
          100% {
            transform: translateX(0);
          }
        }

        .animate-scroll {
          animation: scroll 40s linear infinite;
        }

        .animate-scroll-reverse {
          animation: scroll-reverse 40s linear infinite;
        }

        @media (max-width: 640px) {
          .animate-scroll,
          .animate-scroll-reverse {
            animation-duration: 30s;
          }
        }
      `}</style>
    </section>
  );
};

export default Testimonials;
