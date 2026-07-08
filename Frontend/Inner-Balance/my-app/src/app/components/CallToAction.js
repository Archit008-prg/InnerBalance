"use client";

import React, { useEffect, useRef } from "react";
import Link from "next/link";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const CallToAction = () => {
  const containerRef = useRef(null);
  const titleRef = useRef(null);
  const textRef = useRef(null);
  const buttonRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        [titleRef.current, textRef.current, buttonRef.current],
        { opacity: 0, y: 20 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          stagger: 0.15,
          ease: "power2.out",
          scrollTrigger: {
            trigger: containerRef.current,
            start: "top 85%",
            toggleActions: "play none none none",
          },
        }
      );
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={containerRef}
      className="relative w-full py-16 sm:py-20 lg:py-24 px-6 sm:px-12 lg:px-20 bg-[#030303] overflow-hidden flex items-center justify-center border-t border-white/5"
    >
      {/* Background glow banner: Dark green gradient with radial glow */}
      <div className="absolute inset-0 bg-gradient-to-r from-zinc-950 via-emerald-950/45 to-zinc-950 z-0 opacity-80" />
      <div className="absolute top-1/2 left-1/4 transform -translate-y-1/2 w-80 h-80 bg-emerald-500/10 rounded-full blur-[100px] pointer-events-none z-0" />

      {/* Content wrapper */}
      <div className="relative z-10 w-full max-w-7xl mx-auto flex flex-col lg:flex-row items-start lg:items-center justify-between gap-8 sm:gap-12">
        {/* Left Side: Headline and Tag */}
        <div className="flex flex-col items-start gap-4 max-w-xl">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-primary-accent/10 border border-primary-accent/20 rounded-full text-primary-accent text-[10px] uppercase tracking-wider font-extrabold shadow-sm">
            <span className="w-1.5 h-1.5 bg-primary-accent rounded-full animate-ping" />
            Get Started
          </div>
          <h2
            ref={titleRef}
            className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white leading-tight tracking-tight"
          >
            A space that listens <br />
            whenever you&apos;re ready
          </h2>
        </div>

        {/* Right Side: Paragraph and CTA button */}
        <div className="flex flex-col items-start gap-6 max-w-md">
          <p
            ref={textRef}
            className="text-slate-300 text-xs sm:text-sm md:text-base leading-relaxed font-medium"
          >
            You don&apos;t need perfect words — just a quiet moment to check in with
            yourself. Whether it&apos;s now or later, Inner Balance will be here
            with open arms and no judgment.
          </p>

          <Link href="/test" ref={buttonRef}>
            <button className="px-8 py-3.5 bg-white hover:bg-slate-100 text-slate-900 font-bold rounded-full shadow-2xl hover:shadow-white/20 transition transform hover:scale-105 active:scale-95 text-xs sm:text-sm flex items-center gap-1.5 cursor-pointer">
              Start your first reflection
              <span>→</span>
            </button>
          </Link>
        </div>
      </div>
    </section>
  );
};

export default CallToAction;
