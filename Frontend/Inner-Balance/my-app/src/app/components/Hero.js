"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import boardImage from "./assets/board.png";
import greenFocus from "./assets/green_focus.png";
import { Heart, Activity, CheckCircle2, ShieldAlert } from "lucide-react";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const Hero = () => {
  const heroRef = useRef(null);
  const leftColRef = useRef(null);
  const rightColRef = useRef(null);

  // Monitor dark mode class on document element
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setIsDarkMode(document.documentElement.classList.contains("dark"));
      const observer = new MutationObserver(() => {
        setIsDarkMode(document.documentElement.classList.contains("dark"));
      });
      observer.observe(document.documentElement, { attributes: true });
      return () => observer.disconnect();
    }
  }, []);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Fade in left column content elements sequentially
      if (leftColRef.current) {
        gsap.fromTo(
          leftColRef.current.children,
          { opacity: 0, y: 30 },
          { 
            opacity: 1, 
            y: 0, 
            duration: 1, 
            stagger: 0.2, 
            ease: "power3.out",
            delay: 0.2
          }
        );
      }

      // Fade in right column widget with subtle scale
      if (rightColRef.current) {
        gsap.fromTo(
          rightColRef.current,
          { opacity: 0, scale: 0.8 },
          { 
            opacity: 1, 
            scale: 1, 
            duration: 1.2, 
            ease: "back.out(1.2)",
            delay: 0.6
          }
        );
      }
    }, heroRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={heroRef}
      className="relative flex flex-col justify-center items-center px-6 lg:px-12 pt-28 sm:pt-36 pb-16 sm:pb-24 w-full overflow-hidden min-h-screen transition-colors duration-300 z-10"
      style={{
        backgroundImage: `url(${isDarkMode ? greenFocus.src : boardImage.src})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }}
    >
      {/* Background gradients overlays (semi-transparent dark in dark theme, very clear in light theme) */}
      <div className="absolute inset-0 bg-white/0 dark:bg-black/45 z-0 transition-colors duration-300" />
      
      {/* Animated premium organic green glows matching the SerenityAI design */}
      <div className="absolute bottom-10 right-10 w-96 h-96 bg-primary-accent/10 dark:bg-primary-accent/15 rounded-full blur-[130px] animate-pulse z-0 pointer-events-none" />
      <div className="absolute top-10 left-10 w-72 h-72 bg-teal-500/5 dark:bg-teal-500/10 rounded-full blur-[100px] animate-pulse z-0 pointer-events-none" style={{ animationDelay: '1.5s' }} />

      {/* Main Grid Content */}
      <div className="relative z-10 w-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
        
        {/* Left Column: Text Content (Left Aligned for high-end look) */}
        <div ref={leftColRef} className="lg:col-span-7 flex flex-col items-start text-left gap-6 max-w-2xl">
          
          {/* Subtle Tag pill */}
          <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-primary-accent/10 dark:bg-white/5 border border-primary-accent/20 dark:border-white/10 rounded-full text-primary-accent text-[10px] uppercase tracking-wider font-extrabold shadow-sm">
            <Heart size={10} className="fill-current animate-pulse text-primary-accent" />
            Clinical Intake Companion
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-slate-900 dark:text-white leading-[1.1] tracking-tight drop-shadow-sm">
            Reconnecting Minds, <br />
            <span className="text-primary-accent">
              Restoring Balance
            </span>
          </h1>

          <p className="text-slate-700 dark:text-slate-300 text-sm sm:text-base md:text-lg leading-relaxed font-medium">
            An intelligent AI-powered mental health assessment platform that bridges the gap between 
            standardized screening and personalized clinical evaluation. Experience adaptive questioning 
            grounded in medical evidence.
          </p>

          {/* Action buttons row */}
          <div className="flex flex-wrap gap-4 mt-2">
            <Link href="/test">
              <button className="px-8 py-3.5 bg-primary-accent hover:bg-primary-accent/90 dark:bg-white dark:hover:bg-white/95 text-white dark:text-slate-900 font-bold rounded-full shadow-lg hover:shadow-primary-accent/25 dark:hover:shadow-white/20 transition transform hover:scale-105 active:scale-95 text-xs sm:text-sm flex items-center gap-1.5 cursor-pointer">
                Talk to Serenity
                <span>→</span>
              </button>
            </Link>
            
            <Link href="/about">
              <button className="px-6 py-3.5 bg-slate-900/5 hover:bg-slate-900/10 dark:bg-white/5 dark:hover:bg-white/10 border border-card-border text-foreground font-bold rounded-full transition text-xs sm:text-sm cursor-pointer">
                How it works
              </button>
            </Link>
          </div>

          {/* Description Badge at the bottom left */}
          <div className="mt-8 pt-6 border-t border-card-border/60 w-full flex items-start gap-3">
            <div className="p-1.5 bg-primary-accent/10 dark:bg-white/5 text-primary-accent rounded-xl">
              <Activity size={18} />
            </div>
            <div>
              <div className="text-xs font-bold text-foreground">Inner Balance Care</div>
              <p className="text-[11px] text-text-muted mt-0.5 max-w-md leading-relaxed">
                A compassionate intake companion offering personalized assessment tools, structured clinical guidance, and medical evidence tracking.
              </p>
            </div>
          </div>

        </div>

        {/* Right Column: Ambience circular widget */}
        <div ref={rightColRef} className="lg:col-span-5 flex justify-center lg:justify-end items-center relative py-6">
          <div className="relative w-72 h-72 sm:w-80 sm:h-80 flex items-center justify-center">
            
            {/* Outer orbiting spinning ring */}
            <div className="absolute w-[95%] h-[95%] border border-dashed border-slate-900/10 dark:border-white/10 rounded-full animate-[spin_35s_linear_infinite] pointer-events-none" />
            
            {/* Inner orbit boundary ring */}
            <div className="absolute w-[80%] h-[80%] border border-slate-900/5 dark:border-white/5 rounded-full pointer-events-none" />
            
            {/* Center Portrait container */}
            <div className="absolute w-[45%] h-[45%] bg-white dark:bg-zinc-900 p-1.5 border border-slate-900/10 dark:border-white/15 rounded-full overflow-hidden shadow-2xl backdrop-blur-xl z-20 flex items-center justify-center">
              <div className="w-full h-full rounded-full bg-slate-100 dark:bg-zinc-800 overflow-hidden relative flex items-center justify-center text-3xl select-none">
                🧘‍♀️
              </div>
            </div>
            
            {/* Orbiting State Labels with ping lights */}
            {/* Reflective */}
            <div className="absolute -top-2 left-1/4 transform -translate-x-1/2 flex items-center gap-2 bg-white dark:bg-zinc-900/90 border border-slate-900/10 dark:border-white/15 px-3 py-1.5 rounded-full text-[9px] font-extrabold uppercase tracking-wider text-foreground shadow-lg z-30 transition-transform hover:scale-105">
              <span className="w-1.5 h-1.5 bg-primary-accent rounded-full animate-ping" />
              Reflective
            </div>
            
            {/* Balanced */}
            <div className="absolute top-1/2 -right-4 transform -translate-y-1/2 flex items-center gap-2 bg-white dark:bg-zinc-900/90 border border-slate-900/10 dark:border-white/15 px-3 py-1.5 rounded-full text-[9px] font-extrabold uppercase tracking-wider text-foreground shadow-lg z-30 transition-transform hover:scale-105">
              <span className="w-1.5 h-1.5 bg-primary-accent rounded-full animate-ping" />
              Balanced
            </div>
            
            {/* Calm */}
            <div className="absolute -bottom-2 left-1/3 transform -translate-x-1/2 flex items-center gap-2 bg-white dark:bg-zinc-900/90 border border-slate-900/10 dark:border-white/15 px-3 py-1.5 rounded-full text-[9px] font-extrabold uppercase tracking-wider text-foreground shadow-lg z-30 transition-transform hover:scale-105">
              <span className="w-1.5 h-1.5 bg-primary-accent rounded-full animate-ping" />
              Calm
            </div>
            
          </div>
        </div>

      </div>
    </section>
  );
};

export default Hero;
