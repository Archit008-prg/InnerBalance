"use client";
import React, { useEffect, useRef } from "react";
import Link from "next/link";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import bgImage from "./assets/bg.jpg";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const Hero = () => {
  const heroRef = useRef(null);
  const titleRef = useRef(null);
  const subtitleRef = useRef(null);
  const buttonRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate title with enhanced split text effect
      gsap.fromTo(
        titleRef.current,
        { opacity: 0, y: -50, scale: 0.9, rotationX: -15 },
        { 
          opacity: 1, 
          y: 0, 
          scale: 1,
          rotationX: 0,
          duration: 1.2, 
          ease: "power3.out",
          delay: 0.2
        }
      );

      // Animate subtitle with enhanced fade and slide
      gsap.fromTo(
        subtitleRef.current,
        { opacity: 0, y: 30, blur: 10 },
        { 
          opacity: 1, 
          y: 0,
          blur: 0,
          duration: 1, 
          delay: 0.5, 
          ease: "power2.out" 
        }
      );

      // Animate button with enhanced bounce and glow
      gsap.fromTo(
        buttonRef.current,
        { opacity: 0, scale: 0.7, y: 30, rotation: -5 },
        { 
          opacity: 1, 
          scale: 1, 
          y: 0,
          rotation: 0,
          duration: 0.8, 
          delay: 0.8, 
          ease: "back.out(2)" 
        }
      );

      // Add premium floating animation to button with glow
      if (buttonRef.current) {
        gsap.to(buttonRef.current, {
          y: -8,
          boxShadow: "0 20px 40px rgba(234, 179, 8, 0.4)",
          duration: 2.5,
          repeat: -1,
          yoyo: true,
          ease: "power1.inOut",
          delay: 2,
        });
      }

      // Enhanced parallax effect on background with zoom
      gsap.to(heroRef.current, {
        backgroundPositionY: "35%",
        backgroundSize: "110%",
        scrollTrigger: {
          trigger: heroRef.current,
          start: "top top",
          end: "bottom top",
          scrub: 1.5,
        },
      });

      // Add subtle shimmer effect to title
      if (titleRef.current) {
        gsap.to(titleRef.current, {
          backgroundPosition: "200% 0",
          duration: 3,
          repeat: -1,
          ease: "none",
        });
      }
    }, heroRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={heroRef}
      className="relative flex flex-col justify-center items-center text-center px-4 sm:px-6 pt-24 sm:pt-32 pb-12 sm:pb-20 w-full overflow-hidden min-h-screen"
      style={{
        backgroundImage: `url(${bgImage.src})`,
        backgroundSize: "cover",
        backgroundPosition: "center 20%",
        backgroundRepeat: "no-repeat",
      }}
    >
      {/* Enhanced overlay with gradient for better text readability */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/5 via-black/10 to-black/20 z-0" />
      
      {/* Animated gradient orbs for premium effect */}
      <div className="absolute top-20 left-10 w-72 h-72 bg-yellow-300/20 rounded-full blur-3xl animate-pulse z-0"></div>
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-yellow-400/15 rounded-full blur-3xl animate-pulse z-0" style={{ animationDelay: '1s' }}></div>

      {/* Content */}
      <div className="relative z-10 flex flex-col justify-center items-center max-w-5xl mx-auto">
        <h1
          ref={titleRef}
          className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold text-[#000] drop-shadow-[0_2px_15px_rgba(255,255,255,0.9)] leading-tight px-4 transform-gpu"
        >
          Reconnecting Minds,{" "}
          <span className="bg-gradient-to-r from-yellow-300 via-yellow-400 to-yellow-300 px-3 sm:px-4 py-2 rounded-lg inline-block shadow-lg transform transition-all hover:scale-105">
            Restoring Balance
          </span>
        </h1>

        <p
          ref={subtitleRef}
          className="text-black mt-4 sm:mt-6 text-base sm:text-lg md:text-xl max-w-2xl mx-auto drop-shadow-[0_1px_5px_rgba(255,255,255,0.8)] px-4 leading-relaxed"
        >
          An intelligent AI-powered mental health assessment platform that bridges the gap between 
          standardized screening and personalized clinical evaluation. Experience adaptive questioning 
          grounded in medical evidence.
        </p>

        <Link href="/test" ref={buttonRef}>
          <button
            className="mt-6 sm:mt-8 px-8 sm:px-10 py-3 sm:py-4 bg-gradient-to-r from-yellow-400 via-yellow-500 to-yellow-400 hover:from-yellow-500 hover:via-yellow-600 hover:to-yellow-500 rounded-full text-black font-bold shadow-2xl hover:shadow-yellow-500/50 transition-all duration-300 transform hover:scale-110 active:scale-95 text-sm sm:text-base group relative overflow-hidden"
          >
            <span className="relative z-10 flex items-center">
              Start Assessment
              <span className="inline-block ml-2 group-hover:translate-x-2 transition-transform duration-300">→</span>
            </span>
            <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></span>
          </button>
        </Link>
      </div>
    </section>
  );
};

export default Hero;
