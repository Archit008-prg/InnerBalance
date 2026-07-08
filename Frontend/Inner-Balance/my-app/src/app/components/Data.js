"use client";
import React, { useEffect, useRef } from "react";
import Image from "next/image";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import datagif from "./assets/data.gif";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const Data = () => {
  const sectionRef = useRef(null);
  const headingRef = useRef(null);
  const textRef = useRef(null);
  const leftRef = useRef(null);
  const rightRef = useRef(null);

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
        textRef.current,
        { opacity: 0 },
        {
          opacity: 1,
          duration: 0.8,
          delay: 0.2,
          scrollTrigger: {
            trigger: textRef.current,
            start: "top 80%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate left section with stagger for cards
      gsap.fromTo(
        leftRef.current,
        { opacity: 0, x: -50 },
        {
          opacity: 1,
          x: 0,
          duration: 0.8,
          scrollTrigger: {
            trigger: leftRef.current,
            start: "top 75%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate testimonial cards with stagger
      const cards = leftRef.current?.querySelectorAll(".testimonial-card");
      if (cards && cards.length > 0) {
        gsap.fromTo(
          cards,
          { opacity: 0, y: 20 },
          {
            opacity: 1,
            y: 0,
            duration: 0.6,
            stagger: 0.2,
            scrollTrigger: {
              trigger: leftRef.current,
              start: "top 75%",
              toggleActions: "play none none none",
            },
          }
        );
      }

      // Animate right section with scale
      gsap.fromTo(
        rightRef.current,
        { opacity: 0, x: 50, scale: 0.9 },
        {
          opacity: 1,
          x: 0,
          scale: 1,
          duration: 0.8,
          scrollTrigger: {
            trigger: rightRef.current,
            start: "top 75%",
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
      className="relative bg-background text-foreground py-12 sm:py-16 lg:py-20 px-4 sm:px-6 lg:px-16 overflow-hidden transition-colors duration-300"
    >
      {/* Background Decorative Elements */}
      <div className="absolute top-0 left-0 w-32 sm:w-48 lg:w-64 h-32 sm:h-48 lg:h-64 bg-primary-accent-light rounded-full opacity-30 blur-3xl -z-10"></div>
      <div className="absolute bottom-0 right-0 w-40 sm:w-60 lg:w-72 h-40 sm:h-60 lg:h-72 bg-primary-accent-light rounded-full opacity-20 blur-3xl -z-10"></div>

      <div className="max-w-7xl mx-auto text-center">
        {/* Heading */}
        <h2
          ref={headingRef}
          className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold mb-4 sm:mb-6 px-4 text-foreground animate-fade-in"
        >
          What Our Users{" "}
          <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 px-3 py-1 rounded-full text-xl sm:text-2xl align-middle inline-block">
            Say
          </span>
        </h2>

        {/* Subtext */}
        <p
          ref={textRef}
          className="max-w-2xl mx-auto text-text-muted mb-12 sm:mb-16 lg:mb-20 text-sm sm:text-base lg:text-lg leading-relaxed px-4"
        >
          Real feedback from users who have experienced the transformative power of our 
          AI-powered adaptive assessment system. See how Inner Balance helps bridge the gap 
          between standardized screening and personalized mental health evaluation.
        </p>

        {/* Content Section */}
        <div className="flex flex-col-reverse lg:flex-row items-stretch justify-center gap-8 sm:gap-12 lg:gap-20">
          {/* Left Text Section */}
          <div
            ref={leftRef}
            className="w-full lg:w-1/2 text-left bg-card-bg/90 backdrop-blur-lg p-6 sm:p-8 rounded-2xl shadow-md border border-card-border hover:shadow-lg transition-all duration-300 flex flex-col justify-between"
          >
            <p className="text-text-muted text-sm sm:text-base lg:text-lg leading-relaxed mb-6">
              Inner Balance combines evidence-grounded clinical intelligence with adaptive AI questioning 
              to provide evidence-based mental health assessments. Our two-stage protocol ensures 
              personalized, context-aware evaluation grounded in medical knowledge (DSM-5, NICE, WHO mhGAP).
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 items-stretch">
              {/* Testimonial 1 */}
              <div className="testimonial-card p-4 sm:p-5 lg:p-6 bg-card-bg rounded-xl shadow-sm border border-card-border hover:shadow-md transition-all flex flex-col justify-between h-full">
                <p className="text-text-muted italic text-xs sm:text-sm lg:text-base mb-4 leading-relaxed">
                  &ldquo;The adaptive questioning system truly understood my situation. The AI-generated 
                  follow-up questions were incredibly relevant and helped me articulate feelings I 
                  couldn&apos;t express before.&rdquo;
                </p>
                <div>
                  <h4 className="mt-4 font-semibold text-foreground text-sm sm:text-base border-t border-card-border/60 pt-3">
                    — Sarah Johnson
                  </h4>
                  <p className="text-xs sm:text-sm text-text-muted/80 mt-0.5">
                    Mental Health Advocate
                  </p>
                </div>
              </div>

              {/* Testimonial 2 */}
              <div className="testimonial-card p-4 sm:p-5 lg:p-6 bg-card-bg rounded-xl shadow-sm border border-card-border hover:shadow-md transition-all flex flex-col justify-between h-full">
                <p className="text-text-muted italic text-xs sm:text-sm lg:text-base mb-4 leading-relaxed">
                  &ldquo;As someone who struggled with traditional static assessments, Inner Balance&apos;s 
                  evidence-grounded system provided personalized insights that felt truly tailored to my experience. 
                  The clinical grounding gave me confidence in the results.&rdquo;
                </p>
                <div>
                  <h4 className="mt-4 font-semibold text-foreground text-sm sm:text-base border-t border-card-border/60 pt-3">
                    — Michael Chen
                  </h4>
                  <p className="text-xs sm:text-sm text-text-muted/80 mt-0.5">
                    Patient User
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Image Section */}
          <div
            ref={rightRef}
            className="w-full lg:col-span-6 flex justify-center items-center"
          >
            <div className="relative group w-full max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg overflow-hidden rounded-2xl bg-card-bg border border-card-border shadow-md transition-all duration-300">
              <Image
                src={datagif}
                alt="Data Visualization"
                className="relative w-full rounded-2xl"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Data;
