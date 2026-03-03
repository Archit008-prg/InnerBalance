"use client";
import React, { useEffect, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import aboutGif from "./assets/about.gif";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const About = () => {
  const sectionRef = useRef(null);
  const imageRef = useRef(null);
  const textRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate image
      gsap.fromTo(
        imageRef.current,
        { opacity: 0, x: -50, scale: 0.95 },
        {
          opacity: 1,
          x: 0,
          scale: 1,
          duration: 0.9,
          ease: "power2.out",
          scrollTrigger: {
            trigger: imageRef.current,
            start: "top 75%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate text
      gsap.fromTo(
        textRef.current,
        { opacity: 0, x: 50 },
        {
          opacity: 1,
          x: 0,
          duration: 0.9,
          delay: 0.2,
          ease: "power2.out",
          scrollTrigger: {
            trigger: textRef.current,
            start: "top 75%",
            toggleActions: "play none none none",
          },
        }
      );

      // Hover effect for image
      const image = imageRef.current?.querySelector("img");
      if (image) {
        image.addEventListener("mouseenter", () => {
          gsap.to(image, { scale: 1.05, duration: 0.3, ease: "power2.out" });
        });
        image.addEventListener("mouseleave", () => {
          gsap.to(image, { scale: 1, duration: 0.3, ease: "power2.out" });
        });
      }
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative w-full min-h-screen flex flex-col md:flex-row items-center justify-center px-4 sm:px-8 py-12 sm:py-16 lg:py-20 bg-gradient-to-br from-[#fafafa] via-[#ffffff] to-[#f6f6f6] overflow-hidden"
    >
      {/* Subtle Background Accent Circle */}
      <div className="absolute w-[400px] sm:w-[500px] lg:w-[600px] h-[400px] sm:h-[500px] lg:h-[600px] rounded-full blur-3xl opacity-30 top-1/4 -left-20 sm:-left-40 bg-yellow-200"></div>

      {/* Left: Image */}
      <div
        ref={imageRef}
        className="flex-1 flex justify-center z-10 mb-8 sm:mb-10 md:mb-0"
      >
        <div className="relative group">
          <Image
            src={aboutGif}
            alt="About Illustration"
            className="rounded-3xl shadow-xl shadow-gray-300 border border-gray-100 w-[250px] sm:w-[320px] md:w-[380px] lg:w-[420px] transition-all duration-500 group-hover:shadow-2xl"
          />
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-tr from-transparent via-transparent opacity-30 pointer-events-none"></div>
        </div>
      </div>

      {/* Right: Text */}
      <div
        ref={textRef}
        className="flex-1 text-center md:text-left z-10 max-w-lg mx-auto md:mx-0 px-4"
      >
        <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-[#1b1b1b] leading-tight tracking-tight">
          Adaptive AI Assessment <br />
          <span className="text-[#47634e]">Framework</span>{" "}
          <span className="bg-yellow-300 px-2 sm:px-3 py-1 rounded-lg text-[#1b1b1b] text-xl sm:text-2xl align-middle">
            (FAQs)
          </span>
        </h2>

        <p className="text-gray-600 mt-4 sm:mt-6 max-w-md mx-auto md:mx-0 text-sm sm:text-base lg:text-lg leading-relaxed">
          Inner Balance introduces a two-stage adaptive assessment protocol supported by a 
          RAG-enhanced clinical reasoning engine. Our system ensures safe, context-aware, and 
          evidence-based questioning grounded in medical knowledge.
        </p>

        {/* Decorative Line + CTA */}
        <div className="h-[3px] bg-yellow-400 mt-4 sm:mt-6 mb-4 sm:mb-6 rounded-full w-24 sm:w-32 mx-auto md:mx-0"></div>

        <Link href="/about">
          <button className="bg-[#47634e] text-white px-5 sm:px-6 py-2.5 sm:py-3 rounded-2xl font-medium shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 active:scale-95 text-sm sm:text-base">
            Learn More
          </button>
        </Link>
      </div>
    </section>
  );
};

export default About;
