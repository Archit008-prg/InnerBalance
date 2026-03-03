"use client";
import React, { useEffect, useRef } from "react";
import Image from "next/image";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Mail, Instagram, Youtube } from "lucide-react";
import logo from "./assets/logo.png";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

const Footer = () => {
  const footerRef = useRef(null);
  const logoRef = useRef(null);
  const textRef = useRef(null);
  const formRef = useRef(null);
  const linksRef = useRef(null);
  const socialRef = useRef(null);
  const leaf1Ref = useRef(null);
  const leaf2Ref = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate logo
      gsap.fromTo(
        logoRef.current,
        { opacity: 0, y: -10 },
        {
          opacity: 1,
          y: 0,
          duration: 0.6,
          scrollTrigger: {
            trigger: logoRef.current,
            start: "top 90%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate text
      gsap.fromTo(
        textRef.current,
        { opacity: 0, y: 10 },
        {
          opacity: 1,
          y: 0,
          duration: 0.6,
          delay: 0.2,
          scrollTrigger: {
            trigger: textRef.current,
            start: "top 90%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate form
      gsap.fromTo(
        formRef.current,
        { opacity: 0, scale: 0.9 },
        {
          opacity: 1,
          scale: 1,
          duration: 0.5,
          delay: 0.4,
          scrollTrigger: {
            trigger: formRef.current,
            start: "top 90%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate links
      gsap.fromTo(
        linksRef.current,
        { opacity: 0, y: 15 },
        {
          opacity: 1,
          y: 0,
          duration: 0.5,
          delay: 0.6,
          scrollTrigger: {
            trigger: linksRef.current,
            start: "top 90%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate social icons
      gsap.fromTo(
        socialRef.current,
        { opacity: 0 },
        {
          opacity: 1,
          duration: 0.5,
          delay: 0.8,
          scrollTrigger: {
            trigger: socialRef.current,
            start: "top 90%",
            toggleActions: "play none none none",
          },
        }
      );

      // Animate leaves
      if (leaf1Ref.current && leaf2Ref.current) {
        gsap.fromTo(
          leaf1Ref.current,
          { opacity: 0, x: 20, rotate: 10 },
          {
            opacity: 1,
            x: 0,
            rotate: 0,
            duration: 0.8,
            scrollTrigger: {
              trigger: leaf1Ref.current,
              start: "top 90%",
              toggleActions: "play none none none",
            },
          }
        );

        gsap.fromTo(
          leaf2Ref.current,
          { opacity: 0, x: -20, rotate: -10 },
          {
            opacity: 1,
            x: 0,
            rotate: 0,
            duration: 0.8,
            scrollTrigger: {
              trigger: leaf2Ref.current,
              start: "top 90%",
              toggleActions: "play none none none",
            },
          }
        );
      }
    }, footerRef);

    return () => ctx.revert();
  }, []);

  return (
    <footer
      ref={footerRef}
      className="bg-white text-black flex flex-col items-center relative overflow-hidden pt-8 sm:pt-12"
    >
      {/* Top black line */}
      <div className="w-full h-[0.5px] bg-black" />

      {/* Logo */}
      <div ref={logoRef} className="mt-6 sm:mt-8">
        <Image
          src={logo}
          alt="Logo"
          width={140}
          height={80}
          className="object-contain w-24 sm:w-32 md:w-36"
        />
      </div>

      {/* Subheading */}
      <div ref={textRef} className="text-center mt-4 sm:mt-6 space-y-2 px-4">
        <p className="text-base sm:text-lg font-medium">
          Enter your email address to consult a doctor
        </p>
        <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
          Get Started on your journey to a <br className="hidden sm:block" />
          <span className="font-semibold">healthier & happier life.</span>
        </p>
      </div>

      {/* Email Subscription Section */}
      <div
        ref={formRef}
        className="flex flex-col sm:flex-row items-center justify-center mt-4 sm:mt-6 gap-3 px-4"
      >
        <input
          type="email"
          placeholder="Enter your email"
          className="px-4 sm:px-5 py-2.5 sm:py-3 w-full sm:w-72 md:w-80 rounded-full border-2 border-gray-300 focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-yellow-400 transition-all text-sm sm:text-base"
        />
        <button
          className="bg-yellow-400 hover:bg-yellow-500 text-black font-semibold px-5 sm:px-6 py-2.5 sm:py-3 rounded-full shadow-md hover:shadow-lg transition-all duration-200 transform hover:scale-105 active:scale-95 text-sm sm:text-base w-full sm:w-auto"
        >
          Get App
        </button>
      </div>

      {/* Links */}
      <div
        ref={linksRef}
        className="flex flex-wrap justify-center gap-4 sm:gap-5 mt-6 sm:mt-8 text-xs sm:text-sm font-medium px-4"
      >
        {[
          "Feature",
          "Benefits",
          "About Us",
          "Terms",
          "Privacy",
          "Contact",
          "Helpline",
        ].map((item, index) => (
          <a
            key={index}
            href="#"
            className="hover:text-yellow-500 transition-colors duration-300"
          >
            {item}
          </a>
        ))}
      </div>

      {/* Social Icons */}
      <div
        ref={socialRef}
        className="flex justify-center gap-6 sm:gap-8 mt-4 sm:mt-6 mb-4 sm:mb-6"
      >
        <a
          href="#"
          className="hover:text-yellow-500 transition-colors transform hover:scale-110"
        >
          <Instagram size={20} className="sm:w-6 sm:h-6" />
        </a>
        <a
          href="mailto:someone@example.com"
          className="hover:text-yellow-500 transition-colors transform hover:scale-110"
        >
          <Mail size={20} className="sm:w-6 sm:h-6" />
        </a>
        <a
          href="#"
          className="hover:text-yellow-500 transition-colors transform hover:scale-110"
        >
          <Youtube size={20} className="sm:w-6 sm:h-6" />
        </a>
      </div>

      {/* Bottom yellow line with leaf sitting ON it */}
      <div className="relative w-full h-[3px] bg-yellow-400">
        <div
          ref={leaf1Ref}
          className="absolute left-2 sm:left-4 bottom-[2px] hidden sm:block"
        >
          <Image
            src="https://png.pngtree.com/png-clipart/20220921/original/pngtree-hand-drawing-leaf-vector-illustration-png-image_8625416.png"
            alt="Leaf Illustration"
            width={100}
            height={100}
            className="object-contain animate-bounce-slow"
            unoptimized
          />
        </div>
        <div
          ref={leaf2Ref}
          className="absolute right-2 sm:right-4 bottom-[2px] hidden sm:block"
        >
          <Image
            src="https://png.pngtree.com/png-clipart/20220921/original/pngtree-hand-drawing-leaf-vector-illustration-png-image_8625416.png"
            alt="Leaf Illustration"
            width={100}
            height={100}
            className="object-contain animate-bounce-slow"
            unoptimized
          />
        </div>
      </div>

      {/* Custom slow bounce animation */}
      <style jsx>{`
        @keyframes bounce-slow {
          0%,
          100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-6px);
          }
        }
        .animate-bounce-slow {
          animation: bounce-slow 3s ease-in-out infinite;
        }
      `}</style>
    </footer>
  );
};

export default Footer;
