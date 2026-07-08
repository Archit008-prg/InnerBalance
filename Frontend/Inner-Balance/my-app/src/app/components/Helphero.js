"use client";
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Image from "next/image";

const Helphero = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const checkTheme = () => {
      const isDark = document.documentElement.classList.contains("dark");
      setIsDarkMode(isDark);
    };
    checkTheme();

    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  return (
    <section
      className="relative flex flex-col justify-center items-center min-h-screen bg-background text-foreground transition-colors duration-300 text-center px-6 pt-24 sm:pt-32 overflow-hidden"
    >
      {/* Decorative Light Glows (Green Ambience) */}
      <div className="absolute top-16 left-20 w-44 h-44 bg-primary-accent/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-24 right-24 w-56 h-56 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Animated floating background shapes */}
      <motion.div
        className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_top_right,rgba(0,223,129,0.15),transparent_50%)] z-10"
        animate={{ scale: [1, 1.1, 1] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      ></motion.div>

      {/* Main content */}
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="z-10 flex flex-col justify-center items-center max-w-3xl"
      >
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-4xl sm:text-6xl font-extrabold text-foreground leading-tight"
        >
          Discover Your Path to{" "}
          <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 px-4 py-1.5 rounded-full text-3xl sm:text-5xl align-middle inline-block shadow-sm">
            Inner Balance
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="text-text-muted mt-6 text-lg sm:text-xl leading-relaxed max-w-2xl"
        >
          Experience mindfulness reimagined — where <b>AI-driven insights</b> meet 
          gentle meditation practices. Bring harmony to your thoughts, emotions, 
          and overall well-being.
        </motion.p>

        <motion.button
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="mt-10 px-8 py-3 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold rounded-full shadow-lg transition cursor-pointer"
        >
          Begin Your Journey
        </motion.button>

        {/* 🌿 Animated illustration */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="mt-12"
        >
          <Image
            src="https://gifyard.com/wp-content/uploads/2023/04/ezgif.com-crop.gif"
            alt="Meditation Illustration"
            width={420}
            height={320}
            className="rounded-2xl shadow-xl"
            unoptimized
          />
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Helphero;
