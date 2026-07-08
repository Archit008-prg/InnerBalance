"use client";
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { AiOutlineSmile, AiOutlineHeart, AiOutlineBulb } from "react-icons/ai";
import greenFocus from "./assets/green_focus.png";
import boardImage from "./assets/board.png";

const Featureshero = () => {
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
      className="relative flex flex-col justify-center items-center text-center px-6 pt-32 pb-20 w-full min-h-screen overflow-hidden bg-background text-foreground transition-colors duration-300"
    >
      {/* Decorative Light Glows (Green Ambience with Heartbeat Pulse) */}
      <motion.div
        animate={{
          scale: [1, 1.12, 0.98, 1.12, 1],
          opacity: [0.15, 0.22, 0.12, 0.22, 0.15]
        }}
        transition={{
          duration: 3.5,
          repeat: Infinity,
          ease: "easeInOut"
        }}
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[550px] bg-primary-accent rounded-full blur-[140px] pointer-events-none z-0"
      />
      <div className="absolute top-16 left-20 w-44 h-44 bg-primary-accent/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-24 right-24 w-56 h-56 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
      
      {/* Floating Icons */}
      <motion.div
        className="absolute top-10 left-10 text-primary-accent text-3xl z-10"
        animate={{ y: [0, 15, 0] }}
        transition={{ duration: 4, repeat: Infinity }}
      >
        <AiOutlineSmile />
      </motion.div>

      <motion.div
        className="absolute bottom-20 right-10 text-primary-accent text-3xl z-10"
        animate={{ y: [0, -15, 0] }}
        transition={{ duration: 3, repeat: Infinity }}
      >
        <AiOutlineHeart />
      </motion.div>

      <motion.div
        className="absolute top-1/2 left-5 text-primary-accent text-3xl z-10"
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 5, repeat: Infinity }}
      >
        <AiOutlineBulb />
      </motion.div>

      {/* Hero Content */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="relative z-10 flex flex-col justify-center items-center -mt-10"
      >
        {/* Heading */}
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-4xl sm:text-6xl font-extrabold text-foreground"
        >
          Find Your Inner{" "}
          <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 px-4 py-1.5 rounded-full text-3xl sm:text-5xl align-middle inline-block shadow-sm">
            Balance
          </span>
        </motion.h1>

        {/* Description */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-text-muted mt-5 text-lg sm:text-xl max-w-2xl leading-relaxed"
        >
          Explore a new approach to mental well-being — combining AI insights with
          mindfulness to bring peace, clarity, and emotional health.
        </motion.p>

        {/* Button */}
        <motion.button
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="mt-10 px-10 py-3 bg-primary-accent hover:bg-primary-accent/90 text-slate-900 font-extrabold rounded-full shadow-lg transition-all cursor-pointer"
        >
          Get Started
        </motion.button>

        {/* Floating Image */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          className="mt-12 w-full max-w-sm rounded-2xl overflow-hidden shadow-2xl"
        >
          <motion.img
            src="https://media1.tenor.com/m/l8nFTmR3bwoAAAAC/brain-out-brain.gif"
            alt="Mindfulness illustration"
            className="rounded-2xl w-full h-auto object-cover"
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          />
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Featureshero;


