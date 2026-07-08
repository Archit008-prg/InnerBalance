"use client";
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import Image from "next/image";
import contacthero from "./assets/contacthero.gif";

const ContactHero = () => {
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

  const fadeUp = (delay = 0) => ({
    hidden: { opacity: 0, y: 25 },
    visible: { opacity: 1, y: 0, transition: { delay, duration: 0.6, ease: "easeOut" } },
  });

  return (
    <section className="relative w-full min-h-[90vh] flex flex-col justify-center items-center text-center overflow-hidden px-6 sm:px-16 pt-32 pb-20 bg-background text-foreground transition-colors duration-300">
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

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.15 }}
        transition={{ delay: 0.3, duration: 1.8 }}
        className="absolute bottom-24 right-24 w-56 h-56 bg-teal-500/10 rounded-full blur-3xl pointer-events-none"
      ></motion.div>

      <motion.div
        animate={{ y: [0, 10, 0], opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        className="absolute top-1/2 left-10 w-24 h-24 bg-primary-accent/5 rounded-full blur-2xl pointer-events-none"
      ></motion.div>

      {/* Content */}
      <motion.div
        variants={fadeUp(0.2)}
        initial="hidden"
        animate="visible"
        className="relative z-10 flex flex-col justify-center items-center"
      >
        {/* Heading */}
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-4xl sm:text-5xl font-extrabold text-foreground leading-tight"
        >
          Reconnect with Your{" "}
          <motion.span
            animate={{ backgroundColor: ["rgba(0, 223, 129, 0.15)", "rgba(0, 223, 129, 0.35)", "rgba(0, 223, 129, 0.15)"] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="px-4 py-1.5 rounded-full bg-primary-accent/10 border border-primary-accent/20 text-primary-accent inline-block text-3xl sm:text-4xl align-middle"
          >
            Inner Balance
          </motion.span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          variants={fadeUp(0.3)}
          className="text-text-muted mt-6 text-base sm:text-lg max-w-2xl leading-relaxed px-2"
        >
          Harness the power of mindfulness and AI to rediscover calm, focus, and emotional clarity.
        </motion.p>

        {/* CTA Button */}
        <motion.button
          variants={fadeUp(0.5)}
          whileHover={{
            scale: 1.05,
            boxShadow: "0px 0px 15px rgba(0, 223, 129, 0.4)",
          }}
          whileTap={{ scale: 0.95 }}
          className="mt-8 px-8 py-3 bg-primary-accent text-slate-900 font-extrabold rounded-full shadow-md hover:shadow-lg transition-all duration-300 cursor-pointer"
        >
          Start Your Journey
        </motion.button>

        {/* Hero Image */}
        <motion.div variants={fadeUp(0.7)} className="mt-12 flex justify-center w-full">
          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="relative"
          >
            <div className="absolute inset-0 bg-primary-accent/20 blur-3xl opacity-25 rounded-3xl scale-110 pointer-events-none"></div>
            <Image
              src={contacthero}
              alt="Meditative person achieving balance"
              width={380}
              height={320}
              className="relative rounded-3xl shadow-xl object-contain transition-transform duration-500 hover:scale-105"
            />
          </motion.div>
        </motion.div>
      </motion.div>
    </section>
  );
};

export default ContactHero;
