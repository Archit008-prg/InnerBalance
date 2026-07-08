"use client";
import React from "react";
import { motion } from "framer-motion";
import Image from "next/image";

const Abouthero = () => {
  return (
    <section
      className="relative flex flex-col justify-center items-center min-h-screen bg-background text-foreground text-center px-6 pt-24 sm:pt-32 transition-colors duration-300 overflow-hidden"
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

      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="z-10 flex flex-col justify-center items-center"
      >
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-4xl sm:text-6xl font-bold text-foreground"
        >
          Find Your Inner{" "}
          <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 px-4 py-1.5 rounded-full text-3xl sm:text-5xl align-middle inline-block">Balance</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-text-muted mt-6 text-lg sm:text-xl max-w-2xl"
        >
          Explore a new approach to mental well-being — combining AI insights
          with mindfulness to bring peace, clarity, and emotional health.
        </motion.p>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="mt-10 px-8 py-3 bg-primary-accent hover:bg-primary-accent/90 rounded-full text-white font-bold shadow-md hover:shadow-lg transition cursor-pointer"
        >
          Get Started
        </motion.button>

        {/* 🌿 GIF below button */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="mt-10"
        >
          <Image
            src="https://www.setindiabiz.com/assets/images/document-required.gif"
            alt="Leaf illustration"
            width={400}
            height={300}
            className="rounded-xl shadow-lg"
          />
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Abouthero;
