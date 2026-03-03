"use client";
import React from "react";
import { motion } from "framer-motion";
import Image from "next/image";

const Abouthero = () => {
  return (
    <section
      className="relative flex flex-col justify-center items-center min-h-screen bg-gradient-to-b from-yellow-50 to-white text-center px-6 pt-24 sm:pt-32"
    >
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
          className="text-4xl sm:text-6xl font-bold text-[#000] drop-shadow-[0_2px_10px_rgba(0,0,0,0.4)]"
        >
          Find Your Inner{" "}
          <span className="bg-yellow-300 px-2 rounded-md">Balance</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-black mt-6 text-lg sm:text-xl max-w-2xl drop-shadow-[0_1px_5px_rgba(0,0,0,0.4)]"
        >
          Explore a new approach to mental well-being — combining AI insights
          with mindfulness to bring peace, clarity, and emotional health.
        </motion.p>

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="mt-10 px-8 py-3 bg-yellow-400 rounded-full text-black font-semibold shadow-md hover:shadow-lg transition"
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
