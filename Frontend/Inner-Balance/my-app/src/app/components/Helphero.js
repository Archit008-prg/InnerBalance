"use client";
import React from "react";
import { motion } from "framer-motion";
import Image from "next/image";

const Helphero = () => {
  return (
    <section className="relative flex flex-col justify-center items-center min-h-screen bg-gradient-to-b from-yellow-50 via-white to-yellow-100 text-center px-6 pt-24 sm:pt-32 overflow-hidden">
      {/* Animated floating background shapes */}
      <motion.div
        className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_top_right,rgba(253,224,71,0.3),transparent_50%)]"
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
          className="text-4xl sm:text-6xl font-extrabold text-[#222] leading-tight"
        >
          Discover Your Path to{" "}
          <span className="bg-gradient-to-r from-yellow-400 to-yellow-500 text-white px-3 py-1 rounded-lg">
            Inner Balance
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="text-gray-700 mt-6 text-lg sm:text-xl leading-relaxed max-w-2xl"
        >
          Experience mindfulness reimagined — where <b>AI-driven insights</b> meet 
          gentle meditation practices. Bring harmony to your thoughts, emotions, 
          and overall well-being.
        </motion.p>

        <motion.button
          whileHover={{ scale: 1.08, backgroundColor: "#facc15" }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="mt-10 px-8 py-3 bg-yellow-400 rounded-full text-black font-semibold shadow-lg hover:shadow-xl transition"
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
          />
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Helphero;
