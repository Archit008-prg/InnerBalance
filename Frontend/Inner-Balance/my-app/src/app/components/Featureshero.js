"use client";
import React from "react";
import { motion } from "framer-motion";
import { AiOutlineSmile, AiOutlineHeart, AiOutlineBulb } from "react-icons/ai";

const Featureshero = () => {
  return (
    <section className="relative flex flex-col justify-center items-center text-center px-6 pt-32 pb-20 w-full min-h-screen overflow-hidden bg-gradient-to-br from-yellow-50 via-white to-yellow-100">
      
      {/* Floating Icons */}
      <motion.div
        className="absolute top-10 left-10 text-yellow-400 text-3xl"
        animate={{ y: [0, 15, 0] }}
        transition={{ duration: 4, repeat: Infinity }}
      >
        <AiOutlineSmile />
      </motion.div>

      <motion.div
        className="absolute bottom-20 right-10 text-pink-400 text-3xl"
        animate={{ y: [0, -15, 0] }}
        transition={{ duration: 3, repeat: Infinity }}
      >
        <AiOutlineHeart />
      </motion.div>

      <motion.div
        className="absolute top-1/2 left-5 text-blue-400 text-3xl"
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
          className="text-4xl sm:text-6xl font-extrabold text-gray-900 drop-shadow-lg"
        >
          Find Your Inner{" "}
          <span className="bg-yellow-300 px-3 py-1 rounded-md shadow-sm">
            Balance
          </span>
        </motion.h1>

        {/* Description */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="text-gray-700 mt-5 text-lg sm:text-xl max-w-2xl leading-relaxed"
        >
          Explore a new approach to mental well-being — combining AI insights with
          mindfulness to bring peace, clarity, and emotional health.
        </motion.p>

        {/* Button */}
        <motion.button
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: "spring", stiffness: 200 }}
          className="mt-10 px-10 py-3 bg-yellow-400 text-black font-semibold rounded-full shadow-lg hover:bg-yellow-500 transition-all"
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


