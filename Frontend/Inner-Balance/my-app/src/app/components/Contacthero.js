"use client";
import React from "react";
import { motion } from "framer-motion";
import Image from "next/image";
import contacthero from "./assets/contacthero.gif";

const ContactHero = () => {
  const fadeUp = (delay = 0) => ({
    hidden: { opacity: 0, y: 25 },
    visible: { opacity: 1, y: 0, transition: { delay, duration: 0.6, ease: "easeOut" } },
  });

  return (
    <section className="relative w-full min-h-[90vh] flex flex-col justify-center items-center text-center overflow-hidden px-6 sm:px-16 pt-32 pb-20 bg-gradient-to-b from-yellow-50 via-white to-gray-100">
      {/* Decorative Elements */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.2 }}
        transition={{ duration: 1.5 }}
        className="absolute top-16 left-20 w-28 h-28 bg-yellow-200 rounded-full blur-3xl"
      ></motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.15 }}
        transition={{ delay: 0.3, duration: 1.8 }}
        className="absolute bottom-24 right-24 w-36 h-36 bg-yellow-300 rounded-full blur-3xl"
      ></motion.div>

      <motion.div
        animate={{ y: [0, 10, 0], opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        className="absolute top-1/2 left-10 w-16 h-16 bg-yellow-100 rounded-full blur-2xl"
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
          className="text-4xl sm:text-5xl font-bold text-gray-900 leading-tight drop-shadow-[0_2px_10px_rgba(0,0,0,0.25)]"
        >
          Reconnect with Your{" "}
          <motion.span
            animate={{ backgroundColor: ["#FDE68A", "#FACC15", "#FDE68A"] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="px-2 py-1 rounded-md bg-yellow-300"
          >
            Inner Balance
          </motion.span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          variants={fadeUp(0.3)}
          className="text-gray-700 mt-4 text-base sm:text-lg max-w-2xl leading-relaxed px-2"
        >
          Harness the power of mindfulness and AI to rediscover calm, focus, and emotional clarity.
        </motion.p>

        {/* CTA Button */}
        <motion.button
          variants={fadeUp(0.5)}
          whileHover={{
            scale: 1.05,
            boxShadow: "0px 0px 15px rgba(234,179,8,0.5)",
          }}
          whileTap={{ scale: 0.95 }}
          className="mt-8 px-8 py-3 bg-yellow-400 text-black font-semibold rounded-full shadow-md hover:shadow-lg transition-all duration-300"
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
            <div className="absolute inset-0 bg-yellow-200 blur-3xl opacity-25 rounded-3xl scale-110"></div>
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
