"use client";
import React from "react";
import { motion } from "framer-motion";
import Image from "next/image";
import formBoundary from "./assets/form-boundary.png"; 
const ContactForm = () => {
  const fadeUp = (delay = 0) => ({
    hidden: { opacity: 0, y: 25 },
    visible: { opacity: 1, y: 0, transition: { delay, duration: 0.6, ease: "easeOut" } },
  });

  return (
    <section className="relative bg-gradient-to-b from-white via-yellow-50 to-gray-50 px-6 py-16 sm:py-24 lg:px-8 overflow-hidden">
      
      {/*  Boundary Decorative Image */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.12 }}
        transition={{ duration: 1.5 }}
        className="absolute inset-0 flex justify-center items-center z-0 pointer-events-none select-none"
      >
        <Image
          src={formBoundary}
          alt="Form Decorative Boundary"
          fill
          className="object-contain scale-110"
          priority
        />
      </motion.div>

      {/* Decorative glowing elements */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.15 }}
        transition={{ duration: 1.5 }}
        className="absolute top-20 left-20 w-40 h-40 bg-yellow-300 rounded-full blur-3xl"
      ></motion.div>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.1 }}
        transition={{ delay: 0.3, duration: 1.8 }}
        className="absolute bottom-20 right-24 w-44 h-44 bg-yellow-400 rounded-full blur-3xl"
      ></motion.div>

      <div className="relative z-10 mx-auto max-w-2xl flex flex-col items-center justify-center text-center">
        {/* Heading */}
        <motion.h1
          variants={fadeUp(0.1)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-gray-900 drop-shadow-[0_2px_6px_rgba(0,0,0,0.15)]"
        >
          Let’s Connect with{" "}
          <motion.span
            animate={{ backgroundColor: ["#FDE68A", "#FACC15", "#FDE68A"] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="bg-yellow-300 px-3 py-1 rounded-md"
          >
            Inner Balance
          </motion.span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          variants={fadeUp(0.2)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mt-4 text-gray-700 text-lg leading-relaxed max-w-xl"
        >
          Have a feature idea, suggestion, or simply want to share your thoughts?
          <br /> We’d love to hear from you.
        </motion.p>
      </div>

      {/* Form */}
      <motion.form
        variants={fadeUp(0.3)}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="relative z-10 mx-auto mt-16 max-w-xl bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200 px-6 sm:px-10 py-10 sm:py-12"
      >
        <div className="grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-2">
          <div>
            <label htmlFor="first-name" className="block text-sm font-semibold text-gray-900">
              First Name
            </label>
            <input
              required
              type="text"
              id="first-name"
              placeholder="Your First Name"
              className="mt-2.5 block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm 
                ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-yellow-400 
                sm:text-sm sm:leading-6 transition duration-300 hover:ring-yellow-200"
            />
          </div>

          <div>
            <label htmlFor="last-name" className="block text-sm font-semibold text-gray-900">
              Last Name
            </label>
            <input
              required
              type="text"
              id="last-name"
              placeholder="Your Last Name"
              className="mt-2.5 block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm 
                ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-yellow-400 
                sm:text-sm sm:leading-6 transition duration-300 hover:ring-yellow-200"
            />
          </div>

          <div className="sm:col-span-2">
            <label htmlFor="company" className="block text-sm font-semibold text-gray-900">
              Company
            </label>
            <input
              required
              type="text"
              id="company"
              placeholder="Your Company Name"
              className="mt-2.5 block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm 
                ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-yellow-400 
                sm:text-sm sm:leading-6 transition duration-300 hover:ring-yellow-200"
            />
          </div>

          <div className="sm:col-span-2">
            <label htmlFor="email" className="block text-sm font-semibold text-gray-900">
              Email
            </label>
            <input
              required
              type="email"
              id="email"
              placeholder="Your Email Address"
              className="mt-2.5 block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm 
                ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-yellow-400 
                sm:text-sm sm:leading-6 transition duration-300 hover:ring-yellow-200"
            />
          </div>

          <div className="sm:col-span-2">
            <label htmlFor="phone" className="block text-sm font-semibold text-gray-900">
              Phone Number
            </label>
            <input
              required
              type="tel"
              id="phone"
              placeholder="Your Phone Number"
              className="mt-2.5 block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm 
                ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-yellow-400 
                sm:text-sm sm:leading-6 transition duration-300 hover:ring-yellow-200"
            />
          </div>

          <div className="sm:col-span-2">
            <label htmlFor="message" className="block text-sm font-semibold text-gray-900">
              Message
            </label>
            <textarea
              id="message"
              rows="4"
              placeholder="Share your thoughts..."
              className="mt-2.5 block w-full rounded-md border-0 px-3.5 py-2 text-gray-900 shadow-sm 
                ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-yellow-400 
                sm:text-sm sm:leading-6 transition duration-300 hover:ring-yellow-200"
            ></textarea>
          </div>
        </div>

        <motion.button
          whileHover={{
            scale: 1.05,
            boxShadow: "0 0 20px rgba(250, 204, 21, 0.5)",
          }}
          whileTap={{ scale: 0.97 }}
          type="submit"
          className="relative mt-10 w-full bg-gradient-to-r from-yellow-400 to-yellow-300 text-black font-semibold py-3 rounded-full shadow-md hover:shadow-lg transition-all"
        >
          Submit →
        </motion.button>
      </motion.form>
    </section>
  );
};

export default ContactForm;
