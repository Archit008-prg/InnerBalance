"use client";
import React from "react";
import { useRouter } from "next/navigation"; // Next.js router
import { motion } from "framer-motion";
import { FileText, Presentation, Brain } from "lucide-react";

const HelpCards = () => {
  const router = useRouter();

  // Define the cards with onClick handlers
  const cards = [
    {
      icon: <FileText className="w-10 h-10 text-primary-accent" />,
      title: "Download Documentation",
      description:
        "Access detailed documentation that guides you through every step of your mental wellness journey.",
      buttonText: "Download",
      onClick: () => window.open("/docs.pdf", "_blank"), // example download
    },
    {
      icon: <Presentation className="w-10 h-10 text-primary-accent" />,
      title: "Download PPT",
      description:
        "Get the presentation slides and share insights with your team or peers easily.",
      buttonText: "Download PPT",
      onClick: () => window.open("/presentation.pptx", "_blank"), // example download
    },
    {
      icon: <Brain className="w-10 h-10 text-primary-accent" />,
      title: "Go to Test",
      description:
        "Take a personalized AI-driven test to explore your emotional balance and self-awareness level.",
      buttonText: "Start Test",
      onClick: () => router.push("/test"), // navigate to test page
    },
  ];

  return (
    <section className="relative flex flex-col justify-center items-center min-h-screen bg-background text-foreground transition-colors duration-300 text-center px-6 py-24 overflow-hidden">
      {/* Background animation */}
      <motion.div
        className="absolute inset-0 opacity-20 bg-[radial-gradient(circle_at_bottom_left,rgba(0,223,129,0.15),transparent_50%)]"
        animate={{ scale: [1, 1.05, 1] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
      ></motion.div>

      {/* Heading */}
      <motion.h2
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="text-4xl sm:text-5xl font-extrabold text-foreground leading-tight z-10"
      >
        Explore Helpful{" "}
        <span className="bg-primary-accent/10 text-primary-accent border border-primary-accent/20 px-4 py-1.5 rounded-full text-3xl sm:text-5xl align-middle inline-block shadow-sm">
          Resources
        </span>
      </motion.h2>

      {/* Subtitle */}
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.8 }}
        className="text-text-muted mt-4 text-lg sm:text-xl max-w-2xl z-10"
      >
        Gain insights, prepare with our presentations, and take a guided test to
        align your mind and purpose.
      </motion.p>

      {/* Cards Section */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 1 }}
        className="z-10 mt-16 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-10 w-full max-w-6xl"
      >
        {cards.map((card, index) => (
          <motion.div
            key={index}
            whileHover={{ y: -8, scale: 1.03 }}
            transition={{ type: "spring", stiffness: 200 }}
            className="bg-card-bg/75 backdrop-blur-sm border border-gray-300 dark:border-zinc-700 rounded-2xl shadow-lg hover:shadow-2xl p-8 flex flex-col items-center text-center transition"
          >
            <div className="mb-4">{card.icon}</div>
            <h3 className="text-2xl font-semibold text-foreground mb-3">
              {card.title}
            </h3>
            <p className="text-text-muted mb-6 text-base leading-relaxed">
              {card.description}
            </p>
            <motion.button
              onClick={card.onClick} // Added click handler
              whileHover={{ scale: 1.07 }}
              whileTap={{ scale: 0.95 }}
              className="mt-auto px-6 py-2 bg-primary-accent text-slate-900 font-extrabold rounded-full shadow-md hover:shadow-lg transition cursor-pointer"
            >
              {card.buttonText}
            </motion.button>
          </motion.div>
        ))}
      </motion.div>
    </section>
  );
};

export default HelpCards;
