"use client";
import React from "react";
import Navbar from "../components/Navbar";
import Helphero from "../components/Helphero";
import Featureshero from "../components/Featureshero";
import ContactForm from "../components/ContactForm";
import HelpCards from "../components/HelpCards";
import Footer from "../components/Footer";

const Page = () => {
  return (
    <main className="bg-white text-black min-h-screen flex flex-col">
      {/* Navbar */}
      <Navbar />
          <Featureshero />
          <HelpCards />


      {/* Footer */}
      <Footer />
    </main>
  );
};

export default Page;
