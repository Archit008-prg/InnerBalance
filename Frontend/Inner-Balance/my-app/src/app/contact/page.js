"use client";
import React from "react";
import Navbar from "../components/Navbar";

import Contacthero from "../components/Contacthero";
import ContactForm from "../components/ContactForm";

import Footer from "../components/Footer";

const Page = () => {
  return (
    <main className="bg-white text-black min-h-screen flex flex-col">
      {/* Navbar */}
      <Navbar />

      {/* Contacthero Section */}
      <div className="flex-grow">
        <Contacthero />
      </div>
      
    {/* ContactForm Section */}
      <div className="flex-grow">
        <ContactForm />
      </div>

      {/* Footer */}
      <Footer />
    </main>
  );
};

export default Page;
