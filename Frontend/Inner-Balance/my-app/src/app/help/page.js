"use client";
import React from "react";
import Navbar from "../components/Navbar";

import Helphero from "../components/Helphero";
import ContactForm from "../components/ContactForm";
import HelpCards from "../components/HelpCards";
import Footer from "../components/Footer";

const Page = () => {
  return (
    <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
      {/* Navbar */}
      <Navbar />

      {/* Helphero Section */}
      <div className="flex-grow">
        <Helphero />
      </div>
      
 {/* HelpCards  Section */}
      <div className="flex-grow">
        <HelpCards />
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
