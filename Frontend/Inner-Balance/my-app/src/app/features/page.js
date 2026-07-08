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
    <main className="bg-background text-foreground min-h-screen flex flex-col pt-20">
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
