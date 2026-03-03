"use client";
import React from "react";
import Navbar from "./components/Navbar";
import Data from "./components/Data";
import Hero from "./components/Hero";
import Testimonials from "./components/Testimonials";
import About from "./components/About";
import Faq from "./components/Faq";
import Footer from "./components/Footer";

const Page = () => {
  return (
    <main className="bg-white text-black min-h-screen flex flex-col">
      {/* Navbar */}
      <Navbar />

      {/* Hero Section */}
      <div className="flex-grow">
        <Hero />
      </div>
      
      {/* Data Section */}
      <div className="flex-grow">
        <Data/>
      </div>
      
      {/* Testimonials Section */}
      <div className="flex-grow">
        <Testimonials />
      </div>

 {/* About Section */}
      <div className="flex-grow">
        <About />
      </div>
         {/* FAQ Section */}
      <div className="flex-grow">
        <Faq />
      </div>

      {/* Footer */}
      <Footer />
    </main>
  );
};

export default Page;
