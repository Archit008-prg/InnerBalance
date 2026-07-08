"use client";
import React from "react";
import Navbar from "./components/Navbar";
import Data from "./components/Data";
import Hero from "./components/Hero";
import Testimonials from "./components/Testimonials";
import About from "./components/About";
import Faq from "./components/Faq";
import CallToAction from "./components/CallToAction";
import Footer from "./components/Footer";

const Page = () => {
  return (
    <main className="bg-background text-foreground min-h-screen flex flex-col pt-0">
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

      {/* Call To Action Banner */}
      <div className="flex-grow">
        <CallToAction />
      </div>

      {/* Footer */}
      <Footer />
    </main>
  );
};

export default Page;
