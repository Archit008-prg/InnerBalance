"use client";
import React from "react";
import Navbar from "../components/Navbar";
import Abouthero from "../components/Abouthero";
import ResearchSection from "../components/ResearchSection";
import Footer from "../components/Footer";

const Page = () => {
  return (
    <main className="bg-white text-black min-h-screen flex flex-col">
      {/* Navbar */}
      <Navbar />

      {/* Abouthero Section }*/}
      <div className="flex-grow">
        <Abouthero />
      </div>
      
    {/*ResearchSection Section*/}
      <div className="flex-grow">
        <ResearchSection />
      </div>

      {/* Footer */}
      <Footer />
    </main>
  );
};

export default Page;
