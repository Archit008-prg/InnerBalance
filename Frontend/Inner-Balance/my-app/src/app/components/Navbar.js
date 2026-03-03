"use client";
import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { gsap } from "gsap";
import { Menu, X } from "lucide-react";
import logo from "./assets/logo.png";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();
  const navRef = useRef(null);
  const menuRef = useRef(null);

  useEffect(() => {
    if (navRef.current) {
      gsap.fromTo(
        navRef.current,
        { y: -100, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.6, ease: "power2.out" }
      );
    }
  }, []);

  useEffect(() => {
    if (menuRef.current) {
      if (menuOpen) {
        gsap.fromTo(
          menuRef.current,
          { height: 0, opacity: 0 },
          { height: "auto", opacity: 1, duration: 0.3, ease: "power2.out" }
        );
      } else {
        gsap.to(menuRef.current, {
          height: 0,
          opacity: 0,
          duration: 0.3,
          ease: "power2.in",
        });
      }
    }
  }, [menuOpen]);

  const navItems = [
    { name: "Home", href: "/" },
    { name: "About", href: "/about" },
    { name: "Features", href: "/features" },
    { name: "Test", href: "/test" },
    { name: "Contact", href: "/contact" },
  ];

  return (
    <nav ref={navRef} className="w-full bg-white/95 backdrop-blur-sm shadow-sm fixed top-0 left-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-2 group">
          <Image src={logo} alt="Logo" width={45} height={35} className="transition-transform group-hover:scale-105" />
          <h2 className="text-base sm:text-lg font-bold text-gray-800">Inner Balance</h2>
        </Link>

        {/* Desktop Menu */}
        <ul className="hidden md:flex space-x-6 lg:space-x-10 text-sm font-semibold text-gray-700">
          {navItems.map((item, index) => (
            <li key={index} className="relative group">
              <Link
                href={item.href}
                className={`transition-colors ${
                  pathname === item.href
                    ? "text-yellow-500"
                    : "text-gray-700 group-hover:text-yellow-500"
                }`}
              >
                {item.name}
              </Link>
              <span
                className={`absolute left-0 -bottom-1 h-[2px] bg-yellow-500 transition-all ${
                  pathname === item.href ? "w-full" : "w-0 group-hover:w-full"
                }`}
              ></span>
            </li>
          ))}
        </ul>

        {/* Help Button */}
        <div className="hidden md:block">
          <Link
            href="/help"
            className="px-4 py-1.5 border-2 border-yellow-500 text-yellow-500 rounded-full hover:bg-yellow-500 hover:text-white transition-all text-sm font-medium"
          >
            Help ?
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <div className="md:hidden flex items-center">
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="text-gray-700 focus:outline-none p-2"
            aria-label="Toggle menu"
          >
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Dropdown Menu */}
      <div
        ref={menuRef}
        className="md:hidden bg-white shadow-lg overflow-hidden"
        style={{ height: 0, opacity: 0 }}
      >
        <ul className="flex flex-col items-center py-4 space-y-4 font-semibold text-gray-700">
          {navItems.map((item, index) => (
            <li key={index}>
              <Link
                href={item.href}
                onClick={() => setMenuOpen(false)}
                className={`hover:text-yellow-500 transition ${
                  pathname === item.href ? "text-yellow-500" : ""
                }`}
              >
                {item.name}
              </Link>
            </li>
          ))}
          <li>
            <Link
              href="/help"
              onClick={() => setMenuOpen(false)}
              className="px-4 py-1.5 border-2 border-yellow-500 text-yellow-500 rounded-full hover:bg-yellow-500 hover:text-white transition"
            >
              Help ?
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
