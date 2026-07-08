"use client";
import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { gsap } from "gsap";
import { Menu, X, Sun, Moon } from "lucide-react";
import logo from "./assets/logo.png";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState("light");
  const pathname = usePathname();
  const navRef = useRef(null);
  const menuRef = useRef(null);
  
  // Session states
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userRole, setUserRole] = useState(null);

  useEffect(() => {
    const checkAuth = () => {
      const token = typeof window !== 'undefined' ? localStorage.getItem("access_token") : null;
      const role = typeof window !== 'undefined' ? localStorage.getItem("user_role") : null;
      setIsLoggedIn(!!token);
      setUserRole(role);
    };
    checkAuth();
    
    if (typeof window !== 'undefined') {
      window.addEventListener("storage", checkAuth);
      return () => window.removeEventListener("storage", checkAuth);
    }
  }, [pathname]);

  useEffect(() => {
    const isDark = document.documentElement.classList.contains("dark");
    setTheme(isDark ? "dark" : "light");
  }, []);

  const toggleTheme = () => {
    if (theme === "light") {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
      setTheme("dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
      setTheme("light");
    }
  };

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
    ...(isLoggedIn && userRole === "patient" ? [{ name: "Take Test", href: "/test" }] : []),
    ...(isLoggedIn && userRole === "patient" ? [{ name: "Portal", href: "/patient" }] : []),
    ...(isLoggedIn && userRole === "doctor" ? [{ name: "Dashboard", href: "/doctor" }] : []),
    { name: "Contact", href: "/contact" },
  ];

  const handleSignOut = () => {
    localStorage.clear();
    setMenuOpen(false);
    window.location.href = "/";
  };

  return (
    <nav ref={navRef} className="w-full bg-transparent fixed top-0 left-0 z-50 transition-all duration-300 py-2 sm:py-3 no-print">
      <div className="max-w-7xl mx-auto px-6 py-2 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-2.5 group">
          <Image src={logo} alt="Logo" width={42} height={32} className="transition-transform group-hover:scale-105" />
          <h2 className="text-sm sm:text-base font-bold text-foreground tracking-tight">Inner Balance</h2>
        </Link>

        {/* Desktop Menu - Floating Pill */}
        <ul className="hidden md:flex space-x-6 lg:space-x-8 text-[11px] uppercase tracking-wider font-bold text-foreground bg-slate-900/10 dark:bg-white/5 backdrop-blur-md border border-card-border px-8 py-2.5 rounded-full shadow-[0_8px_32px_rgba(0,0,0,0.06)]">
          {navItems.map((item, index) => (
            <li key={index} className="relative group">
              <Link
                href={item.href}
                className={`transition-colors ${
                  pathname === item.href
                    ? "text-primary-accent"
                    : "text-foreground group-hover:text-primary-accent"
                }`}
              >
                {item.name}
              </Link>
            </li>
          ))}
        </ul>

        {/* Theme and Auth Buttons */}
        <div className="hidden md:flex items-center gap-4">
          <button
            onClick={toggleTheme}
            className="p-2 text-foreground hover:text-primary-accent rounded-full transition-all focus:outline-none cursor-pointer"
            aria-label="Toggle Theme"
          >
            {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          
          {isLoggedIn ? (
            <button
              onClick={handleSignOut}
              className="px-5 py-2.5 bg-[#E53E3E] hover:bg-[#C53030] text-white rounded-full transition-all text-xs font-bold cursor-pointer shadow-md"
            >
              Sign Out
            </button>
          ) : (
            <Link
              href="/login"
              className="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 dark:bg-white dark:hover:bg-white/90 text-white dark:text-slate-900 rounded-full transition-all text-xs font-bold cursor-pointer shadow-md"
            >
              Sign In
            </Link>
          )}
        </div>

        {/* Mobile Menu Button & Mobile Theme Toggle */}
        <div className="md:hidden flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="p-2 text-foreground hover:text-primary-accent rounded-full transition-all focus:outline-none"
            aria-label="Toggle Theme"
          >
            {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="text-foreground focus:outline-none p-2"
            aria-label="Toggle menu"
          >
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Dropdown Menu */}
      <div
        ref={menuRef}
        className="md:hidden bg-card-bg shadow-lg overflow-hidden border-b border-card-border"
        style={{ height: 0, opacity: 0 }}
      >
        <ul className="flex flex-col items-center py-4 space-y-4 font-semibold text-foreground">
          {navItems.map((item, index) => (
            <li key={index}>
              <Link
                href={item.href}
                onClick={() => setMenuOpen(false)}
                className={`hover:text-primary-accent transition ${
                  pathname === item.href ? "text-primary-accent" : ""
                }`}
              >
                {item.name}
              </Link>
            </li>
          ))}
          <li>
            {isLoggedIn ? (
              <button
                onClick={handleSignOut}
                className="px-5 py-2 bg-[#E53E3E]/10 text-[#E53E3E] border border-[#E53E3E]/20 rounded-full transition text-xs font-semibold cursor-pointer"
              >
                Sign Out
              </button>
            ) : (
              <Link
                href="/login"
                onClick={() => setMenuOpen(false)}
                className="px-5 py-2 bg-primary-accent text-white rounded-full transition text-xs font-semibold cursor-pointer"
              >
                Sign In
              </Link>
            )}
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
