"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, Heart, Shield, User, Key, AlertCircle, Eye, EyeOff } from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { login } from "../../lib/api";
import greenFocus from "../components/assets/green_focus.png";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const checkTheme = () => {
      const isDark = document.documentElement.classList.contains("dark");
      setIsDarkMode(isDark);
    };
    checkTheme();

    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  // Redirect if already logged in
  useEffect(() => {
    const role = localStorage.getItem("user_role");
    if (role === "doctor") {
      router.push("/doctor");
    } else if (role === "patient") {
      router.push("/patient");
    }
  }, [router]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username || !password) {
      setError("Please enter both username and password.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const profile = await login(username, password);
      if (profile.role === "doctor") {
        router.push("/doctor");
      } else if (profile.role === "patient") {
        router.push("/patient");
      } else {
        router.push("/");
      }
    } catch (err) {
      setError(err.message || "Invalid credentials. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      className="bg-background text-foreground min-h-screen flex flex-col pt-20 relative transition-colors duration-300 overflow-hidden"
    >
      {/* Blurred background image container */}
      <div 
        className="absolute inset-0 z-0 bg-cover bg-center bg-no-repeat filter blur-md scale-105 transition-all duration-300"
        style={{
          backgroundImage: `url(${greenFocus.src})`,
        }}
      />
      {/* Background gradients overlays (semi-transparent dark in dark theme, very clear in light theme) */}
      <div className="absolute inset-0 bg-white/0 dark:bg-black/45 z-0 transition-colors duration-300 pointer-events-none" />
      <Navbar />

      <div className="flex-grow flex items-center justify-center p-6 md:p-10 relative overflow-hidden z-10">
        {/* Decorative background blur shapes */}
        <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-primary-accent/10 rounded-full blur-[100px] pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-teal-500/10 rounded-full blur-[120px] pointer-events-none" />

        <div className="w-full max-w-md bg-card-bg/75 backdrop-blur-xl border border-card-border rounded-3xl p-8 shadow-[var(--card-shadow)] relative z-10 transition-all duration-300">
          <div className="text-center mb-8">
            <div className="inline-flex p-3 bg-primary-accent/10 text-primary-accent rounded-full mb-3">
              <Heart size={32} className="animate-pulse" />
            </div>
            <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight">Welcome Back</h2>
            <p className="text-sm text-text-muted mt-2">Access the pre-consultation intelligence portal</p>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 p-4 rounded-2xl mb-6 flex items-start gap-2.5 text-sm">
              <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Username</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-text-muted">
                  <User size={18} />
                </span>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full pl-10 pr-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                  placeholder="Enter your username"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Password</label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-text-muted">
                  <Key size={18} />
                </span>
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-10 pr-10 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-text-muted hover:text-primary-accent transition focus:outline-none cursor-pointer"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-4 bg-primary-accent hover:bg-primary-accent/90 disabled:bg-primary-accent/50 disabled:cursor-not-allowed text-slate-900 font-extrabold rounded-2xl shadow-lg transition flex items-center justify-center gap-2 cursor-pointer text-sm"
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  <span>Logging in...</span>
                </>
              ) : (
                <span>Log In</span>
              )}
            </button>
          </form>

          <div className="text-center mt-6 text-sm text-text-muted">
            Don&apos;t have an account?{" "}
            <Link href="/register" className="text-primary-accent hover:underline font-semibold">
              Register here
            </Link>
          </div>
        </div>
      </div>

      <Footer />
    </main>
  );
}
