"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Loader2, Heart, Shield, User, Mail, Key, Phone, BookOpen, Award, CheckCircle, AlertCircle, Eye, EyeOff } from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { register } from "../../lib/api";
import greenFocus from "../components/assets/green_focus.png";

export default function RegisterPage() {
  const router = useRouter();
  const [role, setRole] = useState("patient"); // 'patient' or 'doctor'
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  
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
  
  // Patient fields
  const [gender, setGender] = useState("M");
  const [phone, setPhone] = useState("");
  const [address, setAddress] = useState("");
  const [city, setCity] = useState("");
  const [educationOrOccupation, setEducationOrOccupation] = useState("");
  const [age, setAge] = useState("");
  const [maritalStatus, setMaritalStatus] = useState("Single");

  // Doctor fields
  const [specialization, setSpecialization] = useState("");
  const [license, setLicense] = useState("");
  const [experience, setExperience] = useState("");
  const [hospital, setHospital] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (!username || !email || !password || !firstName || !lastName) {
      setError("Please fill out all basic fields.");
      return;
    }

    setLoading(true);

    const payload = {
      username,
      email,
      password,
      first_name: firstName,
      last_name: lastName,
      role
    };

    if (role === "patient") {
      payload.gender = gender;
      payload.phone_number = phone;
      payload.address = address;
      payload.city = city;
      payload.education_or_occupation = educationOrOccupation;
      payload.age = age ? parseInt(age, 10) : null;
      payload.marital_status = maritalStatus;
    } else {
      payload.specialization = specialization || "General Psychiatry";
      payload.license_number = license || "N/A";
      payload.years_of_experience = parseInt(experience) || 0;
      payload.hospital_affiliation = hospital;
    }

    try {
      await register(payload);
      setSuccess(`Account registered successfully as a ${role}! Redirecting to login...`);
      setTimeout(() => {
        router.push("/login");
      }, 2500);
    } catch (err) {
      setError(err.message || "Registration failed. Please try a different username/email.");
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
        {/* Ambient background glows */}
        <div className="absolute top-1/4 left-1/4 w-80 h-80 bg-primary-accent/10 rounded-full blur-[110px] pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-teal-500/10 rounded-full blur-[130px] pointer-events-none" />

        <div className="w-full max-w-2xl bg-card-bg/75 backdrop-blur-xl border border-card-border rounded-3xl p-8 shadow-[var(--card-shadow)] relative z-10 transition-all duration-300">
          <div className="text-center mb-8">
            <div className="inline-flex p-3 bg-primary-accent/10 text-primary-accent rounded-full mb-3">
              <Heart size={32} />
            </div>
            <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight">Create Account</h2>
            <p className="text-sm text-text-muted mt-2">Join the Inner Balance Clinical Platform</p>
          </div>

          {/* Role selector buttons */}
          <div className="grid grid-cols-2 gap-4 mb-8">
            <button
              type="button"
              onClick={() => setRole("patient")}
              className={`py-3.5 rounded-2xl font-semibold border text-sm transition flex items-center justify-center gap-2 cursor-pointer ${
                role === "patient"
                  ? "bg-primary-accent text-slate-900 border-primary-accent shadow-md font-bold"
                  : "bg-card-bg/40 dark:bg-white/5 border-card-border hover:bg-card-bg/85"
              }`}
            >
              <User size={18} />
              Patient Account
            </button>
            <button
              type="button"
              onClick={() => setRole("doctor")}
              className={`py-3.5 rounded-2xl font-semibold border text-sm transition flex items-center justify-center gap-2 cursor-pointer ${
                role === "doctor"
                  ? "bg-primary-accent text-slate-900 border-primary-accent shadow-md font-bold"
                  : "bg-card-bg/40 dark:bg-white/5 border-card-border hover:bg-card-bg/85"
              }`}
            >
              <Shield size={18} />
              Clinician / Doctor
            </button>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 p-4 rounded-2xl mb-6 flex items-start gap-2.5 text-sm">
              <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="bg-green-500/10 border border-green-500/20 text-green-600 dark:text-green-400 p-4 rounded-2xl mb-6 flex items-start gap-2.5 text-sm">
              <CheckCircle size={18} className="flex-shrink-0 mt-0.5" />
              <span>{success}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Info Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">First Name</label>
                <input
                  type="text"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  className="w-full px-4 py-3.5 bg-slate-900/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition"
                  placeholder="John"
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Last Name</label>
                <input
                  type="text"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  className="w-full px-4 py-3.5 bg-slate-900/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition"
                  placeholder="Doe"
                  required
                />
              </div>
            </div>

            {/* Credential Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                  placeholder="johndoe123"
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Email Address</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                  placeholder="john@example.com"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-4 pr-10 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                  placeholder="Choose a strong password"
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

            <hr className="border-card-border" />

            {/* Role-Specific Fields */}
            {role === "patient" ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Phone Number</label>
                    <input
                      type="tel"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="+1 (555) 019-2834"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Gender</label>
                    <select
                      value={gender}
                      onChange={(e) => setGender(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                    >
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                      <option value="O">Other</option>
                      <option value="N">Prefer not to say</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Home Address</label>
                  <textarea
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    rows={2}
                    className="w-full px-4 py-3 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition resize-none text-foreground"
                    placeholder="123 Care Street, Medical Town"
                  />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">City</label>
                    <input
                      type="text"
                      value={city}
                      onChange={(e) => setCity(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="San Francisco"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Education / Occupation</label>
                    <input
                      type="text"
                      value={educationOrOccupation}
                      onChange={(e) => setEducationOrOccupation(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="Student / Engineer"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Age</label>
                    <input
                      type="number"
                      value={age}
                      onChange={(e) => setAge(e.target.value)}
                      min="0"
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="25"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Marital Status</label>
                    <select
                      value={maritalStatus}
                      onChange={(e) => setMaritalStatus(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                    >
                      <option value="Single">Single</option>
                      <option value="Married">Married</option>
                      <option value="Divorced">Divorced</option>
                      <option value="Widowed">Widowed</option>
                      <option value="Other">Other</option>
                      <option value="N">Prefer not to say</option>
                    </select>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Specialization</label>
                    <input
                      type="text"
                      value={specialization}
                      onChange={(e) => setSpecialization(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="e.g. Cognitive Behavioral Therapy, ADHD"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Medical License Number</label>
                    <input
                      type="text"
                      value={license}
                      onChange={(e) => setLicense(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="e.g. MD-98234-PSY"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Years of Experience</label>
                    <input
                      type="number"
                      value={experience}
                      onChange={(e) => setExperience(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="5"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-text-muted mb-2">Hospital Affiliation</label>
                    <input
                      type="text"
                      value={hospital}
                      onChange={(e) => setHospital(e.target.value)}
                      className="w-full px-4 py-3.5 bg-slate-900/5 dark:bg-white/5 border border-card-border focus:border-primary-accent focus:outline-none rounded-2xl text-sm transition text-foreground"
                      placeholder="e.g. Grace Medical Center"
                    />
                  </div>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-4 bg-primary-accent hover:bg-primary-accent/90 disabled:bg-primary-accent/50 disabled:cursor-not-allowed text-slate-900 font-extrabold rounded-2xl shadow-lg transition flex items-center justify-center gap-2 cursor-pointer text-sm"
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin" />
                  <span>Creating Account...</span>
                </>
              ) : (
                <span>Register Account</span>
              )}
            </button>
          </form>

          <div className="text-center mt-6 text-sm text-text-muted">
            Already have an account?{" "}
            <Link href="/login" className="text-primary-accent hover:underline font-semibold">
              Log in here
            </Link>
          </div>
        </div>
      </div>

      <Footer />
    </main>
  );
}
