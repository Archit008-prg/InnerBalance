import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Primary sans-serif font (for headings/body)
const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  weight: ["400", "500", "600", "700"], // optional, if needed
});

// Mono font (for code blocks, monospace areas)
const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  weight: ["400", "500", "700"],
});

export const metadata = {
  title: "Inner Balance - AI-Powered Mental Health & Wellness",
  description: "Find your inner balance with AI-driven mental health assessments, expert consultations, and personalized wellness guidance. Start your journey to better mental well-being today.",
  keywords: "mental health, wellness, AI assessment, therapy, mindfulness, mental wellness",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
