import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Primary sans-serif font (for headings/body)
const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  weight: ["400", "500", "600", "700"],
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
  const themeScript = `
    (function() {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    })();
  `;

  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} dark`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="antialiased font-sans" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
