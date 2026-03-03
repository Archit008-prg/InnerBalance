"use client";
import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { pageTransition } from "../lib/animations";

export default function PageTransition({ children }) {
  const containerRef = useRef(null);
  const pathname = usePathname();
  const prevPathnameRef = useRef(pathname);

  useEffect(() => {
    if (prevPathnameRef.current !== pathname && containerRef.current) {
      pageTransition.enter(containerRef.current);
      prevPathnameRef.current = pathname;
    }
  }, [pathname]);

  return (
    <div ref={containerRef} className="min-h-screen">
      {children}
    </div>
  );
}

