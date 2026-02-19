import type { Metadata } from "next";
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { AppNav } from "@/components/app-nav";

import "./globals.css";

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
});

const monoFont = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "BLAIRE",
  description: "Blacksite Lab AI Hub",
  applicationName: "BLAIRE",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "BLAIRE",
  },
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${displayFont.variable} ${monoFont.variable}`}>
        <a href="#main-content" className="skip-link">
          Skip to main content
        </a>
        <div className="app-shell-bg" aria-hidden="true" />
        <header className="app-shell-header">
          <div className="app-shell-header-inner">
            <p className="app-shell-title">BLAIRE</p>
            <p className="app-shell-subtitle">Blacksite Lab AI Hub</p>
            <AppNav />
          </div>
        </header>
        <div className="app-shell-content" id="main-content">
          {children}
        </div>
      </body>
    </html>
  );
}
