import type { Metadata } from "next";
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { AppNav } from "@/components/app-nav";
import { AppStatusStrip } from "@/components/app-status-strip";
import { AuthStatusBanner } from "@/components/auth-status-banner";
import { ThemeToggle } from "@/components/theme-toggle";

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
            <div className="app-shell-headline">
              <div>
                <p className="app-shell-title">BLAIRE</p>
                <p className="app-shell-subtitle">Blacksite Lab AI Hub</p>
              </div>
              <ThemeToggle />
            </div>
            <AppNav />
            <AppStatusStrip />
          </div>
        </header>
        <div className="app-shell-content" id="main-content">
          <div className="page-wrap" style={{ marginBottom: "12px" }}>
            <AuthStatusBanner />
          </div>
          {children}
        </div>
      </body>
    </html>
  );
}
