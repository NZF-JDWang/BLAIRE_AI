import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import "./globals.css";

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
      <body>
        <header className="app-shell-header">
          <nav className="app-shell-nav">
            <Link href="/">Home</Link>
            <Link href="/chat">Chat</Link>
            <Link href="/swarm">Swarm</Link>
            <Link href="/knowledge">Knowledge</Link>
            <Link href="/settings">Settings</Link>
            <Link href="/approvals">Approvals</Link>
          </nav>
        </header>
        <div className="app-shell-content">{children}</div>
      </body>
    </html>
  );
}
