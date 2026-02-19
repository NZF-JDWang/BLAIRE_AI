"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/setup", label: "Setup" },
  { href: "/", label: "Home" },
  { href: "/chat", label: "Chat" },
  { href: "/swarm", label: "Swarm" },
  { href: "/knowledge", label: "Knowledge" },
  { href: "/approvals", label: "Approvals" },
  { href: "/settings", label: "Settings" },
  { href: "/search", label: "Search" },
  { href: "/capabilities", label: "Capabilities" },
] as const;

export function AppNav() {
  const pathname = usePathname();

  return (
    <nav className="app-nav" aria-label="Main navigation">
      {NAV_ITEMS.map((item) => {
        const isActive = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={isActive ? "page" : undefined}
            className={isActive ? "app-nav-link active" : "app-nav-link"}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
