"use client";

import { useEffect, useState } from "react";

import { applyThemeMode, getStoredThemeMode, initThemeMode, ThemeMode } from "@/lib/theme";

export function ThemeToggle() {
  const [mode, setMode] = useState<ThemeMode>(() => getStoredThemeMode());

  useEffect(() => {
    initThemeMode();

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onMediaChange = () => {
      const selected = getStoredThemeMode();
      if (selected === "system") {
        initThemeMode();
      }
    };
    media.addEventListener("change", onMediaChange);
    return () => media.removeEventListener("change", onMediaChange);
  }, []);

  function onChange(nextMode: ThemeMode) {
    setMode(nextMode);
    applyThemeMode(nextMode);
  }

  return (
    <label className="field-label theme-toggle">
      Theme
      <select className="select" value={mode} onChange={(e) => onChange(e.target.value as ThemeMode)}>
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
  );
}
