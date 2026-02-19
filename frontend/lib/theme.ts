export type ThemeMode = "system" | "light" | "dark";

const THEME_KEY = "blaire_theme_mode";

function resolveTheme(mode: ThemeMode): "light" | "dark" {
  if (mode === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return mode;
}

export function getStoredThemeMode(): ThemeMode {
  if (typeof window === "undefined") {
    return "system";
  }
  const stored = window.localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark" || stored === "system") {
    return stored;
  }
  return "system";
}

export function applyThemeMode(mode: ThemeMode): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(THEME_KEY, mode);
  const resolved = resolveTheme(mode);
  document.documentElement.dataset.theme = resolved;
  window.dispatchEvent(new Event("blaire-theme-changed"));
}

export function initThemeMode(): ThemeMode {
  const mode = getStoredThemeMode();
  const resolved = resolveTheme(mode);
  document.documentElement.dataset.theme = resolved;
  return mode;
}
