"use client";

import { useEffect, useState } from "react";

import { getRuntimeOptionsTyped, RuntimeOptions } from "@/lib/api";

type Preferences = {
  searchMode: string;
  modelClass: string;
  modelOverride: string;
};

const STORAGE_KEY = "blaire_preferences_v1";

export default function SettingsPage() {
  const [options, setOptions] = useState<RuntimeOptions | null>(null);
  const [prefs, setPrefs] = useState<Preferences>({
    searchMode: "searxng_only",
    modelClass: "general",
    modelOverride: ""
  });

  useEffect(() => {
    getRuntimeOptionsTyped()
      .then((data) => {
        setOptions(data);
        setPrefs((prev) => ({ ...prev, searchMode: data.default_search_mode }));
      })
      .catch(() => setOptions(null));
  }, []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Preferences;
      setPrefs(parsed);
    } catch {
      // Ignore malformed local preferences.
    }
  }, []);

  function save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  }

  const models = options?.model_allowlist[prefs.modelClass] ?? [];

  return (
    <main style={{ maxWidth: "760px", margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>Settings</h1>
      <p style={{ marginBottom: "16px" }}>
        Runtime preference controls. These are saved locally until backend preference persistence is added.
      </p>

      <div style={{ display: "grid", gap: "12px" }}>
        <label>
          Search mode
          <select
            style={{ marginLeft: "8px" }}
            value={prefs.searchMode}
            onChange={(e) => setPrefs((prev) => ({ ...prev, searchMode: e.target.value }))}
          >
            {(options?.search_modes ?? ["searxng_only"]).map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </label>

        <label>
          Model class
          <select
            style={{ marginLeft: "8px" }}
            value={prefs.modelClass}
            onChange={(e) =>
              setPrefs((prev) => ({
                ...prev,
                modelClass: e.target.value,
                modelOverride: ""
              }))
            }
          >
            <option value="general">general</option>
            <option value="vision">vision</option>
            <option value="embedding">embedding</option>
            <option value="code">code</option>
          </select>
        </label>

        <label>
          Model override
          <select
            style={{ marginLeft: "8px" }}
            value={prefs.modelOverride}
            onChange={(e) => setPrefs((prev) => ({ ...prev, modelOverride: e.target.value }))}
          >
            <option value="">(class default)</option>
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button style={{ marginTop: "16px", padding: "10px 16px" }} onClick={save}>
        Save Preferences
      </button>
    </main>
  );
}

