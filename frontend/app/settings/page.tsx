"use client";

import { useEffect, useState } from "react";

import {
  getBrowserApiKey,
  getMyPreferences,
  getRuntimeOptionsTyped,
  RuntimeOptions,
  setBrowserApiKey,
  updateMyPreferences,
} from "@/lib/api";

type Preferences = {
  searchMode: string;
  modelClass: string;
  modelOverride: string;
};

export default function SettingsPage() {
  const [options, setOptions] = useState<RuntimeOptions | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [prefs, setPrefs] = useState<Preferences>({
    searchMode: "searxng_only",
    modelClass: "general",
    modelOverride: "",
  });
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setApiKey(getBrowserApiKey());
    Promise.all([getRuntimeOptionsTyped(), getMyPreferences()])
      .then(([runtime, current]) => {
        setOptions(runtime);
        setPrefs({
          searchMode: current.search_mode,
          modelClass: current.model_class,
          modelOverride: current.model_override ?? "",
        });
      })
      .catch(() => setOptions(null));
  }, []);

  async function save() {
    setSaving(true);
    setError("");
    setStatus("");
    try {
      setBrowserApiKey(apiKey);
      await updateMyPreferences({
        search_mode: prefs.searchMode,
        model_class: prefs.modelClass,
        model_override: prefs.modelOverride || null,
      });
      setStatus("Preferences saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save preferences");
    } finally {
      setSaving(false);
    }
  }

  const models = options?.model_allowlist[prefs.modelClass] ?? [];

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Runtime Preferences</p>
        <h1 className="page-title">Tune search behavior and model defaults per user.</h1>
        <p className="page-description">
          API key is stored in the browser for proxy calls. Preferences are synced through backend user preference APIs.
        </p>
      </section>

      <section className="surface stack" aria-label="Preferences form">
        <label className="field-label">
          API key
          <input
            className="input"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Paste your user API key"
          />
        </label>

        <label className="field-label">
          Search mode
          <select
            className="select"
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

        <label className="field-label">
          Model class
          <select
            className="select"
            value={prefs.modelClass}
            onChange={(e) =>
              setPrefs((prev) => ({
                ...prev,
                modelClass: e.target.value,
                modelOverride: "",
              }))
            }
          >
            <option value="general">general</option>
            <option value="vision">vision</option>
            <option value="embedding">embedding</option>
            <option value="code">code</option>
          </select>
        </label>

        <label className="field-label">
          Model override
          <select
            className="select"
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

        <div className="toolbar">
          <button className="button button-primary" onClick={() => void save()} disabled={saving}>
            {saving ? "Saving..." : "Save preferences"}
          </button>
          {status ? <span className="pill success">{status}</span> : null}
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>
    </main>
  );
}
