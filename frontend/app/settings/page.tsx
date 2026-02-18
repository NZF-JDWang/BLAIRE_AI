"use client";

import { useEffect, useState } from "react";

import {
  getBrowserApiKey,
  getCliUnrestrictedMode,
  getMyPreferences,
  getRuntimeOptionsTyped,
  RuntimeOptions,
  setBrowserApiKey,
  updateCliUnrestrictedMode,
  updateMyPreferences
} from "@/lib/api";

type Preferences = {
  searchMode: string;
  modelClass: string;
  modelOverride: string;
};

const DANGER_CONFIRM_TEXT = "I UNDERSTAND THIS IS DANGEROUS";

export default function SettingsPage() {
  const [options, setOptions] = useState<RuntimeOptions | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [prefs, setPrefs] = useState<Preferences>({
    searchMode: "searxng_only",
    modelClass: "general",
    modelOverride: ""
  });
  const [unrestrictedEnabled, setUnrestrictedEnabled] = useState(false);
  const [dangerAck, setDangerAck] = useState(false);
  const [dangerText, setDangerText] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    setApiKey(getBrowserApiKey());
    Promise.all([getRuntimeOptionsTyped(), getMyPreferences(), getCliUnrestrictedMode()])
      .then(([runtime, current, mode]) => {
        setOptions(runtime);
        setPrefs({
          searchMode: current.search_mode,
          modelClass: current.model_class,
          modelOverride: current.model_override ?? ""
        });
        setUnrestrictedEnabled(mode.enabled);
      })
      .catch(() => setOptions(null));
  }, []);

  async function save() {
    setBrowserApiKey(apiKey);
    await updateMyPreferences({
      search_mode: prefs.searchMode,
      model_class: prefs.modelClass,
      model_override: prefs.modelOverride || null
    });
    setMessage("Preferences saved.");
  }

  async function saveUnrestrictedMode() {
    setMessage("");
    const response = await updateCliUnrestrictedMode({
      enabled: unrestrictedEnabled,
      acknowledged_danger: dangerAck,
      confirmation_text: dangerText
    });
    setUnrestrictedEnabled(response.enabled);
    setMessage(response.enabled ? "Dangerous mode enabled. Restrictions are OFF." : "Dangerous mode disabled.");
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
          API key
          <input
            style={{ marginLeft: "8px", minWidth: "260px" }}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Paste your user API key"
          />
        </label>

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

      <section
        style={{ marginTop: "24px", padding: "12px", border: "1px solid #ef4444", borderRadius: "8px", background: "#fff7ed" }}
      >
        <h2 style={{ margin: "0 0 8px", color: "#b91c1c" }}>Dangerous mode (disable CLI restrictions)</h2>
        <p style={{ margin: "0 0 8px" }}>
          Warning: this disables command restrictions and allows direct local command execution. Use only if you fully trust prompts.
        </p>
        <label style={{ display: "block", marginBottom: "8px" }}>
          <input
            type="checkbox"
            checked={unrestrictedEnabled}
            onChange={(e) => setUnrestrictedEnabled(e.target.checked)}
          />{" "}
          Enable dangerous unrestricted mode
        </label>
        <label style={{ display: "block", marginBottom: "8px" }}>
          <input type="checkbox" checked={dangerAck} onChange={(e) => setDangerAck(e.target.checked)} /> I understand this can be dangerous
        </label>
        <label style={{ display: "block", marginBottom: "8px" }}>
          Type exactly: <code>{DANGER_CONFIRM_TEXT}</code>
          <input
            style={{ marginLeft: "8px", minWidth: "360px" }}
            value={dangerText}
            onChange={(e) => setDangerText(e.target.value)}
            placeholder={DANGER_CONFIRM_TEXT}
          />
        </label>
        <button style={{ padding: "8px 12px" }} onClick={saveUnrestrictedMode}>
          Save Dangerous Mode Setting
        </button>
      </section>
      {message ? <p style={{ marginTop: "10px" }}>{message}</p> : null}
    </main>
  );
}
