"use client";

import { useEffect, useState } from "react";

import {
  ApiRequestError,
  DependencyStatus,
  formatApiError,
  getBrowserApiKey,
  getDependencyStatus,
  getMyPreferences,
  getRuntimeConfig,
  getRuntimeOptionsTyped,
  RuntimeConfigBundle,
  RuntimeOptions,
  setBrowserApiKey,
  updateRuntimeConfig,
  updateMyPreferences,
} from "@/lib/api";

type Preferences = {
  searchMode: string;
  modelClass: string;
  modelOverride: string;
};

export default function SettingsPage() {
  const [options, setOptions] = useState<RuntimeOptions | null>(null);
  const [dependencies, setDependencies] = useState<DependencyStatus | null>(null);
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfigBundle | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [prefs, setPrefs] = useState<Preferences>({
    searchMode: "searxng_only",
    modelClass: "general",
    modelOverride: "",
  });
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [saving, setSaving] = useState(false);
  const [savingRuntime, setSavingRuntime] = useState(false);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    setApiKey(getBrowserApiKey());
    Promise.all([
      getRuntimeOptionsTyped(),
      getMyPreferences(),
      getDependencyStatus(),
      getRuntimeConfig().catch((err) => {
        if (err instanceof ApiRequestError && err.status === 403) {
          return null;
        }
        throw err;
      }),
    ])
      .then(([runtime, current, deps, config]) => {
        setOptions(runtime);
        setDependencies(deps);
        setRuntimeConfig(config);
        setLoadError("");
        setPrefs({
          searchMode: current.search_mode,
          modelClass: current.model_class,
          modelOverride: current.model_override ?? "",
        });
      })
      .catch((err) => {
        setOptions(null);
        setDependencies(null);
        setRuntimeConfig(null);
        setLoadError(formatApiError(err, "Failed to load runtime options"));
      });
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
      setError(formatApiError(err, "Failed to save preferences"));
    } finally {
      setSaving(false);
    }
  }

  async function testConnection() {
    setTesting(true);
    setError("");
    setStatus("");
    try {
      setBrowserApiKey(apiKey);
      await getRuntimeOptionsTyped();
      setStatus("API key verified. Runtime options are reachable.");
    } catch (err) {
      setError(formatApiError(err, "Connection test failed"));
    } finally {
      setTesting(false);
    }
  }

  async function saveRuntimeConfig() {
    if (!runtimeConfig) return;
    setSavingRuntime(true);
    setError("");
    setStatus("");
    try {
      const updated = await updateRuntimeConfig({
        search_mode_default: runtimeConfig.overrides.search_mode_default,
        sensitive_actions_enabled: runtimeConfig.overrides.sensitive_actions_enabled,
        approval_token_ttl_minutes: runtimeConfig.overrides.approval_token_ttl_minutes,
        allowed_network_hosts: runtimeConfig.overrides.allowed_network_hosts,
        allowed_network_tools: runtimeConfig.overrides.allowed_network_tools,
        allowed_obsidian_paths: runtimeConfig.overrides.allowed_obsidian_paths,
        allowed_ha_operations: runtimeConfig.overrides.allowed_ha_operations,
        allowed_homelab_operations: runtimeConfig.overrides.allowed_homelab_operations,
      });
      setRuntimeConfig(updated);
      setStatus("Runtime config saved.");
    } catch (err) {
      setError(formatApiError(err, "Runtime config update failed"));
    } finally {
      setSavingRuntime(false);
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

      {loadError ? (
        <section className="surface stack auth-banner" aria-label="Runtime load warning">
          <h2>Setup issue detected</h2>
          <p className="help-text">{loadError}</p>
          <p className="help-text">Add a valid key below, then run Connection test to confirm access.</p>
        </section>
      ) : null}

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
          <button className="button button-muted" onClick={() => void testConnection()} disabled={testing}>
            {testing ? "Testing..." : "Connection test"}
          </button>
          {status ? <span className="pill success">{status}</span> : null}
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        <p className="help-text">
          Note: system-level configuration (Docker/env, MCP endpoints, allowlists) is still managed in `.env` and requires
          a backend restart today.
        </p>
      </section>

      <section className="surface stack" aria-label="MCP and tool readiness">
        <h2>MCP and tool readiness</h2>
        {!dependencies ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Dependency data unavailable.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Dependency</th>
                  <th>Status</th>
                  <th>Detail</th>
                  <th>Required</th>
                  <th>Enabled</th>
                </tr>
              </thead>
              <tbody>
                {dependencies.dependencies.map((dep) => (
                  <tr key={dep.name}>
                    <td className="mono">{dep.name}</td>
                    <td>
                      <span className={dep.ok ? "pill success" : dep.enabled ? "pill error" : "pill"}>
                        {dep.ok ? "ok" : "issue"}
                      </span>
                    </td>
                    <td>{dep.detail}</td>
                    <td>{dep.required ? "yes" : "no"}</td>
                    <td>{dep.enabled ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="help-text">
          MCP availability is controlled by `ENABLE_MCP_SERVICES` plus per-service env values (for example
          `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`, and vault mounts). Policy allowlists are enforced by backend
          `ALLOWED_*` settings.
        </p>
      </section>

      <section className="surface stack" aria-label="Runtime policy overrides">
        <h2>Admin runtime policy overrides</h2>
        {!runtimeConfig ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>
              Admin runtime config is unavailable for this key. Use an admin API key to view and edit these values.
            </p>
          </div>
        ) : (
          <>
            <p className="help-text">
              These overrides apply live and take precedence over `.env` for policy/runtime controls. Leave a field blank
              to use `.env` value.
            </p>
            <label className="field-label">
              Search mode default override
              <select
                className="select"
                value={runtimeConfig.overrides.search_mode_default ?? ""}
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            search_mode_default: e.target.value || null,
                          },
                        }
                      : prev,
                  )
                }
              >
                <option value="">(use .env)</option>
                {(options?.search_modes ?? ["searxng_only"]).map((mode) => (
                  <option key={mode} value={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </label>

            <label className="field-label">
              Sensitive actions override
              <select
                className="select"
                value={
                  runtimeConfig.overrides.sensitive_actions_enabled === null
                    ? ""
                    : runtimeConfig.overrides.sensitive_actions_enabled
                      ? "true"
                      : "false"
                }
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            sensitive_actions_enabled:
                              e.target.value === "" ? null : e.target.value === "true",
                          },
                        }
                      : prev,
                  )
                }
              >
                <option value="">(use .env)</option>
                <option value="true">enabled</option>
                <option value="false">disabled</option>
              </select>
            </label>

            <label className="field-label">
              Approval token TTL minutes override
              <input
                className="input"
                type="number"
                min={1}
                max={120}
                value={runtimeConfig.overrides.approval_token_ttl_minutes ?? ""}
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            approval_token_ttl_minutes:
                              e.target.value.trim() === "" ? null : Number(e.target.value),
                          },
                        }
                      : prev,
                  )
                }
                placeholder="leave blank to use .env"
              />
            </label>

            <label className="field-label">
              Allowed network hosts (csv)
              <input
                className="input mono"
                value={runtimeConfig.overrides.allowed_network_hosts ?? ""}
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            allowed_network_hosts: e.target.value.trim() === "" ? null : e.target.value,
                          },
                        }
                      : prev,
                  )
                }
                placeholder="host1,host2"
              />
            </label>

            <label className="field-label">
              Allowed network tools (csv)
              <input
                className="input mono"
                value={runtimeConfig.overrides.allowed_network_tools ?? ""}
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            allowed_network_tools: e.target.value.trim() === "" ? null : e.target.value,
                          },
                        }
                      : prev,
                  )
                }
                placeholder="network_probe"
              />
            </label>

            <label className="field-label">
              Allowed Obsidian paths (csv)
              <input
                className="input mono"
                value={runtimeConfig.overrides.allowed_obsidian_paths ?? ""}
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            allowed_obsidian_paths: e.target.value.trim() === "" ? null : e.target.value,
                          },
                        }
                      : prev,
                  )
                }
                placeholder="notes,projects"
              />
            </label>

            <label className="field-label">
              Allowed Home Assistant operations (csv)
              <input
                className="input mono"
                value={runtimeConfig.overrides.allowed_ha_operations ?? ""}
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            allowed_ha_operations: e.target.value.trim() === "" ? null : e.target.value,
                          },
                        }
                      : prev,
                  )
                }
                placeholder="light.turn_on,switch.turn_off"
              />
            </label>

            <label className="field-label">
              Allowed Homelab operations (csv)
              <input
                className="input mono"
                value={runtimeConfig.overrides.allowed_homelab_operations ?? ""}
                onChange={(e) =>
                  setRuntimeConfig((prev) =>
                    prev
                      ? {
                          ...prev,
                          overrides: {
                            ...prev.overrides,
                            allowed_homelab_operations: e.target.value.trim() === "" ? null : e.target.value,
                          },
                        }
                      : prev,
                  )
                }
                placeholder="dns.resolve,tcp.check,http.check"
              />
            </label>

            <div className="toolbar">
              <button className="button button-primary" onClick={() => void saveRuntimeConfig()} disabled={savingRuntime}>
                {savingRuntime ? "Saving runtime..." : "Save runtime policy"}
              </button>
              <span className="pill">
                Effective search mode: <span className="mono">{runtimeConfig.effective.search_mode_default}</span>
              </span>
              <span className="pill">
                Sensitive actions: {runtimeConfig.effective.sensitive_actions_enabled ? "enabled" : "disabled"}
              </span>
            </div>
            <p className="help-text">
              Last runtime config update:{" "}
              {runtimeConfig.overrides.updated_at ? new Date(runtimeConfig.overrides.updated_at).toLocaleString() : "never"} by{" "}
              {runtimeConfig.overrides.updated_by ?? "n/a"}
            </p>
          </>
        )}
      </section>
    </main>
  );
}
