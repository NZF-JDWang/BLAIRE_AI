"use client";

import { useEffect, useState } from "react";

import {
  ApiRequestError,
  DependencyStatus,
  formatApiError,
  getBrowserApiKey,
  getDependencyStatus,
  getIntegrationsStatus,
  getRuntimeConfigAudit,
  getMyPreferences,
  getModelsInfo,
  getRuntimeConfig,
  getRuntimeSystemSummary,
  getRuntimeOptionsTyped,
  IntegrationsStatus,
  RuntimeConfigAuditEvent,
  RuntimeConfigBundle,
  ModelsInfo,
  RuntimeOptions,
  RuntimeSystemSummary,
  setBrowserApiKey,
  pullModel,
  updateRuntimeConfig,
  updateMyPreferences,
} from "@/lib/api";

type Preferences = {
  searchMode: string;
  modelClass: string;
  modelOverride: string;
  temperature: number;
  topP: number;
  maxTokens: string;
  contextWindowTokens: string;
  useRag: boolean;
  retrievalK: number;
};

type SettingsTab = "identity" | "model" | "runtime" | "readiness" | "integrations";

export default function SettingsPage() {
  const [options, setOptions] = useState<RuntimeOptions | null>(null);
  const [dependencies, setDependencies] = useState<DependencyStatus | null>(null);
  const [modelsInfo, setModelsInfo] = useState<ModelsInfo | null>(null);
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfigBundle | null>(null);
  const [runtimeAudit, setRuntimeAudit] = useState<RuntimeConfigAuditEvent[]>([]);
  const [systemSummary, setSystemSummary] = useState<RuntimeSystemSummary | null>(null);
  const [integrationsStatus, setIntegrationsStatus] = useState<IntegrationsStatus | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [prefs, setPrefs] = useState<Preferences>({
    searchMode: "searxng_only",
    modelClass: "general",
    modelOverride: "",
    temperature: 0.7,
    topP: 1.0,
    maxTokens: "",
    contextWindowTokens: "",
    useRag: true,
    retrievalK: 4,
  });
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [saving, setSaving] = useState(false);
  const [savingRuntime, setSavingRuntime] = useState(false);
  const [testing, setTesting] = useState(false);
  const [pullingModel, setPullingModel] = useState(false);
  const [modelPullName, setModelPullName] = useState("");
  const [activeTab, setActiveTab] = useState<SettingsTab>("identity");

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
      getRuntimeConfigAudit(30).catch((err) => {
        if (err instanceof ApiRequestError && err.status === 403) {
          return [];
        }
        throw err;
      }),
      getModelsInfo().catch(() => null),
      getRuntimeSystemSummary().catch((err) => {
        if (err instanceof ApiRequestError && err.status === 403) {
          return null;
        }
        throw err;
      }),
      getIntegrationsStatus().catch((err) => {
        if (err instanceof ApiRequestError && err.status === 403) {
          return null;
        }
        throw err;
      }),
    ])
      .then(([runtime, current, deps, config, audit, models, summary, integrations]) => {
        setOptions(runtime);
        setModelsInfo(models);
        setDependencies(deps);
        setRuntimeConfig(config);
        setRuntimeAudit(audit);
        setSystemSummary(summary);
        setIntegrationsStatus(integrations);
        setLoadError("");
        setPrefs({
          searchMode: current.search_mode,
          modelClass: current.model_class,
          modelOverride: current.model_override ?? "",
          temperature: current.temperature,
          topP: current.top_p,
          maxTokens: current.max_tokens ? String(current.max_tokens) : "",
          contextWindowTokens: current.context_window_tokens ? String(current.context_window_tokens) : "",
          useRag: current.use_rag,
          retrievalK: current.retrieval_k,
        });
      })
      .catch((err) => {
        setOptions(null);
        setDependencies(null);
        setModelsInfo(null);
        setRuntimeConfig(null);
        setRuntimeAudit([]);
        setSystemSummary(null);
        setIntegrationsStatus(null);
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
        temperature: prefs.temperature,
        top_p: prefs.topP,
        max_tokens: prefs.maxTokens.trim() ? Number(prefs.maxTokens) : null,
        context_window_tokens: prefs.contextWindowTokens.trim() ? Number(prefs.contextWindowTokens) : null,
        use_rag: prefs.useRag,
        retrieval_k: prefs.retrievalK,
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
      setRuntimeAudit(await getRuntimeConfigAudit(30));
      setStatus("Runtime config saved.");
    } catch (err) {
      setError(formatApiError(err, "Runtime config update failed"));
    } finally {
      setSavingRuntime(false);
    }
  }

  async function requestModelPull() {
    if (!modelPullName.trim()) {
      setError("Model name is required for pull.");
      return;
    }
    setPullingModel(true);
    setError("");
    setStatus("");
    try {
      const result = await pullModel(modelPullName.trim());
      setStatus(`${result.status}: ${result.model_name}`);
      setModelsInfo(await getModelsInfo());
    } catch (err) {
      setError(formatApiError(err, "Model pull failed"));
    } finally {
      setPullingModel(false);
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

      <section className="surface stack" aria-label="Settings sections">
        <div className="toolbar">
          <button className={activeTab === "identity" ? "button button-primary" : "button button-muted"} onClick={() => setActiveTab("identity")}>
            Identity
          </button>
          <button className={activeTab === "model" ? "button button-primary" : "button button-muted"} onClick={() => setActiveTab("model")}>
            Model controls
          </button>
          <button className={activeTab === "runtime" ? "button button-primary" : "button button-muted"} onClick={() => setActiveTab("runtime")}>
            Runtime policy
          </button>
          <button className={activeTab === "readiness" ? "button button-primary" : "button button-muted"} onClick={() => setActiveTab("readiness")}>
            MCP readiness
          </button>
          <button className={activeTab === "integrations" ? "button button-primary" : "button button-muted"} onClick={() => setActiveTab("integrations")}>
            Integrations
          </button>
        </div>
      </section>

      {loadError ? (
        <section className="surface stack auth-banner" aria-label="Runtime load warning">
          <h2>Setup issue detected</h2>
          <p className="help-text">{loadError}</p>
          <p className="help-text">Add a valid key below, then run Connection test to confirm access.</p>
        </section>
      ) : null}

      {activeTab === "identity" ? (
      <section className="surface stack" aria-label="Identity and access">
        <h2>Identity and access</h2>
        <label className="field-label">
          API key
          <input
            className="input"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Paste your user API key"
          />
        </label>
        <p className="help-text">
          Use `Connection test` to confirm key validity before changing other settings.
        </p>
      </section>
      ) : null}

      {activeTab === "model" ? (
      <section className="surface stack" aria-label="Model and retrieval controls">
        <h2>Model and retrieval controls</h2>
        {modelsInfo ? (
          <>
            <p className="help-text">
              Installed models: {modelsInfo.installed_models.length} | allow-any-inference:{" "}
              {String(modelsInfo.model_allow_any_inference)}
            </p>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Default</th>
                    <th>Allowlist count</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(modelsInfo.defaults).map(([modelClass, defaultModel]) => (
                    <tr key={`model-class-${modelClass}`}>
                      <td className="mono">{modelClass}</td>
                      <td className="mono">{defaultModel ?? "(none)"}</td>
                      <td>{modelsInfo.allowlist[modelClass]?.length ?? 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="toolbar">
              <input
                className="input mono"
                value={modelPullName}
                onChange={(e) => setModelPullName(e.target.value)}
                placeholder="model to pull (admin)"
                style={{ maxWidth: "320px" }}
              />
              <button className="button button-muted" onClick={() => void requestModelPull()} disabled={pullingModel}>
                {pullingModel ? "Pulling..." : "Pull model"}
              </button>
            </div>
          </>
        ) : null}

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

        <label className="field-label">
          Temperature
          <input
            className="input"
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={prefs.temperature}
            onChange={(e) => setPrefs((prev) => ({ ...prev, temperature: Number(e.target.value) }))}
          />
        </label>

        <label className="field-label">
          Top P
          <input
            className="input"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={prefs.topP}
            onChange={(e) => setPrefs((prev) => ({ ...prev, topP: Number(e.target.value) }))}
          />
        </label>

        <label className="field-label">
          Max tokens
          <input
            className="input"
            type="number"
            min={1}
            max={8192}
            value={prefs.maxTokens}
            onChange={(e) => setPrefs((prev) => ({ ...prev, maxTokens: e.target.value }))}
            placeholder="blank = provider default"
          />
        </label>

        <label className="field-label">
          Context window tokens
          <input
            className="input"
            type="number"
            min={256}
            max={262144}
            value={prefs.contextWindowTokens}
            onChange={(e) => setPrefs((prev) => ({ ...prev, contextWindowTokens: e.target.value }))}
            placeholder="blank = provider default"
          />
        </label>

        <label className="field-label">
          Default retrieval K
          <input
            className="input"
            type="number"
            min={1}
            max={12}
            value={prefs.retrievalK}
            onChange={(e) => setPrefs((prev) => ({ ...prev, retrievalK: Number(e.target.value) }))}
          />
        </label>

        <label className="field-label">
          Default RAG behavior
          <select
            className="select"
            value={prefs.useRag ? "enabled" : "disabled"}
            onChange={(e) => setPrefs((prev) => ({ ...prev, useRag: e.target.value === "enabled" }))}
          >
            <option value="enabled">enabled</option>
            <option value="disabled">disabled</option>
          </select>
        </label>
      </section>
      ) : null}

      {activeTab === "identity" || activeTab === "model" ? (
      <section className="surface stack" aria-label="Save user preferences">
        <h2>Save user preferences</h2>
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
      ) : null}

      {activeTab === "readiness" ? (
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
      ) : null}

      {activeTab === "runtime" ? (
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

            <h3>Runtime policy audit</h3>
            {runtimeAudit.length === 0 ? (
              <div className="empty-state">
                <p style={{ margin: 0 }}>No runtime config audit events yet.</p>
              </div>
            ) : (
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Actor</th>
                      <th>Before</th>
                      <th>After</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runtimeAudit.slice(0, 10).map((event) => (
                      <tr key={event.id}>
                        <td>{new Date(event.event_time).toLocaleString()}</td>
                        <td>{event.actor}</td>
                        <td className="mono">{event.previous_overrides.search_mode_default ?? "(none)"}</td>
                        <td className="mono">{event.new_overrides.search_mode_default ?? "(none)"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <h3>System config summary (restart required)</h3>
            {!systemSummary ? (
              <div className="empty-state">
                <p style={{ margin: 0 }}>System summary unavailable for this key.</p>
              </div>
            ) : (
              <>
                <p className="help-text">{systemSummary.restart_required_note}</p>
                <div className="table-wrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Setting</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="mono">app_env</td>
                        <td>{systemSummary.app_env}</td>
                      </tr>
                      <tr>
                        <td className="mono">enable_mcp_services</td>
                        <td>{String(systemSummary.enable_mcp_services)}</td>
                      </tr>
                      <tr>
                        <td className="mono">enable_vllm</td>
                        <td>{String(systemSummary.enable_vllm)}</td>
                      </tr>
                      <tr>
                        <td className="mono">inference_base_url</td>
                        <td className="mono">{systemSummary.inference_base_url}</td>
                      </tr>
                      <tr>
                        <td className="mono">qdrant_url</td>
                        <td className="mono">{systemSummary.qdrant_url}</td>
                      </tr>
                      <tr>
                        <td className="mono">searxng_url</td>
                        <td className="mono">{systemSummary.searxng_url}</td>
                      </tr>
                      <tr>
                        <td className="mono">mcp_obsidian_url</td>
                        <td className="mono">{systemSummary.mcp_obsidian_url}</td>
                      </tr>
                      <tr>
                        <td className="mono">mcp_ha_url</td>
                        <td className="mono">{systemSummary.mcp_ha_url}</td>
                      </tr>
                      <tr>
                        <td className="mono">mcp_homelab_url</td>
                        <td className="mono">{systemSummary.mcp_homelab_url}</td>
                      </tr>
                      <tr>
                        <td className="mono">drop_folder</td>
                        <td className="mono">{systemSummary.drop_folder}</td>
                      </tr>
                      <tr>
                        <td className="mono">obsidian_vault_path</td>
                        <td className="mono">{systemSummary.obsidian_vault_path}</td>
                      </tr>
                      <tr>
                        <td className="mono">brave_api_key_configured</td>
                        <td>{String(systemSummary.brave_api_key_configured)}</td>
                      </tr>
                      <tr>
                        <td className="mono">telegram_configured</td>
                        <td>{String(systemSummary.telegram_configured)}</td>
                      </tr>
                      <tr>
                        <td className="mono">google_oauth_configured</td>
                        <td>{String(systemSummary.google_oauth_configured)}</td>
                      </tr>
                      <tr>
                        <td className="mono">imap_configured</td>
                        <td>{String(systemSummary.imap_configured)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </>
        )}
      </section>
      ) : null}

      {activeTab === "integrations" ? (
      <section className="surface stack" aria-label="Integrations status">
        <h2>Integrations status</h2>
        {!integrationsStatus ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Integrations status unavailable for this key.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Integration</th>
                  <th>Configured</th>
                  <th>Endpoint/Host</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Google OAuth</td>
                  <td>{String(integrationsStatus.google_oauth_configured)}</td>
                  <td className="mono">{integrationsStatus.google_api_base}</td>
                </tr>
                <tr>
                  <td>IMAP</td>
                  <td>{String(integrationsStatus.imap_configured)}</td>
                  <td className="mono">{integrationsStatus.imap_host || "(unset)"}</td>
                </tr>
                <tr>
                  <td>Home Assistant</td>
                  <td>{String(integrationsStatus.home_assistant_configured)}</td>
                  <td className="mono">{integrationsStatus.home_assistant_url}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
        <p className="help-text">
          Tokens/secrets are intentionally not returned by the API. Configure credentials in `.env` (restart required),
          then verify here.
        </p>
      </section>
      ) : null}
    </main>
  );
}
