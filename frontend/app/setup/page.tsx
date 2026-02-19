"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  ApiRequestError,
  DependencyStatus,
  formatApiError,
  getBrowserApiKey,
  getDependencyStatus,
  getOpsStatus,
  getRuntimeDiagnostics,
  getRuntimeConfig,
  getRuntimeOptionsTyped,
  OpsStatus,
  RuntimeDiagnostics,
  setBrowserApiKey,
} from "@/lib/api";

type AccessState = "unknown" | "missing_key" | "user" | "admin" | "invalid";

function depBadge(ok: boolean, enabled: boolean): string {
  if (!enabled) return "pill";
  return ok ? "pill success" : "pill error";
}

export default function SetupPage() {
  const [apiKey, setApiKey] = useState("");
  const [access, setAccess] = useState<AccessState>("unknown");
  const [runtimeStatus, setRuntimeStatus] = useState("");
  const [deps, setDeps] = useState<DependencyStatus | null>(null);
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics | null>(null);
  const [opsStatus, setOpsStatus] = useState<OpsStatus | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const checklist = (() => {
    const items: string[] = [];
    if (!apiKey.trim()) {
      items.push("Add a browser API key.");
    }
    if (access === "invalid") {
      items.push("Use a valid API key from ADMIN_API_KEYS or USER_API_KEYS.");
    }
    if (diagnostics) {
      if (!diagnostics.drop_folder_exists) {
        items.push(`Create/mount drop folder: ${diagnostics.drop_folder_path}`);
      }
      if (!diagnostics.obsidian_vault_exists) {
        items.push(`Create/mount Obsidian vault path: ${diagnostics.obsidian_vault_path}`);
      }
      if (diagnostics.enable_mcp_services && !diagnostics.mcp_obsidian_configured) {
        items.push("Set MCP_OBSIDIAN_URL or disable MCP services.");
      }
      if (diagnostics.enable_mcp_services && !diagnostics.mcp_ha_configured) {
        items.push("Set MCP_HA_URL for Home Assistant MCP.");
      }
      if (diagnostics.enable_mcp_services && !diagnostics.mcp_homelab_configured) {
        items.push("Set MCP_HOMELAB_URL for Homelab MCP.");
      }
    }
    if (deps) {
      for (const dep of deps.dependencies) {
        if (dep.required && dep.enabled && !dep.ok) {
          items.push(`Fix required dependency '${dep.name}' (${dep.detail}).`);
        }
      }
    }
    if (opsStatus && opsStatus.status !== "ready") {
      items.push("Run POST /ops/init and resolve degraded dependencies in ops status.");
    }
    return items;
  })();

  useEffect(() => {
    setApiKey(getBrowserApiKey());
  }, []);

  async function verify() {
    setLoading(true);
    setError("");
    setRuntimeStatus("");
    setDeps(null);
    setDiagnostics(null);
    setOpsStatus(null);

    try {
      setBrowserApiKey(apiKey);
      if (!apiKey.trim()) {
        setAccess("missing_key");
        setRuntimeStatus("Add an API key to continue.");
        return;
      }

      await getRuntimeOptionsTyped();
      setRuntimeStatus("Runtime API reachable.");

      let role: AccessState = "user";
      try {
        await getRuntimeConfig();
        setAccess("admin");
        role = "admin";
      } catch (err) {
        if (err instanceof ApiRequestError && err.status === 403) {
          setAccess("user");
          role = "user";
        } else {
          throw err;
        }
      }

      const dependencyStatus = await getDependencyStatus();
      setDeps(dependencyStatus);
      setDiagnostics(await getRuntimeDiagnostics());
      if (role === "admin") {
        try {
          setOpsStatus(await getOpsStatus());
        } catch (err) {
          if (!(err instanceof ApiRequestError && err.status === 403)) {
            throw err;
          }
        }
      }
    } catch (err) {
      setAccess("invalid");
      setError(formatApiError(err, "Setup verification failed"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">First-Run Setup</p>
        <h1 className="page-title">Connect auth, verify runtime, and confirm capabilities.</h1>
        <p className="page-description">
          Use this guided flow to validate your key, identify your role, and check whether dependencies and MCP services are ready.
        </p>
      </section>

      <section className="surface stack" aria-label="Setup authentication">
        <h2>1. API key</h2>
        <label className="field-label">
          Browser API key
          <input
            className="input"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Paste user or admin API key"
          />
        </label>
        <div className="toolbar">
          <button type="button" className="button button-primary" onClick={() => void verify()} disabled={loading}>
            {loading ? "Verifying..." : "Save and verify"}
          </button>
          <span className="pill">{runtimeStatus || "not verified"}</span>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="surface stack" aria-label="Setup role">
        <h2>2. Access level</h2>
        <div className="toolbar">
          <span className={access === "admin" ? "pill success" : access === "invalid" ? "pill error" : "pill"}>
            {access}
          </span>
          {access === "admin" ? <span className="help-text">Admin runtime policy controls are available.</span> : null}
          {access === "user" ? <span className="help-text">User mode active. Admin-only settings are read-only.</span> : null}
        </div>
      </section>

      <section className="surface stack" aria-label="Setup dependencies">
        <h2>3. Dependency readiness</h2>
        {!deps ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Run verification to load dependency status.</p>
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
                </tr>
              </thead>
              <tbody>
                {deps.dependencies.map((dep) => (
                  <tr key={dep.name}>
                    <td className="mono">{dep.name}</td>
                    <td>
                      <span className={depBadge(dep.ok, dep.enabled)}>{dep.ok ? "ok" : dep.enabled ? "issue" : "disabled"}</span>
                    </td>
                    <td>{dep.detail}</td>
                    <td>{dep.required ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="surface stack" aria-label="Runtime diagnostics">
        <h2>4. Runtime diagnostics</h2>
        {!diagnostics ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Run verification to load runtime diagnostics.</p>
          </div>
        ) : (
          <div className="stats-grid">
            <article className="stat-card">
              <p className="stat-label">Role</p>
              <p className="stat-value">{diagnostics.role}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Auth required</p>
              <p className="stat-value">{String(diagnostics.require_auth)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">MCP Services Enabled</p>
              <p className="stat-value">{String(diagnostics.enable_mcp_services)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Obsidian MCP Configured</p>
              <p className="stat-value">{String(diagnostics.mcp_obsidian_configured)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">HA MCP Configured</p>
              <p className="stat-value">{String(diagnostics.mcp_ha_configured)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Homelab MCP Configured</p>
              <p className="stat-value">{String(diagnostics.mcp_homelab_configured)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Drop folder exists</p>
              <p className="stat-value mono">{String(diagnostics.drop_folder_exists)} ({diagnostics.drop_folder_path})</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Vault path exists</p>
              <p className="stat-value mono">{String(diagnostics.obsidian_vault_exists)} ({diagnostics.obsidian_vault_path})</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Effective search mode</p>
              <p className="stat-value mono">{diagnostics.effective_search_mode_default}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Sensitive actions</p>
              <p className="stat-value">{String(diagnostics.effective_sensitive_actions_enabled)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Approval TTL</p>
              <p className="stat-value">{diagnostics.effective_approval_token_ttl_minutes} min</p>
            </article>
          </div>
        )}
      </section>

      <section className="surface stack" aria-label="Admin ops readiness">
        <h2>5. Admin ops readiness</h2>
        {!opsStatus ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Ops status is available for admin keys only.</p>
          </div>
        ) : (
          <>
            <div className="toolbar">
              <span className={opsStatus.status === "ready" ? "pill success" : "pill error"}>ops status: {opsStatus.status}</span>
              <span className="pill mono">
                app {opsStatus.version.app_version} ({opsStatus.version.environment})
              </span>
            </div>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Init step</th>
                    <th>Ready</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(opsStatus.init_steps).map(([step, ok]) => (
                    <tr key={step}>
                      <td className="mono">{step}</td>
                      <td>
                        <span className={ok ? "pill success" : "pill error"}>{ok ? "yes" : "no"}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      <section className="surface stack" aria-label="Setup checklist">
        <h2>6. Remediation checklist</h2>
        {checklist.length === 0 ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>No blockers detected. You can continue to Settings and Chat.</p>
          </div>
        ) : (
          <ul className="list-reset">
            {checklist.map((item, idx) => (
              <li key={`check-${idx}`}>{item}</li>
            ))}
          </ul>
        )}
      </section>

      <section className="surface stack" aria-label="Next steps">
        <h2>7. Continue</h2>
        <div className="quick-links">
          <Link href="/settings" className="quick-link">
            <p className="quick-link-title">Settings</p>
            <p className="quick-link-copy">Tune user model defaults and runtime policy overrides.</p>
          </Link>
          <Link href="/capabilities" className="quick-link">
            <p className="quick-link-title">Capabilities</p>
            <p className="quick-link-copy">Review tool registry and MCP/dependency state.</p>
          </Link>
          <Link href="/chat" className="quick-link">
            <p className="quick-link-title">Chat</p>
            <p className="quick-link-copy">Run a first message with model and RAG visibility.</p>
          </Link>
        </div>
      </section>
    </main>
  );
}
