"use client";

import { useEffect, useMemo, useState } from "react";

import {
  DependencyStatus,
  formatApiError,
  getDependencyStatus,
  getRuntimeDiagnostics,
  getRuntimeOptionsTyped,
  RuntimeDiagnostics,
  RuntimeOptions,
} from "@/lib/api";

function dependencyClass(ok: boolean, enabled: boolean): string {
  if (!enabled) return "pill";
  return ok ? "pill success" : "pill error";
}

export default function CapabilitiesPage() {
  const [runtime, setRuntime] = useState<RuntimeOptions | null>(null);
  const [diagnostics, setDiagnostics] = useState<RuntimeDiagnostics | null>(null);
  const [deps, setDeps] = useState<DependencyStatus | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const [runtimeOptions, dependencyStatus, runtimeDiagnostics] = await Promise.all([
        getRuntimeOptionsTyped(),
        getDependencyStatus(),
        getRuntimeDiagnostics(),
      ]);
      setRuntime(runtimeOptions);
      setDeps(dependencyStatus);
      setDiagnostics(runtimeDiagnostics);
    } catch (err) {
      setError(formatApiError(err, "Failed to load capabilities"));
      setRuntime(null);
      setDiagnostics(null);
      setDeps(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const toolRows = useMemo(() => runtime?.tools ?? [], [runtime]);
  const dependencyRows = useMemo(() => deps?.dependencies ?? [], [deps]);
  const failedDependencies = useMemo(() => dependencyRows.filter((item) => item.enabled && !item.ok), [dependencyRows]);
  const mcpIssues = useMemo(
    () => dependencyRows.filter((item) => ["mcp_obsidian", "mcp_home_assistant", "mcp_homelab"].includes(item.name) && !item.ok && item.enabled),
    [dependencyRows],
  );

  function dependencyHint(name: string): string {
    if (name === "mcp_obsidian") return "Check ENABLE_MCP_SERVICES, MCP_OBSIDIAN_URL, and vault mount/path.";
    if (name === "mcp_home_assistant") return "Check ENABLE_MCP_SERVICES plus HOME_ASSISTANT_URL/TOKEN on ha-mcp.";
    if (name === "mcp_homelab") return "Check ENABLE_MCP_SERVICES and homelab-mcp container health.";
    if (name === "searxng") return "Check SEARXNG_URL reachability or switch search mode.";
    if (name === "brave_api_key") return "Add BRAVE_API_KEY or use searxng_only mode.";
    if (name === "inference_api") return "Check INFERENCE_BASE_URL and model runtime availability.";
    if (name === "qdrant") return "Check QDRANT_URL and container/network health.";
    return "Review service logs and runtime config.";
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Runtime Capabilities</p>
        <h1 className="page-title">See what tools and MCP services are available right now.</h1>
        <p className="page-description">
          This page merges runtime tool policy with dependency health checks so you can quickly see what is ready,
          disabled, or blocked.
        </p>
      </section>

      <section className="surface stack">
        <div className="toolbar">
          <button type="button" className="button button-primary" onClick={() => void load()} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh capabilities"}
          </button>
          {runtime ? <span className="pill success">runtime options loaded</span> : null}
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="surface stack">
        <h2>MCP guidance</h2>
        {!diagnostics ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Runtime diagnostics unavailable.</p>
          </div>
        ) : (
          <>
            {!diagnostics.enable_mcp_services ? (
              <div className="empty-state">
                <p style={{ margin: 0 }}>
                  MCP services are disabled. Enable `ENABLE_MCP_SERVICES=true` if you want Obsidian/Home Assistant/Homelab actions.
                </p>
              </div>
            ) : mcpIssues.length === 0 ? (
              <div className="empty-state">
                <p style={{ margin: 0 }}>MCP services are enabled and currently healthy.</p>
              </div>
            ) : (
              <div className="panel-list">
                {mcpIssues.map((item) => (
                  <article key={`mcp-${item.name}`} className="surface" style={{ padding: "12px" }}>
                    <p style={{ marginBottom: "6px" }}>
                      <strong>{item.name}</strong>
                    </p>
                    <p className="help-text">{item.detail}</p>
                    <p style={{ marginBottom: 0 }}>{dependencyHint(item.name)}</p>
                  </article>
                ))}
              </div>
            )}
          </>
        )}
      </section>

      <section className="surface stack">
        <h2>Policy snapshot</h2>
        {!runtime ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Runtime policy unavailable.</p>
          </div>
        ) : (
          <div className="stats-grid">
            <article className="stat-card">
              <p className="stat-label">Default search mode</p>
              <p className="stat-value mono">{runtime.default_search_mode}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Sensitive actions</p>
              <p className="stat-value">{String(runtime.sensitive_actions_enabled)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Approval TTL (minutes)</p>
              <p className="stat-value">{runtime.approval_token_ttl_minutes}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Allowed network hosts</p>
              <p className="stat-value mono">{runtime.allowed_network_hosts.join(", ") || "(any)"}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Allowed network tools</p>
              <p className="stat-value mono">{runtime.allowed_network_tools.join(", ") || "(any)"}</p>
            </article>
          </div>
        )}
      </section>

      <section className="surface stack">
        <h2>Actionable issues</h2>
        {failedDependencies.length === 0 ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>No enabled dependency failures detected.</p>
          </div>
        ) : (
          <div className="panel-list">
            {failedDependencies.map((item) => (
              <article key={`issue-${item.name}`} className="surface" style={{ padding: "12px" }}>
                <p style={{ marginBottom: "6px" }}>
                  <strong>{item.name}</strong>
                </p>
                <p className="help-text">{item.detail}</p>
                <p style={{ marginBottom: 0 }}>{dependencyHint(item.name)}</p>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="surface stack">
        <h2>Tool registry</h2>
        {!runtime ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Tool data unavailable.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Tool</th>
                  <th>Action class</th>
                  <th>Target required</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {toolRows.map((tool) => (
                  <tr key={tool.name}>
                    <td className="mono">{tool.name}</td>
                    <td>{tool.action_class}</td>
                    <td>{tool.requires_target_host ? "yes" : "no"}</td>
                    <td>{tool.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="surface stack">
        <h2>Dependencies and MCP connectivity</h2>
        {!deps ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Dependency data unavailable.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Detail</th>
                  <th>Required</th>
                  <th>Enabled</th>
                </tr>
              </thead>
              <tbody>
                {dependencyRows.map((item) => (
                  <tr key={item.name}>
                    <td className="mono">{item.name}</td>
                    <td>
                      <span className={dependencyClass(item.ok, item.enabled)}>{item.ok ? "ok" : "issue"}</span>
                    </td>
                    <td>{item.detail}</td>
                    <td>{item.required ? "yes" : "no"}</td>
                    <td>{item.enabled ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
