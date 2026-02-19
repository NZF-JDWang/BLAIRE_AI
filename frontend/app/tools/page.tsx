"use client";

import { useEffect, useMemo, useState } from "react";

import { formatApiError, getRuntimeOptionsTyped, RuntimeOptions } from "@/lib/api";

function actionClassBadge(actionClass: string): string {
  if (actionClass === "local_safe") return "pill success";
  if (actionClass === "network_sensitive" || actionClass === "local_sensitive") return "pill warn";
  return "pill";
}

export default function ToolsPage() {
  const [runtime, setRuntime] = useState<RuntimeOptions | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError("");
    try {
      setRuntime(await getRuntimeOptionsTyped());
    } catch (err) {
      setRuntime(null);
      setError(formatApiError(err, "Failed to load tools"));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const toolRows = useMemo(() => runtime?.tools ?? [], [runtime]);

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Tool Workspace</p>
        <h1 className="page-title">Inspect available tools and execution requirements.</h1>
        <p className="page-description">
          Use this page to understand what tool actions are enabled, whether approvals are needed, and what host/tool
          allowlists are active.
        </p>
      </section>

      <section className="surface stack" aria-label="Tool policy summary">
        <div className="toolbar">
          <button type="button" className="button button-primary" onClick={() => void load()} disabled={loading}>
            {loading ? "Refreshing..." : "Refresh tools"}
          </button>
          {runtime ? <span className="pill success">{toolRows.length} tools loaded</span> : null}
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        {runtime ? (
          <div className="stats-grid">
            <article className="stat-card">
              <p className="stat-label">Sensitive actions</p>
              <p className="stat-value">{String(runtime.sensitive_actions_enabled)}</p>
            </article>
            <article className="stat-card">
              <p className="stat-label">Approval TTL</p>
              <p className="stat-value">{runtime.approval_token_ttl_minutes} minutes</p>
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
        ) : null}
      </section>

      <section className="surface stack" aria-label="Tool list">
        <h2>Registered tools</h2>
        {!runtime ? (
          <div className="empty-state">
            <p style={{ margin: 0 }}>Tool data unavailable.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Action class</th>
                  <th>Approval required</th>
                  <th>Target host required</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {toolRows.map((tool) => (
                  <tr key={tool.name}>
                    <td className="mono">{tool.name}</td>
                    <td>
                      <span className={actionClassBadge(tool.action_class)}>{tool.action_class}</span>
                    </td>
                    <td>{tool.action_class === "network_sensitive" ? "yes" : "depends on policy"}</td>
                    <td>{tool.requires_target_host ? "yes" : "no"}</td>
                    <td>{tool.description}</td>
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
