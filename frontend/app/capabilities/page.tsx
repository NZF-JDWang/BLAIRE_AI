"use client";

import { useEffect, useMemo, useState } from "react";

import { DependencyStatus, formatApiError, getDependencyStatus, getRuntimeOptionsTyped, RuntimeOptions } from "@/lib/api";

function dependencyClass(ok: boolean, enabled: boolean): string {
  if (!enabled) return "pill";
  return ok ? "pill success" : "pill error";
}

export default function CapabilitiesPage() {
  const [runtime, setRuntime] = useState<RuntimeOptions | null>(null);
  const [deps, setDeps] = useState<DependencyStatus | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError("");
    try {
      const [runtimeOptions, dependencyStatus] = await Promise.all([getRuntimeOptionsTyped(), getDependencyStatus()]);
      setRuntime(runtimeOptions);
      setDeps(dependencyStatus);
    } catch (err) {
      setError(formatApiError(err, "Failed to load capabilities"));
      setRuntime(null);
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
