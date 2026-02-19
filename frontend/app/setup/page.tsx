"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import {
  ApiRequestError,
  DependencyStatus,
  formatApiError,
  getBrowserApiKey,
  getDependencyStatus,
  getRuntimeConfig,
  getRuntimeOptionsTyped,
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
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setApiKey(getBrowserApiKey());
  }, []);

  async function verify() {
    setLoading(true);
    setError("");
    setRuntimeStatus("");
    setDeps(null);

    try {
      setBrowserApiKey(apiKey);
      if (!apiKey.trim()) {
        setAccess("missing_key");
        setRuntimeStatus("Add an API key to continue.");
        return;
      }

      await getRuntimeOptionsTyped();
      setRuntimeStatus("Runtime API reachable.");

      try {
        await getRuntimeConfig();
        setAccess("admin");
      } catch (err) {
        if (err instanceof ApiRequestError && err.status === 403) {
          setAccess("user");
        } else {
          throw err;
        }
      }

      const dependencyStatus = await getDependencyStatus();
      setDeps(dependencyStatus);
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

      <section className="surface stack" aria-label="Next steps">
        <h2>4. Continue</h2>
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
