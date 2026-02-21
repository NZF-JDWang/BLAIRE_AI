import Link from "next/link";

import { getDependencyStatus, getHealth, getRuntimeOptions } from "@/lib/api";

export default async function HomePage() {
  let backendStatus = "unreachable";
  let defaultSearchMode = "unknown";
  let generalModels = "unknown";
  let sensitiveActions = "unknown";
  let dependencySummary = "unknown";

  try {
    const health = (await getHealth()) as { status?: string };
    backendStatus = health.status ?? "unknown";
  } catch {
    backendStatus = "unreachable";
  }

  try {
    const options = (await getRuntimeOptions()) as {
      default_search_mode?: string;
      model_allowlist?: Record<string, string[]>;
      sensitive_actions_enabled?: boolean;
    };
    defaultSearchMode = options.default_search_mode ?? "unknown";
    generalModels = options.model_allowlist?.general?.join(", ") ?? "none";
    sensitiveActions = String(options.sensitive_actions_enabled ?? "unknown");
  } catch {
    defaultSearchMode = "unavailable";
    generalModels = "unavailable";
    sensitiveActions = "unavailable";
  }

  try {
    const deps = await getDependencyStatus();
    const ok = deps.dependencies.filter((d) => d.ok).length;
    dependencySummary = `${ok}/${deps.dependencies.length} healthy`;
  } catch {
    dependencySummary = "unavailable";
  }

  return (
    <main className="page-wrap">
      <section className="page-hero">
        <p className="page-kicker">Operations</p>
        <h1 className="page-title">Run work from one launcher.</h1>
        <p className="page-description">
          Start with Settings, then open the workspace you need.
        </p>
      </section>

      <section className="stats-grid" aria-label="System overview">
        <article className="stat-card">
          <p className="stat-label">Backend</p>
          <p className="stat-value">{backendStatus}</p>
        </article>
        <article className="stat-card">
          <p className="stat-label">Search Mode</p>
          <p className="stat-value mono">{defaultSearchMode}</p>
        </article>
        <article className="stat-card">
          <p className="stat-label">Dependencies</p>
          <p className="stat-value mono">{dependencySummary}</p>
        </article>
        <article className="stat-card">
          <p className="stat-label">General Models</p>
          <p className="stat-value mono">{generalModels}</p>
        </article>
        <article className="stat-card">
          <p className="stat-label">Sensitive Actions</p>
          <p className="stat-value">{sensitiveActions}</p>
        </article>
      </section>

      <section className="surface stack" aria-label="Primary routes">
        <h2>Start here</h2>
        <div className="quick-links">
          <Link href="/settings" className="quick-link">
            <p className="quick-link-title">Settings</p>
            <p className="quick-link-copy">Run first-time setup, pull local models, and set your defaults.</p>
          </Link>
          <Link href="/chat" className="quick-link">
            <p className="quick-link-title">Chat</p>
            <p className="quick-link-copy">Everyday assistant work with model and citation visibility.</p>
          </Link>
          <Link href="/knowledge" className="quick-link">
            <p className="quick-link-title">Knowledge</p>
            <p className="quick-link-copy">Upload files and manage indexing.</p>
          </Link>
        </div>
      </section>

      <section className="surface stack" aria-label="Additional routes">
        <h2>More workspaces</h2>
        <div className="quick-links">
          <Link href="/swarm" className="quick-link">
            <p className="quick-link-title">Swarm</p>
            <p className="quick-link-copy">Parallel multi-agent research and trace review.</p>
          </Link>
          <Link href="/approvals" className="quick-link">
            <p className="quick-link-title">Approvals</p>
            <p className="quick-link-copy">Review and execute approval-gated actions.</p>
          </Link>
          <Link href="/search" className="quick-link">
            <p className="quick-link-title">Search</p>
            <p className="quick-link-copy">Search provider diagnostics and mode testing.</p>
          </Link>
          <Link href="/capabilities" className="quick-link">
            <p className="quick-link-title">Capabilities</p>
            <p className="quick-link-copy">Inspect runtime capability and dependency state.</p>
          </Link>
          <Link href="/tools" className="quick-link">
            <p className="quick-link-title">Tools</p>
            <p className="quick-link-copy">Inspect tool contracts and approval requirements.</p>
          </Link>
        </div>
      </section>
    </main>
  );
}
