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
        <h1 className="page-title">Run work in one place, with status visible before you start.</h1>
        <p className="page-description">
          This page is your operational launcher. Pick the task you want to run, open the matching workspace, and use
          the status cards below to confirm runtime readiness first.
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
        <h2>Choose an operation</h2>
        <p className="help-text">
          Workspaces stay separate because they trigger different backend systems, but this screen is the single place
          to choose and understand them.
        </p>
        <div className="quick-links">
          <Link href="/setup" className="quick-link">
            <p className="quick-link-title">Setup</p>
            <p className="quick-link-copy">
              First-time only. Add your API key, confirm your role, and verify dependency health.
            </p>
          </Link>
          <Link href="/chat" className="quick-link">
            <p className="quick-link-title">Chat</p>
            <p className="quick-link-copy">
              Everyday assistant work. Ask questions, stream responses, and inspect retrieval citations.
            </p>
          </Link>
          <Link href="/swarm" className="quick-link">
            <p className="quick-link-title">Swarm</p>
            <p className="quick-link-copy">
              Parallel research mode. Run multi-agent tasks and review supervisor plus worker traces.
            </p>
          </Link>
          <Link href="/knowledge" className="quick-link">
            <p className="quick-link-title">Knowledge</p>
            <p className="quick-link-copy">
              Corpus maintenance. Upload files, refresh index status, and run Obsidian reindex operations.
            </p>
          </Link>
          <Link href="/approvals" className="quick-link">
            <p className="quick-link-title">Approvals</p>
            <p className="quick-link-copy">
              Safety checkpoint. Review pending actions, issue approval tokens, and execute audited operations.
            </p>
          </Link>
          <Link href="/search" className="quick-link">
            <p className="quick-link-title">Search</p>
            <p className="quick-link-copy">
              Provider diagnostics. Test search behavior directly and compare mode-specific output.
            </p>
          </Link>
          <Link href="/capabilities" className="quick-link">
            <p className="quick-link-title">Capabilities</p>
            <p className="quick-link-copy">
              Runtime visibility. Check MCP availability, policy snapshot, and actionable dependency issues.
            </p>
          </Link>
          <Link href="/tools" className="quick-link">
            <p className="quick-link-title">Tools</p>
            <p className="quick-link-copy">
              Execution contracts. Inspect registered tools, action classes, and approval requirements.
            </p>
          </Link>
          <Link href="/settings" className="quick-link">
            <p className="quick-link-title">Settings</p>
            <p className="quick-link-copy">
              All configuration in one screen: identity, model defaults, runtime policy, readiness, and integrations.
            </p>
          </Link>
        </div>
      </section>
    </main>
  );
}
