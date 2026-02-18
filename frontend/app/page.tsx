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
    <main style={{ maxWidth: "900px", margin: "48px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>BLAIRE</h1>
      <p style={{ lineHeight: 1.5 }}>
        Frontend scaffold is active. Backend integration endpoints and agent UI will be added next.
      </p>
      <p style={{ marginTop: "8px", fontFamily: "monospace" }}>backend: {backendStatus}</p>
      <p style={{ marginTop: "8px", fontFamily: "monospace" }}>search default: {defaultSearchMode}</p>
      <p style={{ marginTop: "8px", fontFamily: "monospace" }}>general models: {generalModels}</p>
      <p style={{ marginTop: "8px", fontFamily: "monospace" }}>sensitive actions enabled: {sensitiveActions}</p>
      <p style={{ marginTop: "8px", fontFamily: "monospace" }}>dependencies: {dependencySummary}</p>
      <p style={{ marginTop: "16px" }}>
        <a href="/chat">Open chat</a>
      </p>
      <p style={{ marginTop: "8px" }}>
        <a href="/settings">Open settings</a>
      </p>
      <p style={{ marginTop: "8px" }}>
        <a href="/approvals">Open approval queue</a>
      </p>
      <p style={{ marginTop: "8px" }}>
        <a href="/knowledge">Open knowledge status</a>
      </p>
      <p style={{ marginTop: "8px" }}>
        <a href="/swarm">Open swarm panel</a>
      </p>
      <p style={{ marginTop: "8px" }}>
        <a href="/search">Open search</a>
      </p>
    </main>
  );
}
