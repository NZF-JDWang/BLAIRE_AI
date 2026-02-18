import { getHealth, getRuntimeOptions } from "@/lib/api";

export default async function HomePage() {
  let backendStatus = "unreachable";
  let defaultSearchMode = "unknown";
  let generalModels = "unknown";
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
    };
    defaultSearchMode = options.default_search_mode ?? "unknown";
    generalModels = options.model_allowlist?.general?.join(", ") ?? "none";
  } catch {
    defaultSearchMode = "unavailable";
    generalModels = "unavailable";
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
    </main>
  );
}
