"use client";

import { FormEvent, useState } from "react";

import { ResearchResponse, runResearch } from "@/lib/api";

export default function SwarmPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await runResearch(query.trim());
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Research failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: "920px", margin: "40px auto", padding: "0 16px" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "12px" }}>Swarm</h1>
      <form onSubmit={onSubmit} style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Research topic..."
          style={{ flex: 1, padding: "10px", border: "1px solid #94a3b8", borderRadius: "6px" }}
        />
        <button type="submit" disabled={loading} style={{ padding: "10px 16px" }}>
          {loading ? "Running..." : "Run Swarm"}
        </button>
      </form>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {result ? (
        <div style={{ border: "1px solid #cbd5e1", borderRadius: "8px", padding: "12px" }}>
          <p>
            <strong>Supervisor:</strong> {result.supervisor_summary}
          </p>
          {result.workers.map((worker) => (
            <div key={worker.worker_id} style={{ marginTop: "10px" }}>
              <p>
                <strong>{worker.worker_id}</strong>: {worker.summary}
              </p>
              <ul>
                {worker.sources.map((source) => (
                  <li key={source}>
                    <a href={source} target="_blank" rel="noreferrer">
                      {source}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      ) : null}
    </main>
  );
}

