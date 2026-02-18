"use client";

import { FormEvent, useState } from "react";

import { getLiveSwarmRuns, ResearchResponse, runResearch, SwarmLiveResponse } from "@/lib/api";

export default function SwarmPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [live, setLive] = useState<SwarmLiveResponse | null>(null);
  const [error, setError] = useState("");

  async function refreshLive() {
    try {
      setLive(await getLiveSwarmRuns(10));
    } catch {
      setLive(null);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await runResearch(query.trim());
      setResult(response);
      await refreshLive();
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
        <button type="button" onClick={() => void refreshLive()} style={{ padding: "10px 16px" }}>
          Refresh Live
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

      {live && live.runs.length > 0 ? (
        <section style={{ marginTop: "16px", border: "1px solid #cbd5e1", borderRadius: "8px", padding: "12px" }}>
          <h2 style={{ marginTop: 0 }}>Live Swarm Runs</h2>
          {live.runs.map((run) => (
            <div key={run.run_id} style={{ marginBottom: "14px", paddingBottom: "10px", borderBottom: "1px solid #e2e8f0" }}>
              <p>
                <strong>{run.query}</strong> ({new Date(run.created_at).toLocaleString()})
              </p>
              <p style={{ margin: "6px 0" }}>{run.supervisor_summary}</p>
              <div style={{ fontFamily: "monospace", fontSize: "0.85rem" }}>
                {run.trace.map((step, idx) => (
                  <div key={`${run.run_id}-${idx}`}>
                    {step.status}: {step.step}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
      ) : null}
    </main>
  );
}
